/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Dell Confidential.
 * This software contains proprietary and confidential information of Dell Inc.
 * and is subject to applicable non-disclosure agreements.
 * Unauthorized copying, modification, distribution, or use is strictly prohibited.
 */
/**
 * @file rdma_listener.cpp
 * @brief RC-mode RDMA listener thread implementation.
 */

#include "rdma_listener.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <rdma/rdma_verbs.h>

#include "log.h"

/**
 * Signal readiness (or failure) to the waiting main thread.
 * Sets state->error, marks ready=1, and broadcasts the condvar.
 */
static void signal_ready(RdmaListenerState *state, int err) {
    pthread_mutex_lock(&state->ready_mutex);
    state->error = err;
    state->ready = 1;
    pthread_cond_signal(&state->ready_cond);
    pthread_mutex_unlock(&state->ready_mutex);
}

void rdma_listener_state_init(RdmaListenerState *state,
                              const char *bind_ip, int port,
                              std::atomic<bool> *do_work) {
    memset(state, 0, sizeof(*state));
    state->bind_ip  = bind_ip;
    state->port     = port;
    state->do_work  = do_work;
    state->pd       = NULL;
    state->ec       = NULL;
    state->error    = 0;
    state->ready    = 0;
    pthread_mutex_init(&state->ready_mutex, NULL);
    pthread_cond_init(&state->ready_cond, NULL);
}

void rdma_listener_wait_ready(RdmaListenerState *state) {
    pthread_mutex_lock(&state->ready_mutex);
    while (!state->ready) {
        pthread_cond_wait(&state->ready_cond, &state->ready_mutex);
    }
    pthread_mutex_unlock(&state->ready_mutex);
}

void *rdma_listener_thread(void *arg) {
    RdmaListenerState *state = static_cast<RdmaListenerState*>(arg);

    struct rdma_cm_id *listen_id = NULL;
    struct rdma_cm_id *cm_id = NULL;
    struct sockaddr_in addr;
    struct ibv_qp_init_attr qp_attr = {};
    struct rdma_conn_param conn_param = {};
    struct rdma_cm_event *event;
    struct ibv_cq *ib_cq = NULL;

    state->ec = rdma_create_event_channel();
    if (!state->ec) {
        LOG_PERROR("rdma_create_event_channel");
        signal_ready(state, -1);
        return NULL;
    }

    // Make the event channel non-blocking so we can poll with a timeout
    // and periodically check the do_work shutdown flag.
    int ec_flags = fcntl(state->ec->fd, F_GETFL);
    fcntl(state->ec->fd, F_SETFL, ec_flags | O_NONBLOCK);

    if (rdma_create_id(state->ec, &listen_id, NULL, RDMA_PS_TCP)) {
        LOG_PERROR("rdma_create_id");
        rdma_destroy_event_channel(state->ec);
        state->ec = NULL;
        signal_ready(state, -1);
        return NULL;
    }

    // Specify the IP address and port explicitly
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(state->port);
    addr.sin_addr.s_addr = inet_addr(state->bind_ip);

    LOG_INFO("Binding to IP Address: %s", inet_ntoa(addr.sin_addr));
    LOG_INFO("Port: %d", ntohs(addr.sin_port));

    if (rdma_bind_addr(listen_id, (struct sockaddr *)&addr)) {
        LOG_PERROR("rdma_bind_addr");
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(state->ec);
        state->ec = NULL;
        signal_ready(state, -1);
        return NULL;
    }

    // Create a Protection Domain (PD)
    state->pd = ibv_alloc_pd(listen_id->verbs);
    if (!state->pd) {
        LOG_PERROR("ibv_alloc_pd");
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(state->ec);
        state->ec = NULL;
        signal_ready(state, -1);
        return NULL;
    }

    // Listen for incoming connection
    if (rdma_listen(listen_id, 10)) {
        LOG_PERROR("rdma_listen");
        ibv_dealloc_pd(state->pd);
        state->pd = NULL;
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(state->ec);
        state->ec = NULL;
        signal_ready(state, -1);
        return NULL;
    }

    LOG_INFO("Listening for incoming connections on port %d...", state->port);

    // CQ is required: every QP needs a send and recv CQ (assigned below in qp_attr).
    // Size must accommodate the QP's max_send_wr + max_recv_wr.
#ifdef USE_MLX5
    // mlx5 (ConnectX): large QP, so CQ must cover 2048 + 128 = 2176.
    ib_cq = ibv_create_cq(listen_id->verbs, 2176, NULL, NULL, 0);
#else
    // Non-mlx5 (e.g. Broadcom bnxt_re): smaller QP, 128 + 128 = 256.
    ib_cq = ibv_create_cq(listen_id->verbs, 256, NULL, NULL, 0);
#endif
    if (!ib_cq) {
        LOG_PERROR("ibv_create_cq");
        ibv_dealloc_pd(state->pd);
        state->pd = NULL;
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(state->ec);
        state->ec = NULL;
        signal_ready(state, -1);
        return NULL;
    }

    // Log device capabilities so we can verify RDMA READ responder support.
    // A max_qp_rd_atom of 0 means the NIC cannot act as an RDMA READ
    // responder (the PUT path would then fail with IBV_WC_RETRY_EXC_ERR).
    {
        struct ibv_device_attr dev_attr;
        memset(&dev_attr, 0, sizeof(dev_attr));
        if (ibv_query_device(listen_id->verbs, &dev_attr) == 0) {
            LOG_INFO("RDMA device caps: max_qp_rd_atom=%d (max_dest_rd_atomic per QP), "
                     "max_qp_init_rd_atom=%d (max_rd_atomic per QP), "
                     "max_res_rd_atom=%d (total responder resources)",
                     dev_attr.max_qp_rd_atom,
                     dev_attr.max_qp_init_rd_atom,
                     dev_attr.max_res_rd_atom);
        } else {
            LOG_WARN("ibv_query_device failed — cannot log device RDMA READ caps");
        }
    }

    // Signal that the listener is ready (no error)
    signal_ready(state, 0);

    while (state->do_work->load()) {
        // Poll the event channel fd with a 100ms timeout so we can
        // periodically re-check the do_work shutdown flag.
        struct pollfd pfd = {};
        pfd.fd = state->ec->fd;
        pfd.events = POLLIN;

        int poll_ret = poll(&pfd, 1, 100);
        if (poll_ret < 0) {
            if (errno == EINTR) continue;  // interrupted by signal, re-check do_work
            LOG_PERROR("poll");
            break;
        }
        if (poll_ret == 0) continue;  // timeout, re-check do_work

        // Event channel is readable
        if (rdma_get_cm_event(state->ec, &event) != 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            // Real error (e.g. channel destroyed during shutdown)
            if (state->do_work->load()) {
                LOG_PERROR("rdma_get_cm_event");
            }
            break;
        }

        switch(event->event) {
        case RDMA_CM_EVENT_CONNECT_REQUEST: {
            cm_id = event->id;
            rdma_ack_cm_event(event);

            // Create Queue Pair (QP) for connection.
            // This is the passive (responder) side — the endpoint does
            // RDMA READ/WRITE via its own QP.
            memset(&qp_attr, 0, sizeof(qp_attr));
            qp_attr.recv_cq = ib_cq;
            qp_attr.send_cq = ib_cq;
#ifdef USE_MLX5
            // mlx5 (ConnectX) handles large QPs without issue.
            qp_attr.cap.max_send_wr  = 2048;
            qp_attr.cap.max_recv_wr  = 128;
            qp_attr.cap.max_send_sge = 16;
            qp_attr.cap.max_recv_sge = 16;
#else
            // Non-mlx5 (e.g. Broadcom bnxt_re): the driver returns EFAULT
            // when max_send_wr is an exact power of two (128, 256, …) inside
            // Docker containers.  This is a bnxt_re kernel driver bug that
            // does NOT reproduce on the host.  Use 127 to stay just under
            // the boundary while preserving near-identical queue depth.
            qp_attr.cap.max_send_wr  = 127;
            qp_attr.cap.max_recv_wr  = 128;
            qp_attr.cap.max_send_sge = 4;
            qp_attr.cap.max_recv_sge = 4;
#endif
            qp_attr.qp_type = IBV_QPT_RC;

            if (rdma_create_qp(cm_id, state->pd, &qp_attr) == 0) {
                LOG_DEBUG("Queue Pair created successfully %d.", cm_id->qp->state);
            } else {
                LOG_PERROR("rdma_create_qp");
                LOG_ERROR("Failed to create QP for incoming connection, skipping");
                rdma_destroy_id(cm_id);
                break;
            }
            
            // Accept the connection.
            // Raise initiator_depth / responder_resources to 8 to match the
            // server's RC pipeline depth: responder_resources caps the QP's
            // max_dest_rd_atomic, so a value of 1 would limit the server to a
            // single outstanding RDMA READ regardless of its own initiator
            // depth.  8 lets the server pipeline up to 8 concurrent RDMA READs
            // (PUT path) over this QP.
            memset(&conn_param, 0, sizeof(conn_param));
            conn_param.initiator_depth = 8;
            conn_param.responder_resources = 8;
            conn_param.rnr_retry_count = 7;
            conn_param.retry_count = 7;
            
            if (rdma_accept(cm_id, &conn_param) == 0) {
                LOG_DEBUG("Accepted connection from client %d.", cm_id->qp->state);

                // Query the actual QP attributes to verify max_dest_rd_atomic.
                // On Broadcom bnxt_re the driver may silently leave this at 0,
                // which causes the server's RDMA READ to get no response and
                // fail with IBV_WC_RETRY_EXC_ERR.
                {
                    struct ibv_qp_attr accepted_qp_attr;
                    struct ibv_qp_init_attr accepted_init_attr;
                    memset(&accepted_qp_attr, 0, sizeof(accepted_qp_attr));
                    memset(&accepted_init_attr, 0, sizeof(accepted_init_attr));
                    int qp_mask = IBV_QP_STATE | IBV_QP_MAX_DEST_RD_ATOMIC |
                                  IBV_QP_MAX_QP_RD_ATOMIC;
                    if (ibv_query_qp(cm_id->qp, &accepted_qp_attr, qp_mask,
                                     &accepted_init_attr) == 0) {
                        LOG_DEBUG("Accepted QP state=%d, max_dest_rd_atomic=%d "
                                  "(must be >=1 for RDMA READ), max_rd_atomic=%d",
                                  accepted_qp_attr.qp_state,
                                  accepted_qp_attr.max_dest_rd_atomic,
                                  accepted_qp_attr.max_rd_atomic);
                        if (accepted_qp_attr.max_dest_rd_atomic == 0) {
                            LOG_ERROR("max_dest_rd_atomic=0: this QP cannot respond "
                                      "to RDMA READ requests — server will get "
                                      "IBV_WC_RETRY_EXC_ERR. Driver may be ignoring "
                                      "responder_resources=%d.",
                                      conn_param.responder_resources);
                        }
                    } else {
                        LOG_WARN("ibv_query_qp failed — cannot verify max_dest_rd_atomic");
                    }
                }
            } else {
                LOG_PERROR("rdma_accept");
                LOG_ERROR("Failed to accept connection, destroying QP");
                rdma_destroy_qp(cm_id);
                rdma_destroy_id(cm_id);
            }
            break;
        }
        case RDMA_CM_EVENT_ESTABLISHED: {
            LOG_DEBUG("Connection established with client %d.", event->event);
            rdma_ack_cm_event(event);
            break;
        }
        case RDMA_CM_EVENT_DISCONNECTED: {
            LOG_DEBUG("Client disconnected.");
            LOG_DEBUG("Destroying Queue Pair...");
            cm_id = event->id;
            rdma_ack_cm_event(event);
            rdma_destroy_qp(cm_id);

            rdma_destroy_id(cm_id);
            break;

        }
        default: {
            LOG_WARN("Unexpected event during connection request: %d", event->event);
            rdma_ack_cm_event(event);
            break;
        }
        }
    }

    LOG_INFO("Shutting down RDMA listener...");
    rdma_destroy_id(listen_id);
    ibv_destroy_cq(ib_cq);
    rdma_destroy_event_channel(state->ec);
    ibv_dealloc_pd(state->pd);
    LOG_INFO("RDMA listener resources cleaned up.");

    return NULL;
}

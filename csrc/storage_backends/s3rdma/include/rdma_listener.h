/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Dell Confidential.
 * This software contains proprietary and confidential information of Dell Inc.
 * and is subject to applicable non-disclosure agreements.
 * Unauthorized copying, modification, distribution, or use is strictly prohibited.
 */
#ifndef S3RDMA_RDMA_LISTENER_H
#define S3RDMA_RDMA_LISTENER_H

/**
 * @file rdma_listener.h
 * @brief RC-mode RDMA listener thread for the S3 RDMA client.
 *
 * Manages the RDMA CM event channel, accepts incoming RC connections,
 * and creates QPs for each connecting peer.  All mutable state lives
 * in RdmaListenerState (no file-scope globals).
 */

#include <atomic>
#include <pthread.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

/**
 * Self-contained state for the RDMA listener thread.
 */
struct RdmaListenerState {
    // ── Inputs (set before thread launch) ──
    const char         *bind_ip;   // IP to bind the RC listener
    int                 port;      // RDMA port (e.g. 7471)
    std::atomic<bool>  *do_work;   // Points to shared shutdown flag (atomic)

    // ── Outputs (set by the listener thread) ──
    struct ibv_pd             *pd;  // Protection Domain (created during bind)
    struct rdma_event_channel *ec;  // CM event channel
    int                        error; // 0 = ok, non-zero = setup failed

    // ── Ready synchronisation ──
    pthread_mutex_t ready_mutex;
    pthread_cond_t  ready_cond;
    int             ready;       // 1 once pd is valid and listener is accepting
};

/**
 * Initialise a RdmaListenerState with the given parameters.
 * Must be called before launching the listener thread.
 */
void rdma_listener_state_init(RdmaListenerState *state,
                              const char *bind_ip, int port,
                              std::atomic<bool> *do_work);

/**
 * RDMA listener thread entry point.
 *
 * @param arg  Pointer to a fully-initialised RdmaListenerState.
 * @return NULL.
 */
void *rdma_listener_thread(void *arg);

/**
 * Block the caller until the listener thread signals readiness
 * (i.e. the PD is allocated and the listener is accepting connections).
 */
void rdma_listener_wait_ready(RdmaListenerState *state);

#endif /* S3RDMA_RDMA_LISTENER_H */

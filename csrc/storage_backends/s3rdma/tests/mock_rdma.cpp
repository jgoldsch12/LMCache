/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file mock_rdma.cpp
 * @brief Mock implementations of vendored RDMA functions for unit testing.
 *
 * Replaces gpu_direct_rdma_access.cpp and rdma_listener.cpp so that
 * S3RdmaClient can be tested without RDMA hardware or kernel modules.
 *
 * Mock rdma_device and rdma_buffer structs mirror the real layout enough
 * for S3RdmaClient to work correctly (it accesses buf_addr, buf_size,
 * rkey, rdma_dev->lid, rdma_dev->is_global, rdma_dev->gid, rdma_dev->tqp).
 */

#include "mock_rdma.h"
#include "rdma_listener.h"

#include <arpa/inet.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

#include <infiniband/verbs.h>

/* We need the struct definitions to match what s3rdma_client.cpp expects
 * when it dereferences rdma_buffer->buf_addr, rdma_buffer->rkey, etc.
 * The real structs are defined in gpu_direct_rdma_access.cpp (not the header),
 * so we define compatible versions here. */

#define QP_PER_THREAD 8
#define SEND_Q_DEPTH 640

struct wr_id_reported {
    uint64_t wr_id;
    uint16_t num_wrs;
    uint16_t flags;
};

struct rdma_device {
    struct rdma_event_channel *cm_channel;
    struct rdma_cm_id *cm_id;

    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_srq *srq;
    struct ibv_qp *qp;
    struct ibv_qp *tqp[QP_PER_THREAD];
    struct ibv_qp_ex *qpex;

    int ib_port;
    int is_global;
    int gidx;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;

    struct wr_id_reported app_wr_id[SEND_Q_DEPTH];
    int app_wr_id_idx;
    int qp_available_wr;
    int rdma_buff_cnt;
};

struct rdma_buffer {
    void *buf_addr;
    size_t buf_size;
    struct ibv_mr *mr;
    uint32_t rkey;
    struct rdma_device *rdma_dev;
    int dmabuf_fd;
};

/* ── Mock state ── */

static bool g_open_device_fail = false;
static bool g_reg_fail = false;
static bool g_listener_fail = false;
static bool g_desc_fail = false;
static int  g_dmabuf_export_result = -1;  /* default: not available */

static std::atomic<size_t> g_reg_count{0};
static std::atomic<size_t> g_dereg_count{0};
static std::atomic<size_t> g_desc_count{0};
static std::atomic<size_t> g_close_device_count{0};

static std::atomic<void *> g_last_reg_addr{nullptr};
static std::atomic<size_t> g_last_reg_size{0};

/* Template for mock device fields */
static struct rdma_device g_mock_device_template;

/* Helper: allocate a heap copy of the mock device template.
 * Callers (rdma_open_device_client, rdma_get_device) return heap pointers
 * because S3RdmaClient::close() calls free(device_) for RC mode. */
static struct rdma_device *alloc_mock_device() {
    struct rdma_device *dev =
        (struct rdma_device *)calloc(1, sizeof(struct rdma_device));
    memcpy(dev, &g_mock_device_template, sizeof(struct rdma_device));
    return dev;
}

void mock_rdma_reset() {
    g_open_device_fail = false;
    g_reg_fail = false;
    g_listener_fail = false;
    g_desc_fail = false;
    g_dmabuf_export_result = -1;
    g_reg_count = 0;
    g_dereg_count = 0;
    g_desc_count = 0;
    g_close_device_count = 0;
    g_last_reg_addr.store(nullptr);
    g_last_reg_size.store(0);

    memset(&g_mock_device_template, 0, sizeof(g_mock_device_template));
    g_mock_device_template.lid = 0x0001;
    g_mock_device_template.is_global = 1;
    /* Set a recognisable GID: ::ffff:10.246.191.101 (0a.f6.bf.65) */
    memset(&g_mock_device_template.gid, 0, sizeof(g_mock_device_template.gid));
    g_mock_device_template.gid.raw[10] = 0xff;
    g_mock_device_template.gid.raw[11] = 0xff;
    g_mock_device_template.gid.raw[12] = 0x0a;
    g_mock_device_template.gid.raw[13] = 0xf6;
    g_mock_device_template.gid.raw[14] = 0xbf;
    g_mock_device_template.gid.raw[15] = 0x65;
}

void mock_rdma_set_open_device_fail(bool fail) { g_open_device_fail = fail; }
void mock_rdma_set_reg_fail(bool fail) { g_reg_fail = fail; }
void mock_rdma_set_listener_fail(bool fail) { g_listener_fail = fail; }
void mock_rdma_set_desc_fail(bool fail) { g_desc_fail = fail; }
void mock_rdma_set_dmabuf_export_result(int result) {
    g_dmabuf_export_result = result;
}

size_t mock_rdma_get_reg_count() { return g_reg_count.load(); }
size_t mock_rdma_get_dereg_count() { return g_dereg_count.load(); }
size_t mock_rdma_get_desc_count() { return g_desc_count.load(); }
size_t mock_rdma_get_close_device_count() { return g_close_device_count.load(); }

void *mock_rdma_last_reg_addr() { return g_last_reg_addr.load(); }
size_t mock_rdma_last_reg_size() { return g_last_reg_size.load(); }

/* ── Mock RDMA functions (extern "C" to match gpu_direct_rdma_access.h) ── */

extern "C" {

struct rdma_device *rdma_open_device_client(struct sockaddr *addr) {
    (void)addr;
    if (g_open_device_fail) return NULL;
    return alloc_mock_device();
}

struct rdma_device *rdma_open_device_server(struct sockaddr *addr) {
    (void)addr;
    if (g_open_device_fail) return NULL;
    return alloc_mock_device();
}

struct rdma_device *rdma_get_device(struct ibv_pd *pd, struct sockaddr *addr) {
    (void)pd;
    (void)addr;
    if (g_open_device_fail) return NULL;
    return alloc_mock_device();
}

struct rdma_buffer *rdma_buffer_reg(struct rdma_device *device,
                                    void *addr, size_t length) {
    if (g_reg_fail) return NULL;

    uint32_t idx = static_cast<uint32_t>(g_reg_count.fetch_add(1));

    struct rdma_buffer *buf =
        (struct rdma_buffer *)calloc(1, sizeof(struct rdma_buffer));
    buf->buf_addr = addr;
    buf->buf_size = length;
    buf->rkey = 0xABCD0000 + idx;
    buf->rdma_dev = device;
    buf->dmabuf_fd = -1;

    g_last_reg_addr.store(addr);
    g_last_reg_size.store(length);
    return buf;
}

struct rdma_buffer *rdma_buffer_reg_dmabuf(struct rdma_device *device,
                                           void *addr, size_t length,
                                           int dmabuf_fd,
                                           uint64_t dmabuf_offset) {
    (void)dmabuf_offset;
    if (g_reg_fail) return NULL;

    uint32_t idx = static_cast<uint32_t>(g_reg_count.fetch_add(1));

    struct rdma_buffer *buf =
        (struct rdma_buffer *)calloc(1, sizeof(struct rdma_buffer));
    buf->buf_addr = addr;
    buf->buf_size = length;
    buf->rkey = 0xD0AB0000 + idx;
    buf->rdma_dev = device;
    buf->dmabuf_fd = dmabuf_fd;

    g_last_reg_addr.store(addr);
    g_last_reg_size.store(length);
    return buf;
}

void rdma_buffer_dereg(struct rdma_buffer *buffer) {
    if (buffer) {
        free(buffer);
        g_dereg_count++;
    }
}

void rdma_close_device(struct rdma_device *device) {
    (void)device;
    g_close_device_count++;
}

int rdma_reset_device(struct rdma_device *device) {
    (void)device;
    return 0;
}

/*
 * Generate a descriptor string matching the real format:
 * addr:size:rkey:lid:dctn:gid_flag:gid
 *
 * This is a faithful reimplementation of the format logic from
 * gpu_direct_rdma_access.cpp so descriptor tests validate correctly.
 */
static void gid_to_wire_gid(const union ibv_gid *gid,
                            char *wgid, size_t remaining) {
    const uint32_t *raw = (const uint32_t *)gid->raw;
    for (int i = 0; i < 4; ++i)
        snprintf(&wgid[i * 8], remaining - (size_t)(i * 8),
                 "%08x", htonl(raw[i]));
}

int rdma_buffer_get_desc_str(struct rdma_buffer *rdma_buff,
                             char *desc_str, size_t desc_length,
                             int qp_idx) {
    /* Injected failure: simulate descriptor generation error */
    if (g_desc_fail) return 0;

    /* Descriptor = 49-char prefix + 32-char GID + null = 82 bytes minimum */
    if (desc_length < 82) return 0;

    snprintf(desc_str, desc_length, "%016llx:%08lx:%08x:%04x:%06x:%d:",
             (unsigned long long)rdma_buff->buf_addr,
             (unsigned long)rdma_buff->buf_size,
             rdma_buff->rkey,
             rdma_buff->rdma_dev->lid,
             (rdma_buff->rdma_dev->tqp[qp_idx])
                 ? rdma_buff->rdma_dev->tqp[qp_idx]->qp_num
                 : 0,
             rdma_buff->rdma_dev->is_global & 0x1);

    /* Append GID after the fixed-length prefix
     * "0102030405060708:01020304:01020304:0102:010203:1:" = 49 chars */
    size_t prefix_len = strlen(desc_str);
    gid_to_wire_gid(&rdma_buff->rdma_dev->gid,
                    desc_str + prefix_len, desc_length - prefix_len);

    g_desc_count++;
    return (int)(strlen(desc_str) + 1);
}

int rdma_submit_task(struct rdma_task_attr *attr) {
    (void)attr;
    return 0;
}

int rdma_poll_completions(struct rdma_device *device,
                          struct rdma_completion_event *event,
                          uint32_t num_entries) {
    (void)device;
    (void)event;
    (void)num_entries;
    return 0;
}

}  /* extern "C" */

/* ── Mock listener functions (C++ linkage, matching rdma_listener.h) ── */

void rdma_listener_state_init(RdmaListenerState *state,
                              const char *bind_ip, int port,
                              std::atomic<bool> *do_work) {
    state->bind_ip = bind_ip;
    state->port = port;
    state->do_work = do_work;
    state->pd = nullptr;
    state->ec = nullptr;
    state->error = g_listener_fail ? 1 : 0;
    state->ready = 0;
    pthread_mutex_init(&state->ready_mutex, nullptr);
    pthread_cond_init(&state->ready_cond, nullptr);
}

void *rdma_listener_thread(void *arg) {
    auto *state = static_cast<RdmaListenerState *>(arg);

    /* Simulate immediate readiness (or failure) */
    pthread_mutex_lock(&state->ready_mutex);
    if (!g_listener_fail) {
        /* Allocate a fake PD so rdma_get_device has something non-null */
        state->pd = reinterpret_cast<struct ibv_pd *>(0xFEEDFACE);
    }
    state->ready = 1;
    pthread_cond_signal(&state->ready_cond);
    pthread_mutex_unlock(&state->ready_mutex);

    /* Spin until told to stop */
    while (state->do_work->load()) {
        struct timespec ts = {0, 10000000}; /* 10ms */
        nanosleep(&ts, nullptr);
    }

    /* Clean up pthread primitives initialised in rdma_listener_state_init.
     *
     * Safety: by this point rdma_listener_wait_ready() has long since
     * returned (it is called in S3RdmaClient's constructor, before any
     * user operation), and S3RdmaClient::close() has set do_work=false
     * and is about to join() this thread.  No other code path touches
     * ready_mutex or ready_cond after the initial wait, so destroying
     * them here — just before the thread returns — is safe.
     *
     * Ideally the caller would destroy them after join(), but the mock
     * does not control stop_rc_listener(). */
    pthread_mutex_destroy(&state->ready_mutex);
    pthread_cond_destroy(&state->ready_cond);

    return nullptr;
}

void rdma_listener_wait_ready(RdmaListenerState *state) {
    pthread_mutex_lock(&state->ready_mutex);
    while (!state->ready) {
        pthread_cond_wait(&state->ready_cond, &state->ready_mutex);
    }
    pthread_mutex_unlock(&state->ready_mutex);
}

/* ── Mock gpuExportDmaBuf (C++ linkage, matching gpu_compat.h) ── */

int gpuExportDmaBuf(const void *ptr, size_t size,
                    int *dmabuf_fd, uint64_t *offset) {
    (void)ptr;
    (void)size;
    if (g_dmabuf_export_result == 0) {
        *dmabuf_fd = 42;   /* fake file descriptor */
        *offset = 0;
    }
    return g_dmabuf_export_result;
}

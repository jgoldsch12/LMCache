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
 * @file s3rdma_client.cpp
 * @brief S3RdmaClient implementation — wraps vendored RDMA functions.
 */

#include "s3rdma_client.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
#include "gpu_direct_rdma_access.h"
}
#include "gpu_compat.h"
#include "rdma_listener.h"
#include "log.h"

namespace lmcache {
namespace s3rdma {

// ── Constructor ──

S3RdmaClient::S3RdmaClient(const S3RdmaConfig& config)
    : config_(config)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(0);  // ephemeral port for device discovery
    addr.sin_addr.s_addr = inet_addr(config_.local_ip.c_str());

    if (addr.sin_addr.s_addr == INADDR_NONE) {
        throw std::runtime_error(
            "S3RdmaClient: invalid RDMA NIC IP address: " + config_.local_ip);
    }

    if (config_.connection_mode == "RC") {
        // RC mode: start listener thread, use its PD for device
        start_rc_listener();
        device_ = rdma_get_device(listener_state_->pd,
                                  reinterpret_cast<struct sockaddr*>(&addr));
    } else {
        // DC mode: open device as client (creates DCT QP)
        device_ = rdma_open_device_client(
            reinterpret_cast<struct sockaddr*>(&addr));
    }

    if (!device_) {
        stop_rc_listener();
        throw std::runtime_error(
            "S3RdmaClient: failed to open RDMA device on " + config_.local_ip);
    }

    LOG_INFO("S3RdmaClient: opened RDMA device on %s (mode=%s, port=%d)",
             config_.local_ip.c_str(), config_.connection_mode.c_str(),
             config_.rdma_port);
}

// ── Destructor ──

S3RdmaClient::~S3RdmaClient() {
    close();
}

// ── Memory Registration ──

std::pair<uintptr_t, size_t>
S3RdmaClient::register_region(uintptr_t ptr, size_t size, bool is_gpu) {
    if (!device_) {
        throw std::runtime_error("S3RdmaClient: device not open");
    }

    struct rdma_buffer* mr = nullptr;

#ifdef GPU_HAS_DMABUF_EXPORT
    if (is_gpu) {
        // AMD HIP path: export GPU memory as dma-buf, then register
        int dmabuf_fd = -1;
        uint64_t dmabuf_offset = 0;
        int rc = gpuExportDmaBuf(reinterpret_cast<const void*>(ptr), size,
                                  &dmabuf_fd, &dmabuf_offset);
        if (rc != 0) {
            throw std::runtime_error(
                "S3RdmaClient: gpuExportDmaBuf failed for GPU memory");
        }
        mr = rdma_buffer_reg_dmabuf(device_, reinterpret_cast<void*>(ptr),
                                     size, dmabuf_fd, dmabuf_offset);
    } else {
        mr = rdma_buffer_reg(device_, reinterpret_cast<void*>(ptr), size);
    }
#else
    // NVIDIA CUDA path: ibv_reg_mr works for both CPU and GPU memory
    // (GPU memory registration is transparent via nvidia-peermem kernel module)
    (void)is_gpu;
    mr = rdma_buffer_reg(device_, reinterpret_cast<void*>(ptr), size);
#endif

    if (!mr) {
        throw std::runtime_error(
            "S3RdmaClient: rdma_buffer_reg failed for addr=" +
            std::to_string(ptr) + " size=" + std::to_string(size));
    }

    {
        std::lock_guard<std::mutex> lock(mrs_mutex_);
        mrs_[ptr] = mr;
    }

    LOG_DEBUG("S3RdmaClient: registered MR at %p, size=%zu, is_gpu=%d",
              reinterpret_cast<void*>(ptr), size, is_gpu);

    return {ptr, size};
}

size_t S3RdmaClient::register_pages(uintptr_t base, size_t total_size,
                                     size_t page_size, bool is_gpu) {
    if (page_size == 0) {
        throw std::runtime_error("S3RdmaClient: page_size must be > 0");
    }

    size_t n_pages = total_size / page_size;
    for (size_t i = 0; i < n_pages; ++i) {
        uintptr_t page_ptr = base + i * page_size;
        register_region(page_ptr, page_size, is_gpu);
    }

    LOG_INFO("S3RdmaClient: registered %zu pages (page_size=%zu, total=%zu, is_gpu=%d)",
             n_pages, page_size, total_size, is_gpu);

    return n_pages;
}

int S3RdmaClient::deregister_region(uintptr_t ptr) noexcept {
    std::lock_guard<std::mutex> lock(mrs_mutex_);
    auto it = mrs_.find(ptr);
    if (it == mrs_.end()) {
        return S3RDMA_FAIL;
    }
    rdma_buffer_dereg(it->second);
    mrs_.erase(it);
    return S3RDMA_SUCCESS;
}

void S3RdmaClient::deregister_all() noexcept {
    std::lock_guard<std::mutex> lock(mrs_mutex_);
    for (auto& [ptr, mr] : mrs_) {
        rdma_buffer_dereg(mr);
    }
    mrs_.clear();
}

// ── Descriptor Generation ──

std::string S3RdmaClient::prepare_put(uintptr_t data_ptr, size_t size) {
    (void)size;  // Size is embedded in the MR descriptor
    struct rdma_buffer* mr = find_mr(data_ptr);
    if (!mr) {
        throw std::runtime_error(
            "S3RdmaClient::prepare_put: address " +
            std::to_string(data_ptr) + " not in any registered region");
    }
    return get_descriptor(mr);
}

std::string S3RdmaClient::prepare_get(uintptr_t data_ptr, size_t size) {
    (void)size;
    struct rdma_buffer* mr = find_mr(data_ptr);
    if (!mr) {
        throw std::runtime_error(
            "S3RdmaClient::prepare_get: address " +
            std::to_string(data_ptr) + " not in any registered region");
    }
    return get_descriptor(mr);
}

// ── Status ──

bool S3RdmaClient::is_connected() const {
    return device_ != nullptr && !closed_;
}

// ── Lifecycle ──

int S3RdmaClient::close() noexcept {
    if (closed_) return S3RDMA_SUCCESS;
    closed_ = true;

    LOG_INFO("S3RdmaClient: closing...");

    // Deregister all MRs
    deregister_all();

    // Stop RC listener (releases PD, CQ, event channel)
    stop_rc_listener();

    // Close RDMA device.
    // For RC mode, the device was created via rdma_get_device() which has
    // a borrowed PD (owned by the listener, already freed above).  We must
    // NOT call rdma_close_device() because it would try to dealloc the PD
    // and destroy QPs/CQs that were never created.  Just free the struct;
    // the CM resources (from open_ib_device_by_addr) are lightweight and
    // will be reclaimed on process exit.
    if (device_) {
        if (config_.connection_mode == "RC") {
            free(device_);
        } else {
            rdma_close_device(device_);
        }
        device_ = nullptr;
    }

    LOG_INFO("S3RdmaClient: closed.");
    return S3RDMA_SUCCESS;
}

// ── Private Helpers ──

struct rdma_buffer* S3RdmaClient::find_mr(uintptr_t ptr) const {
    std::lock_guard<std::mutex> lock(mrs_mutex_);
    auto it = mrs_.find(ptr);
    if (it != mrs_.end()) {
        return it->second;
    }
    return nullptr;
}

std::string S3RdmaClient::get_descriptor(struct rdma_buffer* mr) const {
    // rdma_buffer_get_desc_str generates the full descriptor string:
    // addr:size:rkey:lid:dctn:gid_flag:gid
    char desc[256];
    int len = rdma_buffer_get_desc_str(mr, desc, sizeof(desc), 0 /*qp_idx*/);
    if (len <= 0) {
        throw std::runtime_error("S3RdmaClient: rdma_buffer_get_desc_str failed");
    }
    return std::string(desc);
}

void S3RdmaClient::start_rc_listener() {
    if (config_.connection_mode != "RC") return;

    listener_state_ = new RdmaListenerState();
    do_work_.store(true);

    rdma_listener_state_init(listener_state_,
                             config_.local_ip.c_str(),
                             config_.rdma_port,
                             &do_work_);

    listener_thread_ = std::thread([this]() {
        rdma_listener_thread(static_cast<void*>(listener_state_));
    });

    // Wait for the listener to be ready (PD allocated, accepting connections)
    rdma_listener_wait_ready(listener_state_);

    if (listener_state_->error != 0) {
        do_work_.store(false);
        if (listener_thread_.joinable()) {
            listener_thread_.join();
        }
        delete listener_state_;
        listener_state_ = nullptr;
        throw std::runtime_error(
            "S3RdmaClient: RC listener failed to start on " +
            config_.local_ip + ":" + std::to_string(config_.rdma_port));
    }

    LOG_INFO("S3RdmaClient: RC listener started on %s:%d",
             config_.local_ip.c_str(), config_.rdma_port);
}

void S3RdmaClient::stop_rc_listener() {
    if (!listener_state_) return;

    LOG_INFO("S3RdmaClient: stopping RC listener...");
    do_work_.store(false);

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

    delete listener_state_;
    listener_state_ = nullptr;
    LOG_INFO("S3RdmaClient: RC listener stopped.");
}

}  // namespace s3rdma
}  // namespace lmcache

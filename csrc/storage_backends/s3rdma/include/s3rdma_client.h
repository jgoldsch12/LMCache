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
 * @file s3rdma_client.h
 * @brief RDMA-accelerated S3 client for LMCache storage backend.
 *
 * Wraps the vendored s3rdma_client RDMA functions (gpu_direct_rdma_access,
 * rdma_listener) into a C++ class with per-page memory registration,
 * RDMA descriptor generation for x-rdma-info headers, and RC listener
 * lifecycle management.
 *
 * Supports both NVIDIA (via nvidia-peermem) and AMD (via dma-buf export)
 * GPUs, and both mlx5 (DC+RC) and non-mlx5 (RC only) NICs.
 *
 * Design ref: docs/rdma-accelerated-s3-connector-design.md
 */

#ifndef LMCACHE_S3RDMA_CLIENT_H
#define LMCACHE_S3RDMA_CLIENT_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

// Forward declarations for vendored RDMA types
struct rdma_device;
struct rdma_buffer;
struct RdmaListenerState;

namespace lmcache {
namespace s3rdma {

// Return codes
constexpr int S3RDMA_SUCCESS = 0;
constexpr int S3RDMA_FAIL = 1;

// Memory type constants (match gpu_direct_rdma_access conventions)
constexpr int S3RDMA_MEMORY_SYSTEM = 0;
constexpr int S3RDMA_MEMORY_GPU = 2;

// Maximum memory region size (4 GiB, limited by some HCA MR tables)
constexpr size_t S3RDMA_MAX_MEMORY_REG_SIZE = 4ULL * 1024 * 1024 * 1024;

/**
 * Configuration for S3RdmaClient.
 */
struct S3RdmaConfig {
    std::string local_ip;                  // RDMA NIC IP address
    int         rdma_port      = 7471;     // RC listener port
    std::string connection_mode = "RC";    // "RC" or "DC"
};

/**
 * RDMA client for S3 storage with x-rdma-info header protocol.
 *
 * Lifecycle:
 *   1. Construct with config (opens RDMA device, starts RC listener if RC mode)
 *   2. register_pages() — register bounce buffer pages at init time
 *   3. prepare_put() / prepare_get() — generate x-rdma-info descriptors per request
 *   4. close() or destructor — deregister all MRs, stop listener, release device
 *
 * Thread safety: All public methods are safe to call from any thread.
 * The GIL is released by the pybind11 layer.
 */
class S3RdmaClient {
public:
    /**
     * Open RDMA device and optionally start RC listener.
     *
     * @param config  RDMA configuration (NIC IP, port, transport mode)
     * @throws std::runtime_error if RDMA device cannot be opened
     */
    explicit S3RdmaClient(const S3RdmaConfig& config);

    ~S3RdmaClient();

    // Non-copyable, non-movable
    S3RdmaClient(const S3RdmaClient&) = delete;
    S3RdmaClient& operator=(const S3RdmaClient&) = delete;

    /**
     * Register a contiguous memory region for RDMA.
     *
     * For GPU memory on NVIDIA, uses ibv_reg_mr (transparent via nvidia-peermem).
     * For GPU memory on AMD, uses dma-buf export + ibv_reg_dmabuf_mr.
     * For CPU memory, uses ibv_reg_mr.
     *
     * @param ptr    Virtual address of the buffer
     * @param size   Size in bytes
     * @param is_gpu True if the memory is GPU-allocated
     * @return (ptr, size) pair serving as registration handle
     * @throws std::runtime_error on registration failure
     */
    std::pair<uintptr_t, size_t> register_region(uintptr_t ptr, size_t size,
                                                  bool is_gpu);

    /**
     * Register per-page MRs covering a contiguous buffer.
     *
     * Matches NixlStorageBackend's init_mem_handlers() pattern: one MR per
     * page-sized chunk. This avoids per-MR size limits and eliminates the
     * need for sub-region offset arithmetic in descriptors.
     *
     * @param base       Base address of the buffer
     * @param total_size Total buffer size in bytes
     * @param page_size  Size of each page (e.g., kv_shape.numel() * dtype.itemsize)
     * @param is_gpu     True if the memory is GPU-allocated
     * @return Number of pages registered
     */
    size_t register_pages(uintptr_t base, size_t total_size,
                          size_t page_size, bool is_gpu);

    /**
     * Deregister a previously registered memory region.
     *
     * @param ptr  Base address used during registration
     * @return S3RDMA_SUCCESS or S3RDMA_FAIL
     */
    int deregister_region(uintptr_t ptr) noexcept;

    /**
     * Deregister all registered memory regions.
     */
    void deregister_all() noexcept;

    /**
     * Generate x-rdma-info descriptor string for a PUT operation.
     *
     * The descriptor contains: addr:size:rkey:lid:dctn:gid_flag:gid
     * This is passed as the x-rdma-info HTTP header value.
     * The S3 server (ObjectScale) uses this to RDMA READ from our buffer.
     *
     * @param data_ptr  Address within a registered page
     * @param size      Transfer size in bytes
     * @return Descriptor string for the x-rdma-info header
     * @throws std::runtime_error if data_ptr is not in any registered region
     */
    std::string prepare_put(uintptr_t data_ptr, size_t size);

    /**
     * Generate x-rdma-info descriptor string for a GET operation.
     *
     * Same format as prepare_put — ObjectScale uses this to RDMA WRITE
     * into our buffer.
     *
     * @param data_ptr  Address within a registered page
     * @param size      Transfer size in bytes
     * @return Descriptor string for the x-rdma-info header
     */
    std::string prepare_get(uintptr_t data_ptr, size_t size);

    /**
     * Check if the RDMA device is ready for operations.
     */
    bool is_connected() const;

    /**
     * Release all RDMA resources. Safe to call multiple times.
     *
     * @return S3RDMA_SUCCESS or S3RDMA_FAIL
     */
    int close() noexcept;

private:
    // RDMA device (opened via rdma_open_device_client or rdma_get_device)
    struct rdma_device* device_ = nullptr;

    // Map of registered MRs: base_ptr -> rdma_buffer*
    std::unordered_map<uintptr_t, struct rdma_buffer*> mrs_;

    // RC listener state (only used in RC mode)
    struct RdmaListenerState* listener_state_ = nullptr;
    std::thread listener_thread_;
    std::atomic<bool> do_work_{true};

    // Configuration
    S3RdmaConfig config_;
    bool closed_ = false;

    // Thread safety for MR map
    mutable std::mutex mrs_mutex_;

    // Helper: find the MR covering a given address
    struct rdma_buffer* find_mr(uintptr_t ptr) const;

    // Helper: generate descriptor string from an MR
    std::string get_descriptor(struct rdma_buffer* mr) const;

    // Helper: start RC listener thread
    void start_rc_listener();

    // Helper: stop RC listener thread
    void stop_rc_listener();
};

}  // namespace s3rdma
}  // namespace lmcache

#endif  // LMCACHE_S3RDMA_CLIENT_H

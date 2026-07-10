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
 * @file mock_rdma.h
 * @brief Mock RDMA layer for unit testing S3RdmaClient without hardware.
 *
 * Provides controllable mock implementations of gpu_direct_rdma_access and
 * rdma_listener functions. Tests can configure mock behavior (e.g., force
 * failures) and inspect call history.
 */

#ifndef S3RDMA_MOCK_RDMA_H
#define S3RDMA_MOCK_RDMA_H

#include <atomic>
#include <cstddef>
#include <cstdint>

/**
 * Reset all mock state to defaults (all operations succeed).
 * Call this in test SetUp().
 */
void mock_rdma_reset();

/* ── Failure injection ── */

/** Make rdma_open_device_client() return NULL. */
void mock_rdma_set_open_device_fail(bool fail);

/** Make rdma_buffer_reg() return NULL. */
void mock_rdma_set_reg_fail(bool fail);

/** Make the RC listener report an error. */
void mock_rdma_set_listener_fail(bool fail);

/** Make rdma_buffer_get_desc_str() return 0 (descriptor generation failure). */
void mock_rdma_set_desc_fail(bool fail);

/**
 * Control the return value of gpuExportDmaBuf().
 * Default (-1) = export not available.  Set to 0 for success.
 * Only relevant when s3rdma_client.cpp is compiled with GPU_HAS_DMABUF_EXPORT.
 */
void mock_rdma_set_dmabuf_export_result(int result);

/* ── Call counters ── */

size_t mock_rdma_get_reg_count();
size_t mock_rdma_get_dereg_count();
size_t mock_rdma_get_desc_count();
size_t mock_rdma_get_close_device_count();

/* ── Last registered buffer info ── */

void*  mock_rdma_last_reg_addr();
size_t mock_rdma_last_reg_size();

#endif /* S3RDMA_MOCK_RDMA_H */

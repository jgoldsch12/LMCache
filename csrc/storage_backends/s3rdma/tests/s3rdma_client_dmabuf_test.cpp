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
 * @file s3rdma_client_dmabuf_test.cpp
 * @brief Tests for the GPU_HAS_DMABUF_EXPORT (AMD dma-buf) registration path.
 *
 * This test executable links against s3rdma_client compiled WITH
 * GPU_HAS_DMABUF_EXPORT defined.  When register_region() is called with
 * is_gpu=true, the code calls gpuExportDmaBuf() first, then
 * rdma_buffer_reg_dmabuf() instead of rdma_buffer_reg().
 *
 * The mock's gpuExportDmaBuf() is controllable via
 * mock_rdma_set_dmabuf_export_result() (default -1 = fail, 0 = success).
 */

#include <gtest/gtest.h>

#include "s3rdma_client.h"
#include "mock_rdma.h"

#include <cstdint>
#include <cstring>
#include <string>

using lmcache::s3rdma::S3RdmaClient;
using lmcache::s3rdma::S3RdmaConfig;

class S3RdmaDmabufTest : public ::testing::Test {
protected:
    void SetUp() override { mock_rdma_reset(); }

    static S3RdmaConfig make_dc_config(const std::string& ip = "10.10.0.1") {
        return S3RdmaConfig{ip, 7471, "DC"};
    }

    static uint8_t* alloc_buffer(size_t size) {
        auto* buf = static_cast<uint8_t*>(malloc(size));
        EXPECT_NE(buf, nullptr);
        memset(buf, 0xBB, size);
        return buf;
    }
};

/* ── GPU registration via dma-buf (gpuExportDmaBuf succeeds) ── */

TEST_F(S3RdmaDmabufTest, RegisterGPU_DmabufSuccess) {
    mock_rdma_set_dmabuf_export_result(0);  /* export succeeds */

    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    auto [ptr, sz] = client.register_region(
        reinterpret_cast<uintptr_t>(buf), 4096, true /* is_gpu */);

    EXPECT_EQ(ptr, reinterpret_cast<uintptr_t>(buf));
    EXPECT_EQ(sz, 4096u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 1u);

    /* Should produce a valid descriptor */
    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);
    EXPECT_FALSE(desc.empty());

    free(buf);
}

TEST_F(S3RdmaDmabufTest, RegisterGPU_DmabufSuccess_Descriptor) {
    mock_rdma_set_dmabuf_export_result(0);

    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(
        reinterpret_cast<uintptr_t>(buf), 4096, true /* is_gpu */);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);
    EXPECT_EQ(mock_rdma_get_desc_count(), 1u);

    /* Descriptor should still have 7 colon-separated fields */
    size_t colon_count = 0;
    for (char c : desc) {
        if (c == ':') colon_count++;
    }
    EXPECT_EQ(colon_count, 6u);

    free(buf);
}

/* ── GPU registration via dma-buf (gpuExportDmaBuf fails) ── */

TEST_F(S3RdmaDmabufTest, RegisterGPU_DmabufExportFails_Throws) {
    /* Default: dmabuf export returns -1 (not available) */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    EXPECT_THROW(
        client.register_region(
            reinterpret_cast<uintptr_t>(buf), 4096, true /* is_gpu */),
        std::runtime_error);

    /* No RDMA registration should have happened */
    EXPECT_EQ(mock_rdma_get_reg_count(), 0u);

    free(buf);
}

/* ── CPU registration (is_gpu=false) still uses rdma_buffer_reg ── */

TEST_F(S3RdmaDmabufTest, RegisterCPU_BypassesDmabuf) {
    /* Even with dmabuf export set to fail, CPU registration should succeed
     * because the dmabuf path is only taken when is_gpu=true */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    auto [ptr, sz] = client.register_region(
        reinterpret_cast<uintptr_t>(buf), 4096, false /* is_gpu */);

    EXPECT_EQ(ptr, reinterpret_cast<uintptr_t>(buf));
    EXPECT_EQ(sz, 4096u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 1u);

    free(buf);
}

/* ── GPU dma-buf registration failure after rdma_buffer_reg_dmabuf fails ── */

TEST_F(S3RdmaDmabufTest, RegisterGPU_DmabufOk_RdmaRegFails_Throws) {
    mock_rdma_set_dmabuf_export_result(0);  /* export succeeds */
    mock_rdma_set_reg_fail(true);           /* but RDMA reg fails */

    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    EXPECT_THROW(
        client.register_region(
            reinterpret_cast<uintptr_t>(buf), 4096, true /* is_gpu */),
        std::runtime_error);

    free(buf);
}

/* ── Multiple GPU pages via dma-buf ── */

TEST_F(S3RdmaDmabufTest, RegisterPages_GPU_DmabufSuccess) {
    mock_rdma_set_dmabuf_export_result(0);

    S3RdmaClient client(make_dc_config());
    const size_t page_size = 1024;
    const size_t total = 4 * page_size;
    auto* buf = alloc_buffer(total);

    size_t n = client.register_pages(
        reinterpret_cast<uintptr_t>(buf), total, page_size, true /* is_gpu */);

    EXPECT_EQ(n, 4u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 4u);

    free(buf);
}

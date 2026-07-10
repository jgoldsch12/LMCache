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
 * @file s3rdma_client_test.cpp
 * @brief Google Test suite for S3RdmaClient.
 *
 * Tests run without RDMA hardware — the vendored ibv_* and rdma_* functions
 * are replaced by mock_rdma.cpp which returns fake structs with controllable
 * behavior and call counters.
 *
 * Test categories:
 *   ConfigValidation   — S3RdmaConfig IP validation
 *   Construction       — RC/DC mode device opening, listener lifecycle
 *   MemoryRegistration — register_region, register_pages, deregister
 *   DescriptorGen      — prepare_put/get descriptor format
 *   ErrorHandling      — device failures, registration failures, bad addresses
 *   Lifecycle          — close idempotency, is_connected state transitions
 */

#include <gtest/gtest.h>

#include "s3rdma_client.h"
#include "mock_rdma.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using lmcache::s3rdma::S3RdmaClient;
using lmcache::s3rdma::S3RdmaConfig;
using lmcache::s3rdma::S3RDMA_SUCCESS;
using lmcache::s3rdma::S3RDMA_FAIL;

/* ── Test Fixture ── */

class S3RdmaClientTest : public ::testing::Test {
protected:
    void SetUp() override { mock_rdma_reset(); }
    void TearDown() override {}

    /** Helper: create a default RC config with a valid IP. */
    static S3RdmaConfig make_rc_config(const std::string& ip = "10.10.0.1",
                                       int port = 7471) {
        return S3RdmaConfig{ip, port, "RC"};
    }

    /** Helper: create a default DC config with a valid IP. */
    static S3RdmaConfig make_dc_config(const std::string& ip = "10.10.0.1") {
        return S3RdmaConfig{ip, 7471, "DC"};
    }

    /**
     * Helper: allocate a fake CPU buffer for registration tests.
     * Returns a heap-allocated buffer; caller must free().
     */
    static uint8_t* alloc_buffer(size_t size) {
        auto* buf = static_cast<uint8_t*>(malloc(size));
        EXPECT_NE(buf, nullptr);
        memset(buf, 0xAA, size);
        return buf;
    }
};

/* ════════════════════════════════════════════════════════════════════════
 * ConfigValidation — S3RdmaConfig IP address validation
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, ConfigValidIP_RC) {
    EXPECT_NO_THROW(S3RdmaClient client(make_rc_config("192.168.1.1")));
}

TEST_F(S3RdmaClientTest, ConfigValidIP_DC) {
    EXPECT_NO_THROW(S3RdmaClient client(make_dc_config("192.168.1.1")));
}

TEST_F(S3RdmaClientTest, ConfigInvalidIP_Throws) {
    S3RdmaConfig config{"not-an-ip", 7471, "RC"};
    EXPECT_THROW(S3RdmaClient client(config), std::runtime_error);
}

TEST_F(S3RdmaClientTest, ConfigEmptyIP_Throws) {
    S3RdmaConfig config{"", 7471, "RC"};
    EXPECT_THROW(S3RdmaClient client(config), std::runtime_error);
}

TEST_F(S3RdmaClientTest, ConfigInvalidIP_NoDeviceOpened) {
    S3RdmaConfig config{"bad_ip", 7471, "DC"};
    EXPECT_THROW(S3RdmaClient client(config), std::runtime_error);
    EXPECT_EQ(mock_rdma_get_reg_count(), 0u);
}

TEST_F(S3RdmaClientTest, ConfigZeroIP_Succeeds) {
    /* 0.0.0.0 is a valid inet_addr — it means "any" */
    EXPECT_NO_THROW(S3RdmaClient client(make_rc_config("0.0.0.0")));
}

TEST_F(S3RdmaClientTest, ConfigBroadcastIP_Fails) {
    /* 255.255.255.255 returns INADDR_NONE from inet_addr */
    S3RdmaConfig config{"255.255.255.255", 7471, "RC"};
    EXPECT_THROW(S3RdmaClient client(config), std::runtime_error);
}

/* ════════════════════════════════════════════════════════════════════════
 * Construction — Device opening and mode selection
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, ConstructRC_StartsListener) {
    S3RdmaClient client(make_rc_config());
    EXPECT_TRUE(client.is_connected());
}

TEST_F(S3RdmaClientTest, ConstructDC_OpensDevice) {
    S3RdmaClient client(make_dc_config());
    EXPECT_TRUE(client.is_connected());
}

TEST_F(S3RdmaClientTest, ConstructRC_DeviceOpenFail_Throws) {
    mock_rdma_set_open_device_fail(true);
    EXPECT_THROW(S3RdmaClient client(make_rc_config()), std::runtime_error);
}

TEST_F(S3RdmaClientTest, ConstructDC_DeviceOpenFail_Throws) {
    mock_rdma_set_open_device_fail(true);
    EXPECT_THROW(S3RdmaClient client(make_dc_config()), std::runtime_error);
}

TEST_F(S3RdmaClientTest, ConstructRC_ListenerFail_Throws) {
    mock_rdma_set_listener_fail(true);
    EXPECT_THROW(S3RdmaClient client(make_rc_config()), std::runtime_error);
}

TEST_F(S3RdmaClientTest, ConstructRC_CustomPort) {
    auto config = make_rc_config("10.10.0.1", 9999);
    EXPECT_NO_THROW(S3RdmaClient client(config));
}

/* ════════════════════════════════════════════════════════════════════════
 * MemoryRegistration — register_region, register_pages, deregister
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, RegisterRegion_CPU) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    auto [ptr, size] = client.register_region(
        reinterpret_cast<uintptr_t>(buf), 4096, false);

    EXPECT_EQ(ptr, reinterpret_cast<uintptr_t>(buf));
    EXPECT_EQ(size, 4096u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 1u);
    EXPECT_EQ(mock_rdma_last_reg_addr(), buf);
    EXPECT_EQ(mock_rdma_last_reg_size(), 4096u);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterRegion_RegFail_Throws) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    mock_rdma_set_reg_fail(true);

    EXPECT_THROW(
        client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false),
        std::runtime_error);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterPages_FourPages) {
    S3RdmaClient client(make_dc_config());
    const size_t page_size = 1024;
    const size_t total = 4 * page_size;
    auto* buf = alloc_buffer(total);

    size_t n = client.register_pages(
        reinterpret_cast<uintptr_t>(buf), total, page_size, false);

    EXPECT_EQ(n, 4u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 4u);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterPages_ZeroPageSize_Throws) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    EXPECT_THROW(
        client.register_pages(reinterpret_cast<uintptr_t>(buf), 4096, 0, false),
        std::runtime_error);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterPages_RemainingBytesIgnored) {
    /* total_size=5000, page_size=1024 → 4 pages (5000/1024=4, remainder ignored) */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(5000);

    size_t n = client.register_pages(
        reinterpret_cast<uintptr_t>(buf), 5000, 1024, false);

    EXPECT_EQ(n, 4u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 4u);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterPages_SinglePage) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);

    size_t n = client.register_pages(
        reinterpret_cast<uintptr_t>(buf), 4096, 4096, false);

    EXPECT_EQ(n, 1u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 1u);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterPages_SizeSmallerThanPage) {
    /* total_size < page_size → 0 pages registered */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(512);

    size_t n = client.register_pages(
        reinterpret_cast<uintptr_t>(buf), 512, 1024, false);

    EXPECT_EQ(n, 0u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 0u);

    free(buf);
}

TEST_F(S3RdmaClientTest, DeregisterRegion_Success) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    int rc = client.deregister_region(reinterpret_cast<uintptr_t>(buf));
    EXPECT_EQ(rc, S3RDMA_SUCCESS);
    EXPECT_EQ(mock_rdma_get_dereg_count(), 1u);

    free(buf);
}

TEST_F(S3RdmaClientTest, DeregisterRegion_NotFound) {
    S3RdmaClient client(make_dc_config());

    int rc = client.deregister_region(0xDEADBEEF);
    EXPECT_EQ(rc, S3RDMA_FAIL);
    EXPECT_EQ(mock_rdma_get_dereg_count(), 0u);
}

TEST_F(S3RdmaClientTest, DeregisterRegion_DoubleDeregister) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    EXPECT_EQ(client.deregister_region(reinterpret_cast<uintptr_t>(buf)),
              S3RDMA_SUCCESS);
    EXPECT_EQ(client.deregister_region(reinterpret_cast<uintptr_t>(buf)),
              S3RDMA_FAIL);

    free(buf);
}

TEST_F(S3RdmaClientTest, DeregisterAll) {
    S3RdmaClient client(make_dc_config());
    const size_t page_size = 1024;
    auto* buf = alloc_buffer(4 * page_size);

    client.register_pages(
        reinterpret_cast<uintptr_t>(buf), 4 * page_size, page_size, false);
    EXPECT_EQ(mock_rdma_get_reg_count(), 4u);

    client.deregister_all();
    EXPECT_EQ(mock_rdma_get_dereg_count(), 4u);

    /* After deregister_all, any deregister should fail */
    EXPECT_EQ(client.deregister_region(reinterpret_cast<uintptr_t>(buf)),
              S3RDMA_FAIL);

    free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * DescriptorGen — prepare_put / prepare_get descriptor format validation
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, PreparePut_ReturnsDescriptor) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);

    EXPECT_FALSE(desc.empty());
    EXPECT_EQ(mock_rdma_get_desc_count(), 1u);

    free(buf);
}

TEST_F(S3RdmaClientTest, PrepareGet_ReturnsDescriptor) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_get(
        reinterpret_cast<uintptr_t>(buf), 4096);

    EXPECT_FALSE(desc.empty());
    EXPECT_EQ(mock_rdma_get_desc_count(), 1u);

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorFormat_SevenColonSeparatedFields) {
    /*
     * Expected format: addr:size:rkey:lid:dctn:gid_flag:gid
     * 7 fields separated by 6 colons (but gid_flag has a trailing colon
     * before the 32-char GID, so we see 7 colons total).
     * Regex: 16hex : 8hex : 8hex : 4hex : 6hex : 1digit : 32hex
     */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);

    std::regex pattern(
        "[0-9a-f]{16}:"   /* addr  (16 hex) */
        "[0-9a-f]{8}:"    /* size  (8 hex)  */
        "[0-9a-f]{8}:"    /* rkey  (8 hex)  */
        "[0-9a-f]{4}:"    /* lid   (4 hex)  */
        "[0-9a-f]{6}:"    /* dctn  (6 hex)  */
        "[01]:"            /* gid_flag       */
        "[0-9a-f]{32}"    /* gid   (32 hex) */
    );
    EXPECT_TRUE(std::regex_match(desc, pattern))
        << "Descriptor does not match expected format: " << desc;

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorContainsBufferAddress) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);
    client.register_region(addr, 4096, false);

    std::string desc = client.prepare_put(addr, 4096);

    /* First 16 chars should be the hex address */
    char expected_addr[17];
    snprintf(expected_addr, sizeof(expected_addr), "%016llx",
             (unsigned long long)addr);
    EXPECT_EQ(desc.substr(0, 16), std::string(expected_addr));

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorContainsBufferSize) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(0x1000);  /* 4096 bytes */
    client.register_region(reinterpret_cast<uintptr_t>(buf), 0x1000, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 0x1000);

    /* Size field is at offset 17 (after addr + colon), 8 hex chars */
    std::string size_field = desc.substr(17, 8);
    EXPECT_EQ(size_field, "00001000");

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorContainsLid) {
    /* Mock device has lid=0x0001 */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);

    /* LID field is at offset 35 (addr:size:rkey:), 4 hex chars */
    std::string lid_field = desc.substr(35, 4);
    EXPECT_EQ(lid_field, "0001");

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorDctnZeroForRC) {
    /* RC mode: no DC target QP, so dctn should be 000000 */
    S3RdmaClient client(make_rc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);

    /* DCTN field is at offset 40 (addr:size:rkey:lid:), 6 hex chars */
    std::string dctn_field = desc.substr(40, 6);
    EXPECT_EQ(dctn_field, "000000");

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorGidFlagGlobal) {
    /* Mock device has is_global=1 */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);

    /* GID flag is at offset 47, single char */
    EXPECT_EQ(desc[47], '1');

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorContainsGid) {
    /* Mock device GID: ::ffff:10.246.191.101
     * In wire format: 00000000000000000000ffff0af6bf65 */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);

    /* GID starts at offset 49 (after "...gid_flag:"), 32 hex chars */
    std::string gid_field = desc.substr(49, 32);
    EXPECT_EQ(gid_field, "00000000000000000000ffff0af6bf65");

    free(buf);
}

TEST_F(S3RdmaClientTest, PreparePutAndGetSameDescriptor) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string put_desc = client.prepare_put(
        reinterpret_cast<uintptr_t>(buf), 4096);
    std::string get_desc = client.prepare_get(
        reinterpret_cast<uintptr_t>(buf), 4096);

    EXPECT_EQ(put_desc, get_desc);

    free(buf);
}

TEST_F(S3RdmaClientTest, PerPageDescriptors_DifferentAddresses) {
    S3RdmaClient client(make_dc_config());
    const size_t page_size = 1024;
    auto* buf = alloc_buffer(4 * page_size);
    uintptr_t base = reinterpret_cast<uintptr_t>(buf);

    client.register_pages(base, 4 * page_size, page_size, false);

    /* Each page should have a different descriptor (different addr) */
    std::string desc0 = client.prepare_put(base, page_size);
    std::string desc1 = client.prepare_put(base + page_size, page_size);
    std::string desc2 = client.prepare_put(base + 2 * page_size, page_size);
    std::string desc3 = client.prepare_put(base + 3 * page_size, page_size);

    EXPECT_NE(desc0, desc1);
    EXPECT_NE(desc1, desc2);
    EXPECT_NE(desc2, desc3);

    /* All should have the same size field */
    EXPECT_EQ(desc0.substr(17, 8), desc1.substr(17, 8));

    free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * DescriptorPortBehavior — current ObjectScale design: the RDMA descriptor
 * carries NO listener port.  The server connects back to the client's
 * advertised GID at a fixed port negotiated out-of-band, so prepare_put/get
 * must NOT append ":PORT" to the descriptor in either RC or DC mode, and the
 * descriptor must be independent of the configured rdma_port.  These tests
 * guard against regressing to an earlier per-worker "port-in-descriptor"
 * scheme, which breaks ObjectScale connect-back.
 * ════════════════════════════════════════════════════════════════════════ */

/** Canonical bare descriptor: addr:size:rkey:lid:dctn:gid_flag:gid (no port). */
static const char* kNoPortDescriptorPattern =
    "[0-9a-f]{16}:"   /* addr     (16 hex) */
    "[0-9a-f]{8}:"    /* size     (8 hex)  */
    "[0-9a-f]{8}:"    /* rkey     (8 hex)  */
    "[0-9a-f]{4}:"    /* lid      (4 hex)  */
    "[0-9a-f]{6}:"    /* dctn     (6 hex)  */
    "[01]:"            /* gid_flag (1 digit)*/
    "[0-9a-f]{32}";   /* gid      (32 hex) */

TEST_F(S3RdmaClientTest, DescriptorRC_NoPortSuffix) {
    /* RC mode with a non-default listener port: the descriptor must still be
     * the bare 7-field form (exactly 6 colons, no trailing ":PORT"). */
    S3RdmaClient client(make_rc_config("10.10.0.1", 7474));
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    std::string desc = client.prepare_get(
        reinterpret_cast<uintptr_t>(buf), 4096);

    /* Exactly 6 colons => 7 fields, i.e. no 8th (port) field. */
    EXPECT_EQ(std::count(desc.begin(), desc.end(), ':'), 6)
        << "RC descriptor unexpectedly has an extra (port) field: " << desc;

    /* regex_match anchors the whole string, so a trailing ":7474" would fail. */
    EXPECT_TRUE(std::regex_match(desc, std::regex(kNoPortDescriptorPattern)))
        << "RC descriptor must not append a port: " << desc;

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorRC_IndependentOfListenerPort) {
    /* Two RC clients on the same NIC but different listener ports must produce
     * byte-identical descriptors for the same buffer — the port is not part of
     * the descriptor.  (This is the inverse of the old per-worker port scheme.)
     * The mock assigns rkey deterministically from a per-test registration
     * counter, so mock_rdma_reset() makes both clients' first registration
     * identical. */
    auto* buf = alloc_buffer(4096);
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);

    std::string desc_a;
    {
        S3RdmaClient client_a(make_rc_config("10.10.0.1", 7471));
        client_a.register_region(addr, 4096, false);
        desc_a = client_a.prepare_get(addr, 4096);
    }

    mock_rdma_reset();

    std::string desc_b;
    {
        S3RdmaClient client_b(make_rc_config("10.10.0.1", 7480));
        client_b.register_region(addr, 4096, false);
        desc_b = client_b.prepare_get(addr, 4096);
    }

    EXPECT_EQ(desc_a, desc_b)
        << "Descriptor changed with listener port (port 7471 -> '" << desc_a
        << "', port 7480 -> '" << desc_b << "')";

    free(buf);
}

TEST_F(S3RdmaClientTest, DescriptorRCAndDC_BothBareNoPortFormat) {
    /* RC and DC descriptors share the same bare 7-field format (no port).
     * The dctn field may differ (DC has a target QP, RC does not), so we
     * assert the field count/format rather than equality. */
    auto* buf = alloc_buffer(4096);
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);

    std::string rc_desc;
    {
        S3RdmaClient rc(make_rc_config("10.10.0.1", 7471));
        rc.register_region(addr, 4096, false);
        rc_desc = rc.prepare_get(addr, 4096);
    }

    mock_rdma_reset();

    std::string dc_desc;
    {
        S3RdmaClient dc(make_dc_config("10.10.0.1"));
        dc.register_region(addr, 4096, false);
        dc_desc = dc.prepare_get(addr, 4096);
    }

    EXPECT_EQ(std::count(rc_desc.begin(), rc_desc.end(), ':'), 6) << rc_desc;
    EXPECT_EQ(std::count(dc_desc.begin(), dc_desc.end(), ':'), 6) << dc_desc;
    EXPECT_TRUE(std::regex_match(rc_desc, std::regex(kNoPortDescriptorPattern)))
        << "RC: " << rc_desc;
    EXPECT_TRUE(std::regex_match(dc_desc, std::regex(kNoPortDescriptorPattern)))
        << "DC: " << dc_desc;

    free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * ErrorHandling — failure injection and edge cases
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, PreparePut_UnregisteredAddress_Throws) {
    S3RdmaClient client(make_dc_config());

    EXPECT_THROW(client.prepare_put(0xDEADBEEF, 4096), std::runtime_error);
}

TEST_F(S3RdmaClientTest, PrepareGet_UnregisteredAddress_Throws) {
    S3RdmaClient client(make_dc_config());

    EXPECT_THROW(client.prepare_get(0xDEADBEEF, 4096), std::runtime_error);
}

TEST_F(S3RdmaClientTest, RegisterRegion_AfterClose_Throws) {
    S3RdmaClient client(make_dc_config());
    client.close();

    auto* buf = alloc_buffer(4096);
    EXPECT_THROW(
        client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false),
        std::runtime_error);

    free(buf);
}

TEST_F(S3RdmaClientTest, RegisterPages_PartialFailure) {
    /* Register first page successfully, then fail on second */
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(2048);

    /* Register first page manually */
    client.register_region(reinterpret_cast<uintptr_t>(buf), 1024, false);
    EXPECT_EQ(mock_rdma_get_reg_count(), 1u);

    /* Now make registration fail */
    mock_rdma_set_reg_fail(true);
    EXPECT_THROW(
        client.register_region(reinterpret_cast<uintptr_t>(buf) + 1024, 1024, false),
        std::runtime_error);

    /* First page should still be valid */
    EXPECT_NO_THROW(
        client.prepare_put(reinterpret_cast<uintptr_t>(buf), 1024));

    free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * Lifecycle — close, is_connected, destructor
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, IsConnected_TrueAfterConstruct) {
    S3RdmaClient client(make_dc_config());
    EXPECT_TRUE(client.is_connected());
}

TEST_F(S3RdmaClientTest, IsConnected_FalseAfterClose) {
    S3RdmaClient client(make_dc_config());
    client.close();
    EXPECT_FALSE(client.is_connected());
}

TEST_F(S3RdmaClientTest, Close_Idempotent) {
    S3RdmaClient client(make_dc_config());
    EXPECT_EQ(client.close(), S3RDMA_SUCCESS);
    EXPECT_EQ(client.close(), S3RDMA_SUCCESS);
    EXPECT_FALSE(client.is_connected());
}

TEST_F(S3RdmaClientTest, Close_DeregistersAllMRs) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_pages(
        reinterpret_cast<uintptr_t>(buf), 4096, 1024, false);

    EXPECT_EQ(mock_rdma_get_reg_count(), 4u);

    client.close();
    EXPECT_EQ(mock_rdma_get_dereg_count(), 4u);

    free(buf);
}

TEST_F(S3RdmaClientTest, Close_DC_CallsCloseDevice) {
    S3RdmaClient client(make_dc_config());
    client.close();
    EXPECT_EQ(mock_rdma_get_close_device_count(), 1u);
}

TEST_F(S3RdmaClientTest, Close_RC_FreesDeviceWithoutCloseDevice) {
    /* RC mode: device was created with rdma_get_device (borrowed PD),
     * so close() should free() the struct, NOT call rdma_close_device() */
    S3RdmaClient client(make_rc_config());
    client.close();
    EXPECT_EQ(mock_rdma_get_close_device_count(), 0u);
}

TEST_F(S3RdmaClientTest, DestructorCallsClose) {
    auto* buf = alloc_buffer(4096);
    {
        S3RdmaClient client(make_dc_config());
        client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);
        /* Destructor should call close(), which deregisters all MRs */
    }
    EXPECT_EQ(mock_rdma_get_dereg_count(), 1u);
    free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * Constants — verify exported constants
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, Constants) {
    EXPECT_EQ(lmcache::s3rdma::S3RDMA_SUCCESS, 0);
    EXPECT_EQ(lmcache::s3rdma::S3RDMA_FAIL, 1);
    EXPECT_EQ(lmcache::s3rdma::S3RDMA_MEMORY_SYSTEM, 0);
    EXPECT_EQ(lmcache::s3rdma::S3RDMA_MEMORY_GPU, 2);
    EXPECT_EQ(lmcache::s3rdma::S3RDMA_MAX_MEMORY_REG_SIZE,
              4ULL * 1024 * 1024 * 1024);
}

/* ════════════════════════════════════════════════════════════════════════
 * RC Listener Port Allocation
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, RC_ListenerPortDefault) {
    /* Default port is 7471 */
    auto config = make_rc_config("10.10.0.1", 7471);
    EXPECT_NO_THROW(S3RdmaClient client(config));
}

TEST_F(S3RdmaClientTest, RC_ListenerPortCustom) {
    /* base_port + worker_id pattern: e.g. 7471 + 3 = 7474 */
    auto config = make_rc_config("10.10.0.1", 7474);
    EXPECT_NO_THROW(S3RdmaClient client(config));
}

TEST_F(S3RdmaClientTest, RC_MultiplePortAllocations) {
    /* Simulate TP workers each getting their own port */
    for (int tp_rank = 0; tp_rank < 4; ++tp_rank) {
        mock_rdma_reset();
        int port = 7471 + tp_rank;
        auto config = make_rc_config("10.10.0.1", port);
        EXPECT_NO_THROW(S3RdmaClient client(config));
    }
}

/* ════════════════════════════════════════════════════════════════════════
 * Concurrency — thread-safety of register, prepare, and deregister
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, ConcurrentRegisterAndPrepare) {
    S3RdmaClient client(make_dc_config());
    const int num_threads = 8;
    const size_t buf_size = 4096;

    std::vector<uint8_t*> buffers(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        buffers[i] = alloc_buffer(buf_size);
    }

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            uintptr_t addr = reinterpret_cast<uintptr_t>(buffers[i]);
            auto [ptr, sz] = client.register_region(addr, buf_size, false);
            EXPECT_EQ(ptr, addr);

            std::string desc = client.prepare_put(addr, buf_size);
            EXPECT_FALSE(desc.empty());

            success_count++;
        });
    }

    for (auto& t : threads) t.join();

    EXPECT_EQ(success_count.load(), num_threads);
    EXPECT_EQ(mock_rdma_get_reg_count(), static_cast<size_t>(num_threads));

    for (auto* buf : buffers) free(buf);
}

TEST_F(S3RdmaClientTest, ConcurrentDeregister) {
    S3RdmaClient client(make_dc_config());
    const int num_threads = 8;
    const size_t buf_size = 4096;

    /* Register all buffers sequentially */
    std::vector<uint8_t*> buffers(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        buffers[i] = alloc_buffer(buf_size);
        client.register_region(
            reinterpret_cast<uintptr_t>(buffers[i]), buf_size, false);
    }

    /* Concurrently deregister all */
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            int rc = client.deregister_region(
                reinterpret_cast<uintptr_t>(buffers[i]));
            if (rc == S3RDMA_SUCCESS) success_count++;
        });
    }

    for (auto& t : threads) t.join();

    EXPECT_EQ(success_count.load(), num_threads);
    EXPECT_EQ(mock_rdma_get_dereg_count(),
              static_cast<size_t>(num_threads));

    for (auto* buf : buffers) free(buf);
}

TEST_F(S3RdmaClientTest, ConcurrentRegisterDeregisterInterleaved) {
    /* Some threads register while others deregister different buffers */
    S3RdmaClient client(make_dc_config());
    const size_t buf_size = 4096;

    /* Pre-register 4 buffers for the deregister threads */
    std::vector<uint8_t*> pre_bufs(4);
    for (int i = 0; i < 4; ++i) {
        pre_bufs[i] = alloc_buffer(buf_size);
        client.register_region(
            reinterpret_cast<uintptr_t>(pre_bufs[i]), buf_size, false);
    }

    /* 4 new buffers for the register threads */
    std::vector<uint8_t*> new_bufs(4);
    for (int i = 0; i < 4; ++i) {
        new_bufs[i] = alloc_buffer(buf_size);
    }

    std::vector<std::thread> threads;
    std::atomic<int> reg_ok{0}, dereg_ok{0};

    /* Launch register and deregister threads interleaved */
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i]() {
            client.register_region(
                reinterpret_cast<uintptr_t>(new_bufs[i]), buf_size, false);
            reg_ok++;
        });
        threads.emplace_back([&, i]() {
            if (client.deregister_region(
                    reinterpret_cast<uintptr_t>(pre_bufs[i])) == S3RDMA_SUCCESS)
                dereg_ok++;
        });
    }

    for (auto& t : threads) t.join();

    EXPECT_EQ(reg_ok.load(), 4);
    EXPECT_EQ(dereg_ok.load(), 4);

    for (auto* buf : pre_bufs) free(buf);
    for (auto* buf : new_bufs) free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * DescriptorGenError — rdma_buffer_get_desc_str returns 0
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, PreparePut_DescriptorGenFail_Throws) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    mock_rdma_set_desc_fail(true);
    EXPECT_THROW(
        client.prepare_put(reinterpret_cast<uintptr_t>(buf), 4096),
        std::runtime_error);

    free(buf);
}

TEST_F(S3RdmaClientTest, PrepareGet_DescriptorGenFail_Throws) {
    S3RdmaClient client(make_dc_config());
    auto* buf = alloc_buffer(4096);
    client.register_region(reinterpret_cast<uintptr_t>(buf), 4096, false);

    mock_rdma_set_desc_fail(true);
    EXPECT_THROW(
        client.prepare_get(reinterpret_cast<uintptr_t>(buf), 4096),
        std::runtime_error);

    free(buf);
}

/* ════════════════════════════════════════════════════════════════════════
 * LargeRegion — boundary sizes near S3RDMA_MAX_MEMORY_REG_SIZE
 * ════════════════════════════════════════════════════════════════════════ */

TEST_F(S3RdmaClientTest, RegisterRegion_LargeSize) {
    /* Mock doesn't dereference the address, so a fake pointer is fine */
    S3RdmaClient client(make_dc_config());
    uintptr_t fake_addr = 0x7F0000000000ULL;
    size_t large_size = lmcache::s3rdma::S3RDMA_MAX_MEMORY_REG_SIZE;

    auto [ptr, sz] = client.register_region(fake_addr, large_size, false);
    EXPECT_EQ(ptr, fake_addr);
    EXPECT_EQ(sz, large_size);

    std::string desc = client.prepare_put(fake_addr, large_size);
    EXPECT_FALSE(desc.empty());
}

TEST_F(S3RdmaClientTest, RegisterPages_LargeRegion_ManyPages) {
    S3RdmaClient client(make_dc_config());
    uintptr_t fake_addr = 0x7F0000000000ULL;
    size_t page_size = 64 * 1024 * 1024;  /* 64 MB pages */
    size_t total = page_size * 16;         /* 1 GB total, 16 pages */

    size_t n = client.register_pages(fake_addr, total, page_size, false);
    EXPECT_EQ(n, 16u);
    EXPECT_EQ(mock_rdma_get_reg_count(), 16u);
}

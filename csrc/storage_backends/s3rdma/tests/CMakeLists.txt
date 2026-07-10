cmake_minimum_required(VERSION 3.14)
project(s3rdma_client_tests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ── Google Test ──
find_package(GTest REQUIRED)

# ── Source paths ──
set(S3RDMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(S3RDMA_INCLUDE "${S3RDMA_DIR}/include")
# Stubs directory provides gpu_compat.h without CUDA/HIP runtime dependency.
# It must come BEFORE S3RDMA_INCLUDE so the stub overrides the real header.
set(STUBS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/stubs")

# ── Mock RDMA library ──
# Replaces gpu_direct_rdma_access.cpp and rdma_listener.cpp with mocks
# so tests run without RDMA hardware.
add_library(mock_rdma STATIC
    mock_rdma.cpp
)
target_include_directories(mock_rdma PUBLIC
    ${S3RDMA_INCLUDE}
)

# ── S3RdmaClient library (with mock RDMA, no dmabuf) ──
add_library(s3rdma_client_testable STATIC
    "${S3RDMA_DIR}/s3rdma_client.cpp"
    "${S3RDMA_DIR}/log.cpp"
)
target_include_directories(s3rdma_client_testable PUBLIC
    ${STUBS_INCLUDE}
    ${S3RDMA_INCLUDE}
)
target_link_libraries(s3rdma_client_testable PUBLIC mock_rdma)
# Suppress warnings from vendored header patterns
target_compile_options(s3rdma_client_testable PRIVATE -Wno-missing-field-initializers)

# ── S3RdmaClient library (with mock RDMA + GPU_HAS_DMABUF_EXPORT) ──
add_library(s3rdma_client_testable_dmabuf STATIC
    "${S3RDMA_DIR}/s3rdma_client.cpp"
    "${S3RDMA_DIR}/log.cpp"
)
target_include_directories(s3rdma_client_testable_dmabuf PUBLIC
    ${STUBS_INCLUDE}
    ${S3RDMA_INCLUDE}
)
target_compile_definitions(s3rdma_client_testable_dmabuf PRIVATE GPU_HAS_DMABUF_EXPORT)
target_link_libraries(s3rdma_client_testable_dmabuf PUBLIC mock_rdma)
target_compile_options(s3rdma_client_testable_dmabuf PRIVATE -Wno-missing-field-initializers)

# ── Test executable (main tests, NVIDIA/CPU path) ──
add_executable(s3rdma_client_test
    s3rdma_client_test.cpp
)
target_include_directories(s3rdma_client_test PRIVATE
    ${S3RDMA_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(s3rdma_client_test
    s3rdma_client_testable
    GTest::gtest_main
    pthread
)

# ── Test executable (dmabuf / AMD HIP path) ──
add_executable(s3rdma_client_dmabuf_test
    s3rdma_client_dmabuf_test.cpp
)
target_include_directories(s3rdma_client_dmabuf_test PRIVATE
    ${S3RDMA_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(s3rdma_client_dmabuf_test
    s3rdma_client_testable_dmabuf
    GTest::gtest_main
    pthread
)

# ── Register with CTest ──
include(GoogleTest)
gtest_discover_tests(s3rdma_client_test)
gtest_discover_tests(s3rdma_client_dmabuf_test)

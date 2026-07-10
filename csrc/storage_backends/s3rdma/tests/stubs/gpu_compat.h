/*
 * Stub gpu_compat.h for unit testing without CUDA or HIP runtime.
 *
 * S3RdmaClient only uses gpu_compat.h for the GPU_HAS_DMABUF_EXPORT path
 * (AMD dma-buf export).  The real implementation is provided by
 * mock_rdma.cpp with controllable success/failure via
 * mock_rdma_set_dmabuf_export_result().
 */

#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H

#include <cstddef>
#include <cstdint>

/* Provided by mock_rdma.cpp — controllable via mock_rdma_set_dmabuf_export_result() */
int gpuExportDmaBuf(const void *ptr, size_t size,
                    int *dmabuf_fd, uint64_t *offset);

#endif /* GPU_COMPAT_H */

/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Dell Confidential.
 * This software contains proprietary and confidential information of Dell Inc.
 * and is subject to applicable non-disclosure agreements.
 * Unauthorized copying, modification, distribution, or use is strictly prohibited.
 */
/*
 * gpu_compat.h — Thin GPU runtime abstraction layer.
 *
 * Provides a common set of gpu* symbols that map to either CUDA or HIP
 * depending on the USE_HIP compile-time flag.  This allows agent.cpp to
 * remain a single source file that builds against either runtime.
 *
 * Usage:
 *   #include "gpu_compat.h"
 *   gpuError_t err = gpuMalloc(&ptr, size);
 *
 * Build selection (CMake):
 *   cmake .. -DUSE_HIP=ON   # ROCm / HIP
 *   cmake ..                 # NVIDIA CUDA (default)
 */

#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H

#ifdef __cplusplus
#include <cstdlib>
#include <iostream>
#endif

#ifdef USE_HIP

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

typedef hipError_t gpuError_t;

#define gpuSuccess              hipSuccess
#define gpuMalloc               hipMalloc
#define gpuFree                 hipFree
#define gpuMemcpy               hipMemcpy
#define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define gpuGetDeviceCount       hipGetDeviceCount
#define gpuGetErrorName         hipGetErrorName

#define GPU_RUNTIME_NAME "HIP"
#define GPU_HAS_DMABUF_EXPORT 1

#elif defined(S3RDMA_NO_GPU_RUNTIME)

typedef int gpuError_t;

#define gpuSuccess 0
#define gpuGetErrorName(err) "no GPU runtime"

#define GPU_RUNTIME_NAME "NO_GPU_RUNTIME"

#else /* CUDA (default) */

#include <cuda_runtime.h>
#include <cuda.h>

typedef cudaError_t gpuError_t;

#define gpuSuccess              cudaSuccess
#define gpuMalloc               cudaMalloc
#define gpuFree                 cudaFree
#define gpuMemcpy               cudaMemcpy
#define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define gpuGetDeviceCount       cudaGetDeviceCount
#define gpuGetErrorName         cudaGetErrorName

#define GPU_RUNTIME_NAME "CUDA"

#endif /* USE_HIP */

/*
 * gpuExportDmaBuf — Export GPU memory as a dma-buf file descriptor.
 *
 * On HIP/ROCm this uses hsa_amd_portable_export_dmabuf() which is the
 * vendor-neutral path for registering GPU memory with any RDMA NIC
 * (including non-mlx5 NICs like Broadcom bnxt_re that don't support
 * the legacy ib_peer_memory kernel interface).
 *
 * On CUDA this is not implemented — NVIDIA GPUs use nvidia-peermem
 * which makes plain ibv_reg_mr() work transparently.
 *
 * Returns 0 on success, -1 if dmabuf export is not available.
 */
#ifdef GPU_HAS_DMABUF_EXPORT
static inline int gpuExportDmaBuf(const void *ptr, size_t size,
                                  int *dmabuf_fd, uint64_t *offset)
{
    hsa_status_t rc = hsa_amd_portable_export_dmabuf(ptr, size, dmabuf_fd, offset);
    return (rc == HSA_STATUS_SUCCESS) ? 0 : -1;
}
#else
static inline int gpuExportDmaBuf(const void *ptr, size_t size,
                                  int *dmabuf_fd, uint64_t *offset)
{
    (void)ptr; (void)size; (void)dmabuf_fd; (void)offset;
    return -1; /* Not available — use ibv_reg_mr via nvidia-peermem instead */
}
#endif

/* Common error-checking macro */
#define check_gpu_call(fn)                                                  \
    do {                                                                    \
        gpuError_t res = fn;                                                \
        if (res != gpuSuccess) {                                            \
            const char *str = gpuGetErrorName(res);                         \
            std::cout << GPU_RUNTIME_NAME " runtime api call failed "       \
                      << #fn << " " << __LINE__ << ": " << str << std::endl;\
            std::cout << "EXITING program!!!" << std::endl;                 \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#endif /* GPU_COMPAT_H */

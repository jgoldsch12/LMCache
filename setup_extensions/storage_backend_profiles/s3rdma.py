# SPDX-License-Identifier: Apache-2.0
"""S3 RDMA L2 storage backend profile.

Builds the ``lmcache.lmcache_s3rdma`` extension when libibverbs headers are
available. Enabled via ``BUILD_S3RDMA=1`` or auto-detected from system headers.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile


ROOT_DIR = Path(__file__).parent.parent.parent

BUILD_S3RDMA = os.environ.get("BUILD_S3RDMA", "auto")
USE_MLX5 = os.environ.get("USE_MLX5", "auto")


def _check_header(path: str) -> bool:
    """Return True when a development header exists on the build host."""
    return os.path.exists(path)


def _check_cuda_available() -> bool:
    """Check if CUDA runtime headers are available."""
    for root in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr",
    ):
        if root and _check_header(os.path.join(root, "include", "cuda_runtime.h")):
            return True
    return False


def _check_mlx5_available() -> bool:
    """Check if mlx5 direct-verbs headers are available."""
    if USE_MLX5 == "0":
        return False
    if USE_MLX5 == "1":
        return True
    return _check_header("/usr/include/infiniband/mlx5dv.h")


class S3RdmaStorageBackend(StorageBackendProfile):
    """S3 RDMA storage backend."""

    name = "s3rdma"
    env_var = "BUILD_S3RDMA"

    def detect(self) -> bool:
        """Detect libibverbs development headers."""
        if BUILD_S3RDMA == "0":
            return False
        if BUILD_S3RDMA == "1":
            return True
        return _check_header("/usr/include/infiniband/verbs.h")

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the S3 RDMA CppExtension."""
        # Third Party
        from torch.utils import cpp_extension

        s3rdma_dir = ROOT_DIR / "csrc" / "storage_backends" / "s3rdma"
        s3rdma_sources = [
            str(s3rdma_dir / "s3rdma_client.cpp"),
            str(s3rdma_dir / "pybind.cpp"),
            str(s3rdma_dir / "gpu_direct_rdma_access.cpp"),
            str(s3rdma_dir / "rdma_listener.cpp"),
            str(s3rdma_dir / "log.cpp"),
        ]
        s3rdma_include_dirs = [
            str(s3rdma_dir / "include"),
        ]
        s3rdma_libraries = ["ibverbs", "rdmacm"]
        s3rdma_defines: list[tuple[str, str]] = []

        is_hip_build = (
            os.environ.get("BUILD_WITH_HIP", "0") == "1"
            or getattr(cpp_extension, "IS_HIP_EXTENSION", False)
        )

        if is_hip_build:
            s3rdma_defines.append(("USE_HIP", "1"))
            s3rdma_include_dirs.append(
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
            )
        elif not _check_cuda_available():
            s3rdma_defines.append(("S3RDMA_NO_GPU_RUNTIME", "1"))
            print("Building S3 RDMA extension without CUDA/HIP runtime headers")

        if _check_mlx5_available():
            s3rdma_defines.append(("USE_MLX5", "1"))
            s3rdma_libraries.append("mlx5")
            print("Building S3 RDMA extension with mlx5 DC transport support")
        else:
            print("Building S3 RDMA extension without mlx5 (RC mode only)")

        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_s3rdma",
                sources=s3rdma_sources,
                include_dirs=s3rdma_include_dirs,
                libraries=s3rdma_libraries,
                define_macros=s3rdma_defines,
                extra_compile_args={
                    "cxx": extra_cxx_flags
                    + ["-O3", "-std=c++17", "-Wno-missing-field-initializers"],
                },
            ),
        ]

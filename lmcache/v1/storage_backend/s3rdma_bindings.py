# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python wrapper for the S3 RDMA client (lmcache_s3rdma C++ extension).

Uses libibverbs (NVIDIA + AMD) for RDMA-accelerated S3 PUT/GET via
``x-rdma-info`` headers. Supports per-page memory registration (matching
the NixlStorageBackend pattern) and single-region registration.

Build-time fallback: if ``libibverbs`` headers were not available when
``setup.py`` ran, the ``lmcache_s3rdma`` module will not exist and
``S3RdmaClient`` is set to ``None``.
"""

# Standard
from dataclasses import dataclass
from typing import Optional, Tuple

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

try:
    # First Party
    from lmcache.lmcache_s3rdma import (
        S3RDMA_FAIL,
        S3RDMA_MAX_MEMORY_REG_SIZE,
        S3RDMA_MEMORY_GPU,
        S3RDMA_MEMORY_SYSTEM,
        S3RDMA_SUCCESS,
    )
    from lmcache.lmcache_s3rdma import S3RdmaClient as _NativeS3RdmaClient
    from lmcache.lmcache_s3rdma import S3RdmaConfig as _NativeS3RdmaConfig

    _S3RDMA_AVAILABLE = True
except ImportError:
    _NativeS3RdmaClient = None  # type: ignore[assignment,misc]
    _NativeS3RdmaConfig = None  # type: ignore[assignment,misc]
    S3RDMA_SUCCESS = 0
    S3RDMA_FAIL = 1
    S3RDMA_MAX_MEMORY_REG_SIZE = 4 * 1024 * 1024 * 1024
    S3RDMA_MEMORY_SYSTEM = 0
    S3RDMA_MEMORY_GPU = 2
    _S3RDMA_AVAILABLE = False


def is_s3rdma_available() -> bool:
    """Return True if the lmcache_s3rdma C++ extension was built."""
    return _S3RDMA_AVAILABLE


@dataclass
class S3RdmaConfig:
    """Configuration for the S3 RDMA client.

    Attributes:
        local_ip: RDMA NIC IP address to bind to.
        rdma_port: RC listener port (default 7471).
        connection_mode: ``"RC"`` (Reliable Connected) or ``"DC"``
            (Datagram Connected, mlx5 only).
    """

    local_ip: str
    rdma_port: int = 7471
    connection_mode: str = "RC"


class S3RdmaClientWrapper:
    """Python wrapper around the ``lmcache_s3rdma.S3RdmaClient`` C++ class.

    Provides a safe Python interface with:
    - Build-time fallback (raises ``ImportError`` if C++ ext missing)
    - Per-page and single-region RDMA memory registration
    - ``x-rdma-info`` descriptor generation for PUT and GET
    - ``x-amz-rdma-reply`` response header parsing

    Args:
        config: RDMA configuration. If ``None``, uses defaults.

    Raises:
        ImportError: If ``lmcache_s3rdma`` C++ extension is not available.
        RuntimeError: If the RDMA device cannot be opened.
    """

    def __init__(self, config: Optional[S3RdmaConfig] = None) -> None:
        if not _S3RDMA_AVAILABLE:
            raise ImportError(
                "lmcache_s3rdma C++ extension not available. "
                "Rebuild with libibverbs and RDMA development headers."
            )
        if config is None:
            config = S3RdmaConfig(local_ip="0.0.0.0")

        self._config = config
        native_config = _NativeS3RdmaConfig()
        native_config.local_ip = config.local_ip
        native_config.rdma_port = config.rdma_port
        native_config.connection_mode = config.connection_mode
        self._client = _NativeS3RdmaClient(native_config)

    def register_pages(
        self,
        base_ptr: int,
        total_size: int,
        page_size: int,
        is_gpu: bool = False,
    ) -> int:
        """Register per-page MRs covering a contiguous buffer.

        Matches NixlStorageBackend's ``init_mem_handlers()`` pattern: one
        MR per page-sized chunk.

        Args:
            base_ptr: Base address of the contiguous buffer.
            total_size: Total byte size of the buffer.
            page_size: Size of each page (one MR per page).
            is_gpu: True if the pointer is GPU memory.

        Returns:
            Number of pages registered.
        """
        return self._client.register_pages(base_ptr, total_size, page_size, is_gpu)

    def register_region(
        self, ptr: int, size: int, is_gpu: bool = False
    ) -> Tuple[int, int]:
        """Register a single contiguous memory region for RDMA.

        Args:
            ptr: Base address of the memory region.
            size: Byte size.
            is_gpu: True if the pointer is GPU memory.

        Returns:
            ``(ptr, size)`` tuple serving as the registration handle.

        Raises:
            RuntimeError: On registration failure.
        """
        return self._client.register_region(ptr, size, is_gpu)

    def deregister_all(self) -> None:
        """Deregister all previously registered memory regions."""
        self._client.deregister_all()

    def prepare_put(self, ptr: int, size: int) -> str:
        """Generate an ``x-rdma-info`` descriptor for a PUT operation.

        The descriptor encodes the buffer's RDMA address, rkey, lid, dctn,
        and GID so that the ObjectScale server can perform RDMA READ from
        the client buffer.

        Args:
            ptr: Data pointer within a registered region.
            size: Byte size of the data to upload.

        Returns:
            The ``x-rdma-info`` header value string.

        Raises:
            RuntimeError: If ``ptr`` is not within any registered region.
        """
        return self._client.prepare_put(ptr, size)

    def prepare_get(self, ptr: int, size: int) -> str:
        """Generate an ``x-rdma-info`` descriptor for a GET operation.

        The server uses this to RDMA WRITE directly into the client buffer.

        Args:
            ptr: Destination pointer within a registered region.
            size: Expected byte size.

        Returns:
            The ``x-rdma-info`` header value string.

        Raises:
            RuntimeError: If ``ptr`` is not within any registered region.
        """
        return self._client.prepare_get(ptr, size)

    def is_connected(self) -> bool:
        """Check if the RDMA device is ready.

        Returns:
            True if the device is open and functional.
        """
        return self._client.is_connected()

    @staticmethod
    def parse_rdma_reply(reply_header: Optional[str]) -> bool:
        """Parse the ``x-amz-rdma-reply`` response header.

        ObjectScale returns ``x-amz-rdma-reply: 200`` on successful RDMA
        transfers (both PUT and GET). On failure, the header is absent and
        the HTTP status is 500.

        Args:
            reply_header: The value of the ``x-amz-rdma-reply`` header,
                or ``None`` if absent.

        Returns:
            True if the RDMA transfer was successful.
        """
        if not reply_header or not reply_header.strip():
            return False
        try:
            return int(reply_header.strip()) == 200
        except ValueError:
            return False

    def close(self) -> int:
        """Release all RDMA resources.

        Deregisters all MRs, stops the RC listener (if running), and
        releases the RDMA device. Safe to call multiple times.

        Returns:
            0 on success.
        """
        if self._client is not None:
            self._client.deregister_all()
            rc = self._client.close()
            self._client = None  # type: ignore[assignment]
            return rc
        return S3RDMA_SUCCESS

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self.close()

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

"""S3 RDMA storage backend for LMCache.

Follows the NixlStorageBackend pattern:
- Owns a GPU (or CPU) bounce buffer
- Wraps it in PagedTensorMemoryAllocator
- Returns self from get_allocator_backend()
- Manages S3 RDMA transfers via S3RdmaClientWrapper

On GET: S3 -> RDMA WRITE -> GPU bounce buffer -> GPU scatter -> vLLM KV
On PUT: vLLM KV -> D2H -> CPU -> H2D -> GPU bounce -> RDMA READ -> S3

Design ref: docs/rdma-accelerated-s3-connector-design.md sections 4, 6, 7.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Set, Union
from urllib.parse import quote as url_quote
from urllib.parse import urlparse
import asyncio
import threading
import time

# Third Party
from awscrt import auth, io, s3
from awscrt.http import HttpHeaders, HttpRequest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    _allocate_cpu_memory,
    _allocate_gpu_memory,
    _free_cpu_memory,
)
from lmcache.v1.memory_allocators import PagedTensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.s3rdma_bindings import (
    S3RdmaClientWrapper,
    S3RdmaConfig,
)
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


def _select_for_worker(
    value: Union[str, Sequence[str]],
    worker_id: int,
    field_name: str,
) -> str:
    """Resolve a per-TP-worker config value from a scalar or a list.

    Enables multi-NIC / multi-endpoint deployments: ``s3rdma_local_ip`` and
    ``s3rdma_s3_endpoint`` may be either a single string (shared by all TP
    workers — the legacy behavior) or a list. When a list is given, worker
    ``worker_id`` selects element ``worker_id % len(value)`` (round-robin), so
    each TP rank binds a distinct RDMA NIC and/or talks to a distinct
    S3/ObjectScale endpoint. Order the list so that index == TP rank == the
    worker's GPU, i.e. put the topology-optimal NIC for each GPU at that GPU's
    rank index.

    Args:
        value: A single value, or a list/tuple of values indexed by TP rank.
        worker_id: The TP worker id (rank) for this backend instance.
        field_name: Config field name, used only for error messages.

    Returns:
        The resolved string value for this worker.

    Raises:
        AssertionError: If ``value`` is a list/tuple but empty.
    """
    if isinstance(value, (list, tuple)):
        assert len(value) > 0, f"{field_name} list must be non-empty"
        return str(value[worker_id % len(value)])
    return str(value)


@dataclass
class S3RdmaStorageConfig:
    """Configuration for the S3 RDMA storage backend.

    Attributes:
        buffer_size: Size of the bounce buffer in bytes.
        buffer_device: Device for the bounce buffer (``"gpu"`` or ``"cpu"``).
            ``"cuda"`` is accepted as an alias for ``"gpu"``.
        local_ip: RDMA NIC IP address.
        rdma_port: RC listener port (default 7471).  Ignored in DC mode.
        connection_mode: ``"RC"`` or ``"DC"``.
        bucket: S3 bucket name.
        region: S3 region (default ``"us-east-1"``).  Only meaningful for
            AWS; S3-compatible stores like ObjectScale ignore this.
        s3_endpoint: S3-compatible endpoint URL (e.g.
            ``"http://100.68.213.151:9020"``).  Required for non-AWS
            S3 stores like Dell ObjectScale.
        disable_tls: Whether to disable TLS for S3.
        s3_num_io_threads: Number of S3 CRT I/O threads.
        s3_max_concurrency: Maximum number of concurrent S3 RDMA requests
            (GET or PUT) in flight at once.  Higher values overlap HTTP
            latency with RDMA transfers but consume more bounce-buffer
            pages.  Default 16.
        aws_access_key_id: AWS access key (optional).
        aws_secret_access_key: AWS secret key (optional).
    """

    buffer_size: int
    buffer_device: str
    local_ip: str
    rdma_port: int
    connection_mode: str
    bucket: str
    region: str
    s3_endpoint: str
    disable_tls: bool
    s3_num_io_threads: int
    s3_max_concurrency: int
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> "S3RdmaStorageConfig":
        """Build config from LMCacheEngineConfig extra_config.

        Args:
            config: The LMCache engine config.
            metadata: The LMCache metadata.

        Returns:
            S3RdmaStorageConfig instance.

        Raises:
            AssertionError: If required fields are missing.
        """
        extra = config.extra_config
        assert extra is not None
        assert extra.get("enable_s3rdma_storage")

        buffer_device = str(extra.get("s3rdma_buffer_device", "gpu"))
        # Normalize: "gpu" and "cuda" both map to PyTorch "cuda" device.
        if buffer_device == "gpu":
            buffer_device = "cuda"
        corrected_device = get_correct_device(buffer_device, metadata.worker_id)

        buffer_size = extra.get("s3rdma_buffer_size")
        assert buffer_size is not None, "s3rdma_buffer_size is required"

        bucket = extra.get("s3rdma_bucket")
        assert bucket is not None, "s3rdma_bucket is required"

        region = str(extra.get("s3rdma_region", "us-east-1"))

        # Multi-endpoint: s3rdma_s3_endpoint may be a single endpoint (shared by
        # all TP workers) or a list, in which case worker N uses endpoint
        # N % len (round-robin). Lets each TP rank target a distinct ObjectScale
        # node.
        raw_endpoint = extra.get("s3rdma_s3_endpoint")
        assert raw_endpoint is not None, "s3rdma_s3_endpoint is required"
        s3_endpoint = _select_for_worker(
            raw_endpoint, metadata.worker_id, "s3rdma_s3_endpoint"
        )

        # Multi-NIC: s3rdma_local_ip may be a single IP or a list. With a list,
        # worker N binds the RDMA NIC owning IP N % len. Order the list so index
        # == TP rank == the worker's GPU for topology-optimal GPU<->NIC pairing.
        raw_local_ip = extra.get("s3rdma_local_ip", "0.0.0.0")
        multi_nic = isinstance(raw_local_ip, (list, tuple)) and len(raw_local_ip) > 1
        local_ip = _select_for_worker(
            raw_local_ip, metadata.worker_id, "s3rdma_local_ip"
        )

        base_port = int(extra.get("s3rdma_port", 7471))
        connection_mode = str(extra.get("s3rdma_connection_mode", "RC"))
        # RC listener port. The RDMA descriptor sent to the server carries NO
        # port (required for ObjectScale compatibility), so the server connects
        # back to the client's advertised GID at a FIXED port == base_port.
        #   - Single shared NIC (scalar local_ip): all TP workers share one IP,
        #     so each needs a distinct port (base_port + worker_id) to avoid a
        #     bind conflict. (Only works with an endpoint that learns the port
        #     out-of-band, e.g. s3_rdma_endpoint with
        #     include_port_in_rdma_descriptor.)
        #   - Multi-NIC (list local_ip): each worker has its own IP, so all
        #     listen on base_port — and MUST, so the server's fixed-port
        #     connect-back reaches every worker (not just worker 0). This is the
        #     ObjectScale-compatible multi-NIC path (ISGFGE-3712).
        if connection_mode == "RC":
            rdma_port = base_port if multi_nic else base_port + metadata.worker_id
        else:
            rdma_port = base_port

        logger.info(
            "S3RdmaStorageConfig: worker %d -> NIC %s, endpoint %s (mode=%s, "
            "port=%d)",
            metadata.worker_id,
            local_ip,
            s3_endpoint,
            connection_mode,
            rdma_port,
        )

        return S3RdmaStorageConfig(
            buffer_size=int(buffer_size),
            buffer_device=corrected_device,
            local_ip=local_ip,
            rdma_port=rdma_port,
            connection_mode=connection_mode,
            bucket=str(bucket),
            region=str(region),
            s3_endpoint=str(s3_endpoint),
            disable_tls=bool(extra.get("disable_tls", True)),
            s3_num_io_threads=int(extra.get("s3_num_io_threads", 64)),
            s3_max_concurrency=int(extra.get("s3rdma_max_concurrency", 16)),
            aws_access_key_id=extra.get("aws_access_key_id"),
            aws_secret_access_key=extra.get("aws_secret_access_key"),
        )


@dataclass
class S3RdmaKeyMetadata:
    """Metadata stored per cached key.

    Attributes:
        shape: Tensor shape of the KV chunk.
        dtype: Tensor dtype.
        fmt: Memory format.
        address: Allocator page address.
        pin_count: Number of active pins.
    """

    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: Optional[MemoryFormat] = None
    address: int = 0
    pin_count: int = 0

    def pin(self) -> bool:
        """Increment pin count."""
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        """Decrement pin count."""
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        """Return True if pinned."""
        return self.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """Return True if not pinned."""
        return not self.is_pinned


class S3RdmaStorageBackend(AllocatorBackendInterface):
    """S3 storage backend with RDMA-accelerated data transfer via libibverbs.

    Follows the NixlStorageBackend pattern: owns a bounce buffer, wraps it
    in ``PagedTensorMemoryAllocator``, returns ``self`` from
    ``get_allocator_backend()``, and manages S3 RDMA transfers via
    ``S3RdmaClientWrapper``.

    The S3 HTTP requests serve as the control plane. The ``x-rdma-info``
    header carries the RDMA descriptor, the body is empty (Content-Length: 0),
    and the ObjectScale server performs RDMA READ (PUT) or RDMA WRITE (GET)
    directly into/from the registered bounce buffer.

    Args:
        s3rdma_config: S3 RDMA configuration.
        config: LMCache engine config.
        metadata: LMCache metadata.
        loop: asyncio event loop.
    """

    def __init__(
        self,
        s3rdma_config: S3RdmaStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(dst_device=s3rdma_config.buffer_device)

        self.loop = loop
        self.key_lock = threading.RLock()
        self.progress_lock = threading.RLock()
        self.progress_set: Set[CacheEngineKey] = set()

        # Allocate bounce buffer (mirrors NixlStorageBackend.initialize_allocator)
        self.memory_allocator = self._initialize_allocator(
            s3rdma_config, config, metadata
        )

        # Initialize RDMA client and register bounce buffer pages
        rdma_config = S3RdmaConfig(
            local_ip=s3rdma_config.local_ip,
            rdma_port=s3rdma_config.rdma_port,
            connection_mode=s3rdma_config.connection_mode,
        )
        self._rdma_client = S3RdmaClientWrapper(rdma_config)

        is_gpu = s3rdma_config.buffer_device != "cpu"
        n_pages = self._rdma_client.register_pages(
            self.memory_allocator.buffer_ptr,
            self.memory_allocator.buffer_size,
            self.memory_allocator.align_bytes,
            is_gpu=is_gpu,
        )
        logger.info(
            "S3RdmaStorageBackend: registered %d RDMA pages (%d bytes each, device=%s)",
            n_pages,
            self.memory_allocator.align_bytes,
            s3rdma_config.buffer_device,
        )

        # Concurrency limiter for S3 RDMA requests
        self._s3_semaphore = asyncio.Semaphore(s3rdma_config.s3_max_concurrency)
        logger.info(
            "S3RdmaStorageBackend: max concurrent S3 RDMA requests = %d",
            s3rdma_config.s3_max_concurrency,
        )

        # Initialize S3 CRT client (control plane only)
        parsed = urlparse(s3rdma_config.s3_endpoint)
        self._s3_host = (
            f"{parsed.hostname}:{parsed.port}" if parsed.port else str(parsed.hostname)
        )
        self._s3_bucket = s3rdma_config.bucket
        self._s3_region = s3rdma_config.region
        self._s3_client, self._credentials_provider = self._init_s3_client(
            s3rdma_config
        )

        # Initialize chunk metadata
        self.meta_shape = torch.Size(metadata.kv_shape)
        self.meta_dtype = metadata.kv_dtype
        self.meta_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )

        # Cache policy and key tracking
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.key_dict = self.cache_policy.init_mutable_mapping()

    def _initialize_allocator(
        self,
        s3rdma_config: S3RdmaStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> PagedTensorMemoryAllocator:
        """Allocate bounce buffer and create paged allocator.

        Args:
            s3rdma_config: S3 RDMA configuration.
            config: LMCache engine config.
            metadata: LMCache metadata.

        Returns:
            PagedTensorMemoryAllocator wrapping the bounce buffer.
        """
        device = s3rdma_config.buffer_device

        if device == "cpu":
            self.buffer = _allocate_cpu_memory(s3rdma_config.buffer_size)
            self.free_pinned_buffer = True
        else:
            base_buffer, self.buffer = _allocate_gpu_memory(
                s3rdma_config.buffer_size, device
            )
            torch.cuda.set_device(device)
            self.base_buffer = base_buffer
            self.free_pinned_buffer = False

        return PagedTensorMemoryAllocator(
            self.buffer,
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
            MemoryFormat.KV_2LTD,
        )

    @staticmethod
    def _init_s3_client(
        s3rdma_config: S3RdmaStorageConfig,
    ) -> tuple:
        """Initialize the AWS CRT S3 client for control-plane HTTP requests.

        Args:
            s3rdma_config: S3 RDMA configuration.

        Returns:
            Tuple of (S3Client, credentials_provider).
        """
        event_loop_group = io.EventLoopGroup(s3rdma_config.s3_num_io_threads)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

        if s3rdma_config.aws_access_key_id and s3rdma_config.aws_secret_access_key:
            credentials_provider = auth.AwsCredentialsProvider.new_static(
                s3rdma_config.aws_access_key_id,
                s3rdma_config.aws_secret_access_key,
            )
        else:
            credentials_provider = auth.AwsCredentialsProvider.new_default_chain(
                client_bootstrap
            )

        tls_mode = (
            s3.S3RequestTlsMode.DISABLED
            if s3rdma_config.disable_tls
            else s3.S3RequestTlsMode.ENABLED
        )

        # Disable multipart uploads: set threshold to max so CRT sends a
        # single PUT request (required for x-rdma-info header protocol).
        s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region=s3rdma_config.region,
            tls_mode=tls_mode,
            part_size=0,
            multipart_upload_threshold=2**63 - 1,
        )

        return s3_client, credentials_provider

    def _format_object_key(self, key: CacheEngineKey) -> str:
        """Format a CacheEngineKey as a path-style S3 object path.

        Uses path-style addressing (``/<bucket>/<key>``) which is
        required for custom S3-compatible endpoints like ObjectScale.

        Args:
            key: The cache engine key.

        Returns:
            URL-encoded S3 object path (e.g.
            ``/my-bucket/model_layer_chunk``).
        """
        key_str = key.to_string()
        flat_key_str = key_str.replace("/", "_").replace("@", "_")
        return "/" + self._s3_bucket + "/" + url_quote(flat_key_str, safe="")

    # --- AllocatorBackendInterface ---

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> MemoryAllocatorInterface:
        """Return the already-initialized memory allocator.

        Args:
            config: LMCache engine config (unused, allocator already created).
            metadata: LMCache metadata (unused).

        Returns:
            The PagedTensorMemoryAllocator.
        """
        return self.memory_allocator

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """Return the underlying memory allocator.

        Returns:
            The PagedTensorMemoryAllocator.
        """
        return self.memory_allocator

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate a MemoryObj from the bounce buffer.

        Args:
            shapes: Tensor shape(s).
            dtypes: Tensor dtype(s).
            fmt: Memory format.
            eviction: Whether eviction is allowed.
            busy_loop: Whether to busy-wait for space.

        Returns:
            MemoryObj or None if allocation fails.
        """
        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """Batch-allocate MemoryObjs from the bounce buffer.

        Args:
            shapes: Tensor shape(s).
            dtypes: Tensor dtype(s).
            batch_size: Number to allocate.
            fmt: Memory format.
            eviction: Whether eviction is allowed.
            busy_loop: Whether to busy-wait for space.

        Returns:
            List of MemoryObjs or None if allocation fails.
        """
        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    def get_allocator_backend(self) -> "S3RdmaStorageBackend":
        """Return self as the allocator backend.

        Returns:
            self.
        """
        return self

    # --- StorageBackendInterface ---

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in the local key dict.

        Args:
            key: The key to check.
            pin: Whether to pin the key.

        Returns:
            True if key exists.
        """
        with self.key_lock:
            if key in self.key_dict:
                if pin:
                    self.key_dict[key].pin()
                return True
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check if key is in ongoing put tasks.

        Args:
            key: The key to check.

        Returns:
            True if the key is being written.
        """
        with self.progress_lock:
            return key in self.progress_set

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Submit a batch of PUT operations via RDMA.

        For each key, generates an ``x-rdma-info`` descriptor and sends an
        HTTP PUT with empty body. ObjectScale performs RDMA READ from the
        bounce buffer.

        Args:
            keys: Cache engine keys.
            memory_objs: MemoryObjs allocated in the bounce buffer.
            transfer_spec: Unused.
            on_complete_callback: Optional per-key completion callback.
        """
        # Increment ref_count so pages stay alive while the async RDMA
        # PUT is in flight.  storage_manager.batched_put() will call
        # ref_count_down() immediately after we return, bringing each
        # page from 2 -> 1.  The async _put_one() finally-block brings
        # it from 1 -> 0 once the RDMA READ completes.
        # (Matches the LocalCPUBackend pattern at local_cpu_backend.py:158.)
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key)
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()

        coro = self._mem_to_storage(keys, memory_objs, on_complete_callback)
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        except Exception:
            coro.close()
            # Undo ref_count_up if async submission fails so pages
            # are not leaked at ref_count=1 forever.
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()
            with self.progress_lock:
                for key in keys:
                    self.progress_set.discard(key)
            raise

    async def _mem_to_storage(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Async PUT: bounce buffer -> S3 via RDMA (concurrent).

        Issues concurrent HTTP PUT requests (up to ``s3_max_concurrency``)
        to overlap HTTP control-plane latency with RDMA data transfers.

        Args:
            keys: Cache engine keys.
            memory_objs: MemoryObjs in the bounce buffer.
            on_complete_callback: Optional per-key callback.
        """
        start_time = time.time()

        async def _put_one(key: CacheEngineKey, memory_obj: MemoryObj) -> None:
            async with self._s3_semaphore:
                try:
                    rdma_desc = self._rdma_client.prepare_put(
                        memory_obj.data_ptr, memory_obj.get_physical_size()
                    )
                    await self._s3_rdma_put(
                        key, rdma_desc, memory_obj.get_physical_size()
                    )

                    with self.key_lock:
                        self.key_dict[key] = S3RdmaKeyMetadata(
                            shape=memory_obj.meta.shape,
                            dtype=memory_obj.meta.dtype,
                            fmt=memory_obj.meta.fmt,
                            address=memory_obj.meta.address,
                        )
                        self.cache_policy.update_on_put(key)

                    if on_complete_callback is not None:
                        try:
                            on_complete_callback(key)
                        except Exception:
                            logger.exception("on_complete_callback raised")
                except Exception:
                    logger.exception("S3 RDMA PUT failed for %s", key.to_string())
                finally:
                    with self.progress_lock:
                        self.progress_set.discard(key)
                    memory_obj.ref_count_down()

        await asyncio.gather(
            *(_put_one(k, o) for k, o in zip(keys, memory_objs, strict=True))
        )

        duration = time.time() - start_time
        logger.debug(
            "S3 RDMA PUT %d keys in %.4f s (concurrent)",
            len(keys),
            duration,
        )

    async def _s3_rdma_put(
        self, key: CacheEngineKey, rdma_desc: str, data_size: int
    ) -> None:
        """Send HTTP PUT with x-rdma-info header, empty body.

        ObjectScale reads data from our registered buffer via RDMA READ.

        Args:
            key: Cache engine key.
            rdma_desc: The x-rdma-info descriptor string.
            data_size: Size of data being uploaded.

        Raises:
            RuntimeError: On S3 upload failure.
        """
        headers = HttpHeaders()
        headers.add("Host", self._s3_host)
        headers.add("Content-Length", "0")
        headers.add("Content-Type", "application/octet-stream")
        headers.add("x-rdma-info", rdma_desc)

        path = self._format_object_key(key)
        req = HttpRequest("PUT", path, headers)

        result: dict = {"err": None, "status": None, "rdma_reply": None}

        def on_headers(
            status_code: int = 0, headers: Optional[list] = None, **kwargs: Any
        ) -> None:
            result["status"] = status_code
            for name, value in (headers or []):
                if name.lower() == "x-amz-rdma-reply":
                    result["rdma_reply"] = value

        def on_done(error: Any = None, status_code: Any = None, **kwargs: Any) -> None:
            result["err"] = error

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.DEFAULT,
            operation_name="PutObject",
            request=req,
            on_headers=on_headers,
            credential_provider=self._credentials_provider,
            region=self._s3_region,
            on_done=on_done,
        )

        await asyncio.wrap_future(s3_req.finished_future)

        if result["err"] or result["status"] not in (200, 201):
            raise RuntimeError(
                f"S3 RDMA PUT failed for {key.to_string()}: "
                f"status={result['status']}, err={result['err']}"
            )

        if not S3RdmaClientWrapper.parse_rdma_reply(result["rdma_reply"]):
            raise RuntimeError(
                f"S3 RDMA PUT for {key.to_string()}: x-amz-rdma-reply "
                f"not 200 (got {result['rdma_reply']}). "
                f"RDMA transfer was not completed by the server."
            )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking GET: S3 -> RDMA WRITE -> bounce buffer.

        Args:
            key: The key to retrieve.

        Returns:
            MemoryObj in the bounce buffer, or None on failure.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._storage_to_mem([key]), self.loop
        )
        obj_list = future.result()
        return obj_list[0]

    def batched_get_blocking(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Blocking batched GET.

        Args:
            keys: The keys to retrieve.

        Returns:
            List of MemoryObjs (None for failures).
        """
        if not keys:
            return []
        future = asyncio.run_coroutine_threadsafe(self._storage_to_mem(keys), self.loop)
        return future.result()

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Non-blocking batched GET.

        Args:
            lookup_id: Lookup identifier.
            keys: The keys to retrieve.
            transfer_spec: Unused.

        Returns:
            List of successfully retrieved MemoryObjs.
        """
        obj_list = await self._storage_to_mem(keys)
        return [obj for obj in obj_list if obj is not None]

    async def _storage_to_mem(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        """Async GET: S3 -> RDMA WRITE -> bounce buffer.

        Issues concurrent HTTP GET requests (up to ``s3_max_concurrency``)
        to overlap HTTP control-plane latency with RDMA data transfers.
        Each request allocates a bounce-buffer page, generates an
        ``x-rdma-info`` descriptor, and sends an HTTP GET.  ObjectScale
        performs RDMA WRITE directly into the bounce buffer.

        Args:
            keys: Cache engine keys to retrieve.

        Returns:
            List of MemoryObjs (None for failures), in the same order as
            *keys*.
        """
        start_time = time.time()
        n_keys = len(keys)

        # Slot for each key's landing page; filled in by the per-key coroutine
        # on success so the returned list preserves request order.
        objs: list[Optional[MemoryObj]] = [None] * n_keys

        async def _get_one(idx: int) -> bool:
            """Fetch a single chunk.  Returns True on success.

            Allocation of the landing page and generation of its RDMA
            descriptor happen *inside* the concurrency semaphore (mirroring
            the PUT path's ``prepare_put`` placement).  This overlaps the
            per-chunk setup with other chunks' in-flight HTTP/RDMA transfers
            and bounds the number of pages held at once to the concurrency
            limit plus already-completed hits — a page for a failed GET is
            returned to the pool immediately rather than being held until the
            whole batch finishes.
            """
            async with self._s3_semaphore:
                obj = self.memory_allocator.allocate(
                    self.meta_shape, self.meta_dtype, self.meta_fmt
                )
                if obj is None:
                    logger.warning("Failed to allocate memory for S3 RDMA GET")
                    return False
                try:
                    desc = self._rdma_client.prepare_get(
                        obj.data_ptr, obj.get_physical_size()
                    )
                    await self._s3_rdma_get(keys[idx], desc)
                    with self.key_lock:
                        self.cache_policy.update_on_hit(keys[idx], self.key_dict)
                    objs[idx] = obj
                    return True
                except Exception:
                    logger.exception(
                        "S3 RDMA GET failed for %s", keys[idx].to_string()
                    )
                    self.memory_allocator.free(obj)
                    return False

        results = await asyncio.gather(*(_get_one(i) for i in range(n_keys)))

        obj_list: list[Optional[MemoryObj]] = [
            objs[i] if results[i] else None for i in range(n_keys)
        ]

        duration = time.time() - start_time
        logger.debug(
            "S3 RDMA GET %d keys in %.4f s (concurrent)",
            n_keys,
            duration,
        )
        return obj_list

    async def _s3_rdma_get(self, key: CacheEngineKey, rdma_desc: str) -> None:
        """Send HTTP GET with x-rdma-info header.

        ObjectScale writes data into our registered buffer via RDMA WRITE.
        The HTTP response body is empty (Content-Length: 0).

        Args:
            key: Cache engine key.
            rdma_desc: The x-rdma-info descriptor string.

        Raises:
            RuntimeError: On S3 download failure.
        """
        headers = HttpHeaders()
        headers.add("Host", self._s3_host)
        headers.add("x-rdma-info", rdma_desc)

        path = self._format_object_key(key)
        req = HttpRequest("GET", path, headers)

        result: dict = {"err": None, "status": None, "rdma_reply": None}

        def on_headers(
            status_code: int = 0, headers: Optional[list] = None, **kwargs: Any
        ) -> None:
            result["status"] = status_code
            for name, value in (headers or []):
                if name.lower() == "x-amz-rdma-reply":
                    result["rdma_reply"] = value

        def on_done(error: Any = None, status_code: Any = None, **kwargs: Any) -> None:
            result["err"] = error

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.DEFAULT,
            operation_name="GetObject",
            request=req,
            on_headers=on_headers,
            credential_provider=self._credentials_provider,
            region=self._s3_region,
            on_done=on_done,
        )

        await asyncio.wrap_future(s3_req.finished_future)

        if result["err"] or result["status"] not in (200, 206):
            raise RuntimeError(
                f"S3 RDMA GET failed for {key.to_string()}: "
                f"status={result['status']}, err={result['err']}"
            )

        if not S3RdmaClientWrapper.parse_rdma_reply(result["rdma_reply"]):
            raise RuntimeError(
                f"S3 RDMA GET for {key.to_string()}: x-amz-rdma-reply "
                f"not 200 (got {result['rdma_reply']}). "
                f"RDMA transfer was not completed by the server."
            )

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a key from the local key dict.

        Args:
            key: The key to remove.
            force: Whether to force-evict.

        Returns:
            True if the key was removed.
        """
        with self.key_lock:
            metadata = self.key_dict.pop(key, None)
            if metadata is None:
                return False
            if force:
                self.cache_policy.update_on_force_evict(key)
        return True

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a key to prevent eviction.

        Args:
            key: The key to pin.

        Returns:
            True if the key was found and pinned.
        """
        with self.key_lock:
            if key in self.key_dict:
                self.key_dict[key].pin()
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a key.

        Args:
            key: The key to unpin.

        Returns:
            True if the key was found and unpinned.
        """
        with self.key_lock:
            if key in self.key_dict:
                self.key_dict[key].unpin()
                return True
            return False

    def close(self) -> None:
        """Release all resources: RDMA client, allocator, bounce buffer."""
        self._rdma_client.close()
        self.memory_allocator.close()
        if self.free_pinned_buffer:
            _free_cpu_memory(self.buffer)

    def __str__(self) -> str:
        return "S3RdmaStorageBackend"

    @staticmethod
    def CreateS3RdmaStorageBackend(
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        metadata: LMCacheMetadata,
    ) -> "S3RdmaStorageBackend":
        """Factory method to create an S3RdmaStorageBackend.

        Args:
            config: LMCache engine config.
            loop: asyncio event loop.
            metadata: LMCache metadata.

        Returns:
            S3RdmaStorageBackend instance.
        """
        s3rdma_config = S3RdmaStorageConfig.from_cache_engine_config(config, metadata)
        return S3RdmaStorageBackend(s3rdma_config, config, metadata, loop)

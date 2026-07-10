# SPDX-License-Identifier: Apache-2.0
"""Tests for the S3RdmaStorageBackend PUT path ref_count lifecycle.

Reproduces the double-free bug where ``storage_manager.batched_put()``
calls ``ref_count_down()`` immediately after ``batched_submit_put_task()``
returns, while the async ``_put_one()`` coroutine is still in flight
and will call ``ref_count_down()`` again in its ``finally`` block.

Without the fix (``ref_count_up()`` in ``batched_submit_put_task``),
ref_count goes 1 -> 0 (freed!) -> -1 (double-free).

With the fix, ref_count goes 1 -> 2 -> 1 -> 0 (freed once, correctly).
"""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import logging
import sys
import threading

# Inject a stub ``lmcache.c_ops`` when the CUDA extension was not compiled.
if "lmcache.c_ops" not in sys.modules:
    try:
        # Standard
        import importlib

        importlib.import_module("lmcache.c_ops")
    except (ImportError, ModuleNotFoundError):
        sys.modules["lmcache.c_ops"] = MagicMock()

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_allocators import PagedTensorMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.s3rdma_storage_backend import (
    S3RdmaStorageBackend,
    S3RdmaStorageConfig,
)

_ASYNC_TIMEOUT = 5.0  # seconds — generous for slow CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key(idx: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=idx,
        dtype=torch.bfloat16,
    )


def _make_allocator(
    kv_shape: tuple = (4, 2, 256, 8, 128),
) -> PagedTensorMemoryAllocator:
    """Create a small CPU-backed paged allocator (enough for a few pages)."""
    shape = torch.Size(kv_shape)
    dtype = torch.bfloat16
    page_bytes = 1
    for d in shape:
        page_bytes *= d
    page_bytes *= dtype.itemsize
    # Allocate space for 4 pages
    buf = torch.empty(page_bytes * 4, dtype=torch.uint8)
    return PagedTensorMemoryAllocator(buf, [shape], [dtype], MemoryFormat.KV_2LTD)


def _wait_for_coroutine(
    loop: asyncio.AbstractEventLoop,
    coro,
    timeout: float = _ASYNC_TIMEOUT,
):
    """Submit *coro* on *loop* and block until it finishes or times out."""
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)


@pytest.fixture()
def event_loop():
    """A dedicated asyncio event loop running in a background thread."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


@pytest.fixture()
def backend(event_loop):
    """Build an S3RdmaStorageBackend with all RDMA/S3 internals mocked.

    The ``_done_event`` attribute (asyncio.Event on the backend's loop)
    is set after every ``_mem_to_storage`` call completes so tests can
    wait deterministically instead of using ``time.sleep()``.
    """
    allocator = _make_allocator()

    with (
        patch.object(
            S3RdmaStorageBackend, "__init__", lambda self, *a, **kw: None
        ),
    ):
        be = S3RdmaStorageBackend.__new__(S3RdmaStorageBackend)

    # Wire up the attributes that batched_submit_put_task / _mem_to_storage need
    be.loop = event_loop
    be.memory_allocator = allocator
    be.key_lock = threading.RLock()
    be.progress_lock = threading.RLock()
    be.progress_set = set()
    be._s3_semaphore = asyncio.Semaphore(16)
    be.cache_policy = MagicMock()
    be.key_dict = {}

    # Mock the RDMA client so prepare_put returns a dummy descriptor
    be._rdma_client = MagicMock()
    be._rdma_client.prepare_put.return_value = "fake-rdma-descriptor"

    # Mock _s3_rdma_put to succeed instantly
    be._s3_rdma_put = AsyncMock()

    # Wrap _mem_to_storage so tests can wait for completion via _done_event
    be._done_event = asyncio.Event()
    _original_mem_to_storage = S3RdmaStorageBackend._mem_to_storage

    async def _signalling_mem_to_storage(self, *args, **kwargs):
        try:
            await _original_mem_to_storage(self, *args, **kwargs)
        finally:
            self._done_event.set()

    be._mem_to_storage = _signalling_mem_to_storage.__get__(be)

    return be


def _wait_for_put(backend, timeout: float = _ASYNC_TIMEOUT) -> None:
    """Block until the backend's ``_mem_to_storage`` wrapper sets _done_event."""
    _wait_for_coroutine(backend.loop, backend._done_event.wait(), timeout)
    # Reset for the next call
    backend._done_event.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPutRefCountLifecycle:
    """Verify the batched_submit_put_task ref_count lifecycle.

    These tests reproduce the exact call pattern from
    ``storage_manager.batched_put()``: call ``batched_submit_put_task()``,
    then immediately call ``ref_count_down()`` on each memory_obj.

    The async ``_put_one()`` coroutine also calls ``ref_count_down()``
    in its ``finally`` block when it completes.
    """

    def test_ref_count_never_negative(self, backend):
        """ref_count must never go below zero during the PUT lifecycle.

        This is the primary regression test for the double-free bug.
        The memory_management code logs a "Double free" warning when
        ref_count goes negative (then clamps it back to 0), so we
        capture log output to detect the violation.
        """
        allocator = backend.memory_allocator
        mem_obj = allocator.allocate(
            allocator.shapes, allocator.dtypes, MemoryFormat.KV_2LTD
        )
        assert mem_obj is not None
        assert mem_obj.get_ref_count() == 1

        key = _make_key(0)

        # Capture warnings from the memory_management logger
        double_free_warnings = []
        mm_logger = logging.getLogger("lmcache.v1.memory_management")
        original_warning = mm_logger.warning

        def capture_warning(msg, *args, **kwargs):
            original_warning(msg, *args, **kwargs)
            if "Double free" in str(msg):
                double_free_warnings.append(str(msg))

        mm_logger.warning = capture_warning

        try:
            # --- Simulate storage_manager.batched_put() ---
            backend.batched_submit_put_task([key], [mem_obj])
            mem_obj.ref_count_down()

            _wait_for_put(backend)

            # --- Assertions ---
            assert len(double_free_warnings) == 0, (
                f"Double-free detected! {len(double_free_warnings)} "
                f"warning(s): {double_free_warnings}"
            )
            assert mem_obj.get_ref_count() == 0
        finally:
            mm_logger.warning = original_warning

    def test_page_freed_exactly_once(self, backend):
        """The page must be returned to free_blocks exactly once.

        A double-free would cause the same page to appear in free_blocks
        multiple times, leading to two callers getting the same page.
        """
        allocator = backend.memory_allocator
        initial_free = len(allocator.free_blocks)

        mem_obj = allocator.allocate(
            allocator.shapes, allocator.dtypes, MemoryFormat.KV_2LTD
        )
        assert mem_obj is not None
        assert len(allocator.free_blocks) == initial_free - 1

        key = _make_key(1)

        backend.batched_submit_put_task([key], [mem_obj])
        mem_obj.ref_count_down()

        _wait_for_put(backend)

        after_put_free = len(allocator.free_blocks)
        assert after_put_free == initial_free, (
            f"Expected {initial_free} free blocks after PUT completes, "
            f"got {after_put_free}. "
            "A double-free would add the page twice."
        )

    def test_batched_put_multi_key(self, backend):
        """Multiple keys in a single batch must each have correct lifecycle."""
        allocator = backend.memory_allocator
        n_keys = 3
        initial_free = len(allocator.free_blocks)

        mem_objs = allocator.batched_allocate(
            allocator.shapes, allocator.dtypes, n_keys, MemoryFormat.KV_2LTD
        )
        assert mem_objs is not None
        assert len(mem_objs) == n_keys
        for obj in mem_objs:
            assert obj.get_ref_count() == 1

        keys = [_make_key(i) for i in range(n_keys)]

        backend.batched_submit_put_task(keys, mem_objs)
        for obj in mem_objs:
            obj.ref_count_down()

        _wait_for_put(backend)

        for obj in mem_objs:
            assert obj.get_ref_count() == 0, (
                f"ref_count is {obj.get_ref_count()}, expected 0"
            )
        assert len(allocator.free_blocks) == initial_free

    def test_page_alive_during_async_put(self, backend):
        """The bounce buffer page must remain valid while RDMA PUT is in flight.

        This verifies that the page is NOT freed between
        batched_submit_put_task() return and _put_one() completion.
        """
        allocator = backend.memory_allocator

        mem_obj = allocator.allocate(
            allocator.shapes, allocator.dtypes, MemoryFormat.KV_2LTD
        )
        assert mem_obj is not None

        # Make _s3_rdma_put block until we release it
        put_started = asyncio.Event()
        put_gate = asyncio.Event()

        async def slow_put(*args, **kwargs):
            put_started.set()
            await put_gate.wait()

        backend._s3_rdma_put = slow_put

        key = _make_key(10)

        backend.batched_submit_put_task([key], [mem_obj])
        mem_obj.ref_count_down()

        # Wait for the async PUT to start (deterministic, no sleep)
        _wait_for_coroutine(backend.loop, put_started.wait())

        # While RDMA PUT is in flight, ref_count should be 1 (not 0!)
        assert mem_obj.get_ref_count() == 1, (
            f"ref_count is {mem_obj.get_ref_count()} while PUT is in flight. "
            "Expected 1 — the page would be prematurely freed at ref_count=0."
        )

        # Release the gate so _put_one can finish
        backend.loop.call_soon_threadsafe(put_gate.set)
        _wait_for_put(backend)

        assert mem_obj.get_ref_count() == 0

    def test_ref_count_on_put_failure(self, backend):
        """Page must be freed even if the S3 PUT raises an exception."""
        allocator = backend.memory_allocator
        free_before_alloc = len(allocator.free_blocks)

        mem_obj = allocator.allocate(
            allocator.shapes, allocator.dtypes, MemoryFormat.KV_2LTD
        )
        assert mem_obj is not None
        assert len(allocator.free_blocks) == free_before_alloc - 1

        backend._s3_rdma_put = AsyncMock(
            side_effect=RuntimeError("simulated S3 failure")
        )

        key = _make_key(20)

        backend.batched_submit_put_task([key], [mem_obj])
        mem_obj.ref_count_down()

        _wait_for_put(backend)

        assert mem_obj.get_ref_count() == 0
        assert len(allocator.free_blocks) == free_before_alloc

    def test_async_submission_failure_cleans_up(self, backend):
        """If asyncio.run_coroutine_threadsafe() raises, ref_count_up must be undone.

        When the event loop is dead or broken, the async coroutine is
        never submitted.  The try-except in batched_submit_put_task()
        must call ref_count_down() for each page to avoid leaking them
        at ref_count=1 forever.
        """
        allocator = backend.memory_allocator
        free_before_alloc = len(allocator.free_blocks)

        mem_obj = allocator.allocate(
            allocator.shapes, allocator.dtypes, MemoryFormat.KV_2LTD
        )
        assert mem_obj is not None
        assert mem_obj.get_ref_count() == 1

        key = _make_key(30)

        # Simulate a dead event loop by patching run_coroutine_threadsafe
        with patch(
            "lmcache.v1.storage_backend.s3rdma_storage_backend"
            ".asyncio.run_coroutine_threadsafe",
            side_effect=RuntimeError("event loop is closed"),
        ):
            with pytest.raises(RuntimeError, match="event loop is closed"):
                backend.batched_submit_put_task([key], [mem_obj])

        # ref_count should be back to 1 (ref_count_up was undone)
        # so that storage_manager's ref_count_down() will bring it to 0
        assert mem_obj.get_ref_count() == 1, (
            f"ref_count is {mem_obj.get_ref_count()} after failed submission. "
            "Expected 1 — ref_count_up should have been undone."
        )

        # progress_set should also be cleaned up
        assert key not in backend.progress_set

        # Simulate storage_manager's ref_count_down (brings 1 -> 0, frees page)
        mem_obj.ref_count_down()
        assert mem_obj.get_ref_count() == 0
        assert len(allocator.free_blocks) == free_before_alloc

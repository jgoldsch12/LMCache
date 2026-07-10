# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S3 RDMA bindings and storage backend.

Tests use mocking to avoid requiring RDMA hardware or S3 infrastructure.
"""

# Standard
from unittest.mock import MagicMock, patch
import sys

# Inject a stub ``lmcache.c_ops`` when the CUDA extension was not compiled
# (e.g. CUDA version mismatch).  Must happen before any lmcache imports that
# transitively pull in ``memory_management -> c_ops``.
if "lmcache.c_ops" not in sys.modules:
    try:
        # Standard
        import importlib

        importlib.import_module("lmcache.c_ops")
    except (ImportError, ModuleNotFoundError):
        sys.modules["lmcache.c_ops"] = MagicMock()

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.s3rdma_bindings import (
    S3RDMA_SUCCESS,
    S3RdmaClientWrapper,
    S3RdmaConfig,
    is_s3rdma_available,
)
from lmcache.v1.storage_backend.s3rdma_storage_backend import _select_for_worker


class TestSelectForWorker:
    """Tests for per-TP-worker multi-NIC / multi-endpoint selection."""

    def test_scalar_returns_same_for_all_workers(self) -> None:
        """A scalar value is shared by every worker (legacy behavior)."""
        for wid in range(4):
            assert _select_for_worker("10.0.0.1", wid, "f") == "10.0.0.1"

    def test_list_round_robins_by_worker_id(self) -> None:
        """A list selects element worker_id % len."""
        ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
        assert [_select_for_worker(ips, w, "f") for w in range(4)] == ips

    def test_list_wraps_when_more_workers_than_entries(self) -> None:
        """worker_id >= len wraps round-robin."""
        eps = ["a", "b"]
        assert _select_for_worker(eps, 2, "f") == "a"
        assert _select_for_worker(eps, 3, "f") == "b"

    def test_single_element_list(self) -> None:
        """A one-element list behaves like a scalar for all workers."""
        for wid in range(3):
            assert _select_for_worker(["only"], wid, "f") == "only"

    def test_empty_list_raises(self) -> None:
        """An empty list is a configuration error."""
        with pytest.raises(AssertionError, match="non-empty"):
            _select_for_worker([], 0, "s3rdma_local_ip")

    def test_non_string_list_elements_are_stringified(self) -> None:
        """Elements are coerced to str (YAML may parse unquoted values)."""
        assert _select_for_worker([1, 2, 3], 1, "f") == "2"


class TestS3RdmaConfig:
    """Tests for S3RdmaConfig dataclass."""

    def test_default_values(self) -> None:
        """S3RdmaConfig should have sensible defaults."""
        config = S3RdmaConfig(local_ip="192.168.1.1")
        assert config.local_ip == "192.168.1.1"
        assert config.rdma_port == 7471
        assert config.connection_mode == "RC"

    def test_custom_values(self) -> None:
        """S3RdmaConfig should accept custom values."""
        config = S3RdmaConfig(
            local_ip="10.0.0.1",
            rdma_port=9000,
            connection_mode="DC",
        )
        assert config.local_ip == "10.0.0.1"
        assert config.rdma_port == 9000
        assert config.connection_mode == "DC"


class TestS3RdmaClientWrapperImportFallback:
    """Tests for build-time fallback when C++ extension is unavailable."""

    def test_is_s3rdma_available_returns_bool(self) -> None:
        """is_s3rdma_available() should return a bool."""
        result = is_s3rdma_available()
        assert isinstance(result, bool)

    @patch("lmcache.v1.storage_backend.s3rdma_bindings._S3RDMA_AVAILABLE", False)
    def test_constructor_raises_import_error_when_unavailable(self) -> None:
        """S3RdmaClientWrapper should raise ImportError if C++ ext missing."""
        with pytest.raises(ImportError, match="lmcache_s3rdma C\\+\\+ extension"):
            S3RdmaClientWrapper(S3RdmaConfig(local_ip="0.0.0.0"))


class TestParseRdmaReply:
    """Tests for the static parse_rdma_reply method."""

    def test_success(self) -> None:
        """parse_rdma_reply('200') should return True."""
        assert S3RdmaClientWrapper.parse_rdma_reply("200") is True

    def test_success_with_whitespace(self) -> None:
        """parse_rdma_reply(' 200 ') should return True."""
        assert S3RdmaClientWrapper.parse_rdma_reply(" 200 ") is True

    def test_failure_non_200(self) -> None:
        """parse_rdma_reply('500') should return False."""
        assert S3RdmaClientWrapper.parse_rdma_reply("500") is False

    def test_failure_none(self) -> None:
        """parse_rdma_reply(None) should return False."""
        assert S3RdmaClientWrapper.parse_rdma_reply(None) is False

    def test_failure_empty(self) -> None:
        """parse_rdma_reply('') should return False."""
        assert S3RdmaClientWrapper.parse_rdma_reply("") is False

    def test_failure_whitespace_only(self) -> None:
        """parse_rdma_reply('   ') should return False."""
        assert S3RdmaClientWrapper.parse_rdma_reply("   ") is False

    def test_failure_non_numeric(self) -> None:
        """parse_rdma_reply('ok') should return False."""
        assert S3RdmaClientWrapper.parse_rdma_reply("ok") is False

    def test_failure_zero(self) -> None:
        """parse_rdma_reply('0') should return False."""
        assert S3RdmaClientWrapper.parse_rdma_reply("0") is False


class TestS3RdmaClientWrapperWithMock:
    """Tests for S3RdmaClientWrapper with mocked C++ layer."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        """Create a mocked native client."""
        client = MagicMock()
        client.register_pages.return_value = 16
        client.register_region.return_value = (0x7F0000000000, 8 * 1024 * 1024)
        client.deregister_all.return_value = None
        client.prepare_put.return_value = (
            "00007f0000000000:00800000:0000abcd:0001:000000:1:"
            "fe800000000000000000000000000001"
        )
        client.prepare_get.return_value = (
            "00007f0000000000:00800000:0000abcd:0001:000000:1:"
            "fe800000000000000000000000000001"
        )
        client.is_connected.return_value = True
        client.close.return_value = 0
        return client

    @pytest.fixture()
    def wrapper(self, mock_client: MagicMock) -> S3RdmaClientWrapper:
        """Create a wrapper with mocked internals."""
        with (
            patch("lmcache.v1.storage_backend.s3rdma_bindings._S3RDMA_AVAILABLE", True),
            patch(
                "lmcache.v1.storage_backend.s3rdma_bindings._NativeS3RdmaConfig"
            ) as mock_config_cls,
            patch(
                "lmcache.v1.storage_backend.s3rdma_bindings._NativeS3RdmaClient",
                return_value=mock_client,
            ),
        ):
            mock_config_cls.return_value = MagicMock()
            w = S3RdmaClientWrapper(S3RdmaConfig(local_ip="10.0.0.1"))
        return w

    def test_register_pages(self, wrapper: S3RdmaClientWrapper) -> None:
        """register_pages should delegate to native client."""
        n = wrapper.register_pages(
            base_ptr=0x7F0000000000,
            total_size=128 * 1024 * 1024,
            page_size=8 * 1024 * 1024,
            is_gpu=True,
        )
        assert n == 16

    def test_register_region(self, wrapper: S3RdmaClientWrapper) -> None:
        """register_region should delegate to native client."""
        ptr, size = wrapper.register_region(
            ptr=0x7F0000000000,
            size=8 * 1024 * 1024,
            is_gpu=False,
        )
        assert ptr == 0x7F0000000000
        assert size == 8 * 1024 * 1024

    def test_prepare_put(self, wrapper: S3RdmaClientWrapper) -> None:
        """prepare_put should return a descriptor string."""
        desc = wrapper.prepare_put(ptr=0x7F0000000000, size=8 * 1024 * 1024)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_prepare_get(self, wrapper: S3RdmaClientWrapper) -> None:
        """prepare_get should return a descriptor string."""
        desc = wrapper.prepare_get(ptr=0x7F0000000000, size=8 * 1024 * 1024)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_is_connected(self, wrapper: S3RdmaClientWrapper) -> None:
        """is_connected should return True when device is open."""
        assert wrapper.is_connected() is True

    def test_close(self, wrapper: S3RdmaClientWrapper) -> None:
        """close should deregister all and return success."""
        rc = wrapper.close()
        assert rc == 0

    def test_close_idempotent(self, wrapper: S3RdmaClientWrapper) -> None:
        """close should be safe to call multiple times."""
        rc1 = wrapper.close()
        rc2 = wrapper.close()
        assert rc1 == 0
        assert rc2 == S3RDMA_SUCCESS


class TestS3RdmaStorageConfig:
    """Tests for S3RdmaStorageConfig."""

    def test_from_cache_engine_config(self) -> None:
        """from_cache_engine_config should parse extra_config correctly."""
        # First Party
        from lmcache.v1.storage_backend.s3rdma_storage_backend import (
            S3RdmaStorageConfig,
        )

        config = MagicMock()
        config.extra_config = {
            "enable_s3rdma_storage": True,
            "s3rdma_buffer_size": 1024 * 1024 * 1024,
            "s3rdma_buffer_device": "cpu",
            "s3rdma_local_ip": "10.0.0.1",
            "s3rdma_port": 8000,
            "s3rdma_connection_mode": "DC",
            "s3rdma_bucket": "test-bucket",
            "s3rdma_region": "us-east-1",
            "s3rdma_s3_endpoint": "http://10.0.0.100:9020",
            "disable_tls": True,
            "s3_num_io_threads": 32,
            "aws_access_key_id": "AKTEST",
            "aws_secret_access_key": "SECRET",
        }

        metadata = MagicMock()
        metadata.worker_id = 0

        with patch(
            "lmcache.v1.storage_backend.s3rdma_storage_backend.get_correct_device",
            return_value="cpu",
        ):
            result = S3RdmaStorageConfig.from_cache_engine_config(config, metadata)

        assert result.buffer_size == 1024 * 1024 * 1024
        assert result.buffer_device == "cpu"
        assert result.local_ip == "10.0.0.1"
        assert result.rdma_port == 8000
        assert result.connection_mode == "DC"
        assert result.bucket == "test-bucket"
        assert result.region == "us-east-1"
        assert result.s3_endpoint == "http://10.0.0.100:9020"
        assert result.disable_tls is True
        assert result.s3_num_io_threads == 32
        assert result.aws_access_key_id == "AKTEST"
        assert result.aws_secret_access_key == "SECRET"

    def test_missing_required_fields_raises(self) -> None:
        """Missing required fields should raise AssertionError."""
        # First Party
        from lmcache.v1.storage_backend.s3rdma_storage_backend import (
            S3RdmaStorageConfig,
        )

        config = MagicMock()
        config.extra_config = {
            "enable_s3rdma_storage": True,
            # Missing s3rdma_buffer_size, s3rdma_bucket, s3rdma_region
        }

        metadata = MagicMock()
        metadata.worker_id = 0

        with patch(
            "lmcache.v1.storage_backend.s3rdma_storage_backend.get_correct_device",
            return_value="cpu",
        ):
            with pytest.raises(AssertionError):
                S3RdmaStorageConfig.from_cache_engine_config(config, metadata)


class TestS3RdmaKeyMetadata:
    """Tests for S3RdmaKeyMetadata."""

    def test_pin_unpin(self) -> None:
        """pin/unpin should track pin count correctly."""
        # First Party
        from lmcache.v1.storage_backend.s3rdma_storage_backend import (
            S3RdmaKeyMetadata,
        )

        meta = S3RdmaKeyMetadata()
        assert not meta.is_pinned
        assert meta.can_evict

        meta.pin()
        assert meta.is_pinned
        assert not meta.can_evict

        meta.pin()
        assert meta.pin_count == 2

        meta.unpin()
        assert meta.is_pinned

        meta.unpin()
        assert not meta.is_pinned
        assert meta.can_evict

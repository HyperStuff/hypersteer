"""Tests for hypersteer/utils/helpers.py"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from hypersteer.utils.helpers import (
    _get_current_device,
    get_device_ordinal,
    get_rank,
    get_world_size,
    is_distributed,
)


class TestGetCurrentDevice:
    """Tests for _get_current_device function."""

    def test_returns_cpu_when_no_gpu(self):
        """Test returns CPU device when no GPU is available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                device = _get_current_device()
                assert device == torch.device("cpu")

    def test_returns_mps_when_available(self):
        """Test returns MPS device when available (and no CUDA)."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                device = _get_current_device()
                assert device == torch.device("mps")


class TestDistributedHelpers:
    """Tests for distributed training helper functions."""

    def test_is_distributed_false_when_not_initialized(self):
        """Test is_distributed returns False when not initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=False):
                assert is_distributed() is False

    def test_is_distributed_false_when_not_available(self):
        """Test is_distributed returns False when not available."""
        with patch("torch.distributed.is_available", return_value=False):
            assert is_distributed() is False

    def test_get_rank_returns_zero_when_not_distributed(self):
        """Test get_rank returns 0 when not in distributed mode."""
        with patch("hypersteer.utils.helpers.is_distributed", return_value=False):
            assert get_rank() == 0

    def test_get_world_size_returns_one_when_not_distributed(self):
        """Test get_world_size returns 1 when not in distributed mode."""
        with patch("hypersteer.utils.helpers.is_distributed", return_value=False):
            assert get_world_size() == 1


class TestGetDeviceOrdinal:
    """Tests for get_device_ordinal function."""

    def test_extracts_cuda_ordinal(self):
        """Test extracting ordinal from cuda:2."""
        assert get_device_ordinal("cuda:2") == 2

    def test_extracts_cuda_zero(self):
        """Test extracting ordinal from cuda:0."""
        assert get_device_ordinal("cuda:0") == 0

    def test_returns_cpu_string(self):
        """Test returns 'cpu' for cpu device."""
        assert get_device_ordinal("cpu") == "cpu"

    def test_handles_torch_device(self):
        """Test handles torch.device object."""
        device = torch.device("cuda:3")
        assert get_device_ordinal(device) == 3

    def test_handles_cpu_torch_device(self):
        """Test handles cpu torch.device object."""
        device = torch.device("cpu")
        assert get_device_ordinal(device) == "cpu"

"""
Tests for hardware detection and configuration.

Property 17: Hardware Detection Correctness
Validates: Requirements 9.4
"""

import pytest
from hypothesis import given, strategies as st, settings

from src.utils.hardware import (
    DeviceType,
    HardwareConfig,
    detect_device,
    get_device_string,
    get_hardware_config,
    set_seed,
    get_optimal_workers,
    DEFAULT_BATCH_SIZES,
)


class TestDeviceDetection:
    """Tests for device detection functionality."""
    
    def test_detect_device_returns_valid_type(self):
        """Device detection should return a valid DeviceType."""
        device = detect_device()
        assert isinstance(device, DeviceType)
        assert device in [DeviceType.MPS, DeviceType.CUDA, DeviceType.CPU]
    
    def test_detect_device_auto_mode(self):
        """Auto mode should detect best available device."""
        device = detect_device("auto")
        assert isinstance(device, DeviceType)
    
    def test_detect_device_cpu_always_available(self):
        """CPU should always be available as fallback."""
        device = detect_device("cpu")
        assert device == DeviceType.CPU
    
    def test_detect_device_invalid_raises(self):
        """Invalid device preference should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown device type"):
            detect_device("invalid_device")


class TestDeviceString:
    """Tests for device string conversion."""
    
    def test_mps_device_string(self):
        """MPS device should return 'mps' string."""
        assert get_device_string(DeviceType.MPS) == "mps"
    
    def test_cuda_device_string(self):
        """CUDA device should return 'cuda' string."""
        assert get_device_string(DeviceType.CUDA) == "cuda"
    
    def test_cpu_device_string(self):
        """CPU device should return 'cpu' string."""
        assert get_device_string(DeviceType.CPU) == "cpu"


class TestHardwareConfig:
    """Tests for hardware configuration."""
    
    def test_get_hardware_config_returns_valid_config(self):
        """Hardware config should return complete configuration."""
        config = get_hardware_config()
        
        assert isinstance(config, HardwareConfig)
        assert isinstance(config.device, DeviceType)
        assert isinstance(config.device_name, str)
        assert config.max_memory_gb > 0
        assert isinstance(config.supports_mixed_precision, bool)
        assert isinstance(config.recommended_batch_sizes, dict)
    
    def test_hardware_config_has_required_batch_sizes(self):
        """Config should include batch sizes for all components."""
        config = get_hardware_config()
        
        required_keys = ["text_embedding", "vision_embedding", "router"]
        for key in required_keys:
            assert key in config.recommended_batch_sizes
            assert config.recommended_batch_sizes[key] > 0
    
    def test_hardware_config_cpu_fallback(self):
        """CPU config should work as fallback."""
        config = get_hardware_config(preferred_device="cpu")
        
        assert config.device == DeviceType.CPU
        assert config.device_name == "CPU"
    
    def test_hardware_config_custom_memory(self):
        """Custom memory limit should be respected."""
        config = get_hardware_config(max_memory_gb=4.0)
        
        assert config.max_memory_gb == 4.0


class TestSeedSetting:
    """Tests for random seed management."""
    
    def test_set_seed_is_reproducible(self):
        """Setting seed should produce reproducible results."""
        import torch
        import numpy as np
        import random
        
        set_seed(42)
        torch_val1 = torch.rand(1).item()
        np_val1 = np.random.rand()
        py_val1 = random.random()
        
        set_seed(42)
        torch_val2 = torch.rand(1).item()
        np_val2 = np.random.rand()
        py_val2 = random.random()
        
        assert torch_val1 == torch_val2
        assert np_val1 == np_val2
        assert py_val1 == py_val2
    
    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        import torch
        
        set_seed(42)
        val1 = torch.rand(1).item()
        
        set_seed(123)
        val2 = torch.rand(1).item()
        
        assert val1 != val2


class TestOptimalWorkers:
    """Tests for worker count optimization."""
    
    def test_optimal_workers_positive(self):
        """Optimal workers should be positive."""
        workers = get_optimal_workers()
        assert workers > 0
    
    def test_optimal_workers_bounded(self):
        """Optimal workers should be bounded (2-8)."""
        workers = get_optimal_workers()
        assert 2 <= workers <= 8


# Property-Based Tests
# Feature: adaptive-retrieval-system, Property 17: Hardware Detection Correctness

class TestHardwareDetectionProperty:
    """
    Property 17: Hardware Detection Correctness
    
    For any system with available hardware (MPS, CUDA, or CPU-only),
    the automatic hardware detection SHALL correctly identify the available
    backend and configure PyTorch to use it.
    
    Validates: Requirements 9.4
    """
    
    @given(st.sampled_from(["auto", "cpu"]))
    @settings(max_examples=100)
    def test_property_device_detection_always_succeeds(self, device_pref: str):
        """
        Property: Device detection should always succeed for valid preferences.
        
        # Feature: adaptive-retrieval-system, Property 17: Hardware Detection Correctness
        """
        device = detect_device(device_pref)
        assert device in [DeviceType.MPS, DeviceType.CUDA, DeviceType.CPU]
    
    @given(st.sampled_from(["auto", "cpu"]))
    @settings(max_examples=100)
    def test_property_hardware_config_complete(self, device_pref: str):
        """
        Property: Hardware config should always be complete and valid.
        
        # Feature: adaptive-retrieval-system, Property 17: Hardware Detection Correctness
        """
        config = get_hardware_config(preferred_device=device_pref)
        
        # Config should have all required fields
        assert config.device is not None
        assert config.device_name is not None
        assert config.max_memory_gb > 0
        assert config.recommended_batch_sizes is not None
        
        # Batch sizes should be positive
        for key, value in config.recommended_batch_sizes.items():
            assert value > 0, f"Batch size for {key} should be positive"
    
    @given(st.floats(min_value=0.1, max_value=128.0))
    @settings(max_examples=100)
    def test_property_memory_limit_respected(self, memory_gb: float):
        """
        Property: Custom memory limits should be respected in config.
        
        # Feature: adaptive-retrieval-system, Property 17: Hardware Detection Correctness
        """
        config = get_hardware_config(preferred_device="cpu", max_memory_gb=memory_gb)
        assert config.max_memory_gb == memory_gb
    
    @given(st.integers(min_value=0, max_value=10000))
    @settings(max_examples=100)
    def test_property_seed_reproducibility(self, seed: int):
        """
        Property: Any valid seed should produce reproducible results.
        
        # Feature: adaptive-retrieval-system, Property 17: Hardware Detection Correctness
        """
        import torch
        
        set_seed(seed)
        val1 = torch.rand(10).tolist()
        
        set_seed(seed)
        val2 = torch.rand(10).tolist()
        
        assert val1 == val2, "Same seed should produce identical results"


class TestDefaultBatchSizes:
    """Tests for default batch size configurations."""
    
    def test_all_device_types_have_defaults(self):
        """All device types should have default batch sizes."""
        for device_type in DeviceType:
            assert device_type in DEFAULT_BATCH_SIZES
    
    def test_batch_sizes_are_reasonable(self):
        """Batch sizes should be reasonable for each device."""
        # MPS should have smaller batches than CUDA
        assert DEFAULT_BATCH_SIZES[DeviceType.MPS]["vision_embedding"] <= \
               DEFAULT_BATCH_SIZES[DeviceType.CUDA]["vision_embedding"]
        
        # CPU should have smallest batches
        assert DEFAULT_BATCH_SIZES[DeviceType.CPU]["vision_embedding"] <= \
               DEFAULT_BATCH_SIZES[DeviceType.MPS]["vision_embedding"]

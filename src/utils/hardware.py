"""
Hardware detection and backend configuration.

This module provides utilities for detecting available hardware (MPS, CUDA, CPU)
and configuring PyTorch backends appropriately.

Requirements: 9.4, 9.5, 9.6
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types."""
    
    MPS = "mps"      # Apple Silicon GPU
    CUDA = "cuda"    # NVIDIA GPU
    CPU = "cpu"      # CPU fallback


@dataclass
class HardwareConfig:
    """Hardware configuration for the system."""
    
    device: DeviceType
    device_name: str
    max_memory_gb: float
    supports_mixed_precision: bool
    recommended_batch_sizes: dict[str, int]
    
    def __str__(self) -> str:
        return f"HardwareConfig(device={self.device.value}, name={self.device_name})"


# Default batch sizes per device type
DEFAULT_BATCH_SIZES = {
    DeviceType.MPS: {
        "text_embedding": 16,
        "vision_embedding": 2,
        "router": 32,
    },
    DeviceType.CUDA: {
        "text_embedding": 64,
        "vision_embedding": 8,
        "router": 64,
    },
    DeviceType.CPU: {
        "text_embedding": 8,
        "vision_embedding": 1,
        "router": 16,
    },
}


def detect_device(preferred: Optional[str] = None) -> DeviceType:
    """
    Detect the best available device for computation.
    
    Args:
        preferred: Preferred device type ("mps", "cuda", "cpu", or "auto").
                  If "auto" or None, automatically detects best available.
    
    Returns:
        DeviceType enum indicating the selected device.
    
    Raises:
        ValueError: If preferred device is not available.
    """
    import torch
    
    # Handle explicit preference
    if preferred and preferred != "auto":
        preferred_lower = preferred.lower()
        
        if preferred_lower == "mps":
            if torch.backends.mps.is_available():
                return DeviceType.MPS
            raise ValueError("MPS requested but not available")
        
        elif preferred_lower == "cuda":
            if torch.cuda.is_available():
                return DeviceType.CUDA
            raise ValueError("CUDA requested but not available")
        
        elif preferred_lower == "cpu":
            return DeviceType.CPU
        
        else:
            raise ValueError(f"Unknown device type: {preferred}")
    
    # Auto-detect best available
    if torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) detected")
        return DeviceType.MPS
    
    if torch.cuda.is_available():
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        return DeviceType.CUDA
    
    logger.info("No GPU detected, using CPU")
    return DeviceType.CPU


def get_device_string(device_type: DeviceType) -> str:
    """
    Get the PyTorch device string for a device type.
    
    Args:
        device_type: The device type enum.
    
    Returns:
        String suitable for torch.device().
    """
    return device_type.value


def get_hardware_config(
    preferred_device: Optional[str] = None,
    max_memory_gb: Optional[float] = None,
) -> HardwareConfig:
    """
    Get complete hardware configuration for the system.
    
    Args:
        preferred_device: Preferred device ("mps", "cuda", "cpu", "auto").
        max_memory_gb: Maximum memory to use (None = auto-detect).
    
    Returns:
        HardwareConfig with all settings.
    """
    import torch
    
    device_type = detect_device(preferred_device)
    
    # Get device name
    if device_type == DeviceType.CUDA:
        device_name = torch.cuda.get_device_name(0)
        # Auto-detect memory if not specified
        if max_memory_gb is None:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory_gb = total_memory / (1024 ** 3) * 0.9  # Use 90% of available
    elif device_type == DeviceType.MPS:
        device_name = "Apple Silicon (MPS)"
        # MPS doesn't expose memory info easily, use conservative default
        if max_memory_gb is None:
            max_memory_gb = 8.0  # Conservative for M1 Pro
    else:
        device_name = "CPU"
        if max_memory_gb is None:
            max_memory_gb = 4.0  # Conservative CPU default
    
    # Determine mixed precision support
    supports_mixed_precision = device_type in (DeviceType.CUDA, DeviceType.MPS)
    
    # Get recommended batch sizes
    batch_sizes = DEFAULT_BATCH_SIZES.get(device_type, DEFAULT_BATCH_SIZES[DeviceType.CPU])
    
    config = HardwareConfig(
        device=device_type,
        device_name=device_name,
        max_memory_gb=max_memory_gb,
        supports_mixed_precision=supports_mixed_precision,
        recommended_batch_sizes=batch_sizes,
    )
    
    logger.info(f"Hardware config: {config}")
    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all frameworks.
    
    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For MPS, manual_seed covers it
    
    # Set deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_optimal_workers() -> int:
    """
    Get optimal number of data loader workers for current system.
    
    Returns:
        Number of workers to use.
    """
    cpu_count = os.cpu_count() or 4
    # Use half of available CPUs, minimum 2, maximum 8
    return min(max(cpu_count // 2, 2), 8)


def check_memory_available(required_gb: float, device_type: Optional[DeviceType] = None) -> bool:
    """
    Check if sufficient memory is available for an operation.
    
    Args:
        required_gb: Required memory in GB.
        device_type: Device to check (None = current device).
    
    Returns:
        True if sufficient memory is available.
    """
    import torch
    
    if device_type is None:
        device_type = detect_device()
    
    if device_type == DeviceType.CUDA:
        free_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory -= torch.cuda.memory_allocated(0)
        free_gb = free_memory / (1024 ** 3)
        return free_gb >= required_gb
    
    # For MPS and CPU, we can't easily check available memory
    # Return True and let the operation fail if needed
    return True


def clear_memory_cache(device_type: Optional[DeviceType] = None) -> None:
    """
    Clear memory cache for the specified device.
    
    Args:
        device_type: Device to clear (None = all available).
    """
    import torch
    import gc
    
    gc.collect()
    
    if device_type is None or device_type == DeviceType.CUDA:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    if device_type is None or device_type == DeviceType.MPS:
        if torch.backends.mps.is_available():
            # MPS doesn't have explicit cache clearing, but gc helps
            pass
    
    logger.debug("Memory cache cleared")

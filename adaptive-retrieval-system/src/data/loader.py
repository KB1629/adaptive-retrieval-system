"""
Base data loader interface and configuration.

Provides a protocol for dataset loaders and common utilities
for caching, retry logic, and normalization.

Requirements: 7.4, 7.5, 7.7
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

from src.models import BenchmarkDataset, Page, Document

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """
    Configuration for data loading.
    
    Attributes:
        cache_dir: Directory for caching downloaded data
        use_cache: Whether to use cached data if available
        max_retries: Maximum retry attempts for downloads
        retry_delay: Initial delay between retries (seconds)
        timeout: Download timeout in seconds
    """
    cache_dir: str = "~/.cache/huggingface/datasets"
    use_cache: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 300
    
    def __post_init__(self):
        """Expand cache directory path."""
        self.cache_dir = os.path.expanduser(self.cache_dir)
    
    def get_cache_path(self, dataset_name: str) -> Path:
        """Get cache path for a specific dataset."""
        return Path(self.cache_dir) / dataset_name


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for dataset loaders.
    
    All dataset loaders must implement this interface to ensure
    consistent behavior across different data sources.
    """
    
    @property
    def name(self) -> str:
        """Return dataset name."""
        ...
    
    @property
    def description(self) -> str:
        """Return dataset description."""
        ...
    
    def load(self, config: Optional[DataLoaderConfig] = None) -> BenchmarkDataset:
        """
        Load the dataset.
        
        Args:
            config: Optional loader configuration
            
        Returns:
            BenchmarkDataset with normalized data
        """
        ...
    
    def is_cached(self, config: Optional[DataLoaderConfig] = None) -> bool:
        """
        Check if dataset is already cached.
        
        Args:
            config: Optional loader configuration
            
        Returns:
            True if cached data exists
        """
        ...


class BaseDataLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    Provides common functionality for caching, retry logic,
    and error handling.
    """
    
    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """Initialize loader with configuration."""
        self.config = config or DataLoaderConfig()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return dataset name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return dataset description."""
        pass
    
    @abstractmethod
    def _load_raw(self) -> dict:
        """
        Load raw data from source.
        
        Returns:
            Raw dataset dictionary
        """
        pass
    
    @abstractmethod
    def _normalize(self, raw_data: dict) -> BenchmarkDataset:
        """
        Normalize raw data to BenchmarkDataset format.
        
        Args:
            raw_data: Raw dataset dictionary
            
        Returns:
            Normalized BenchmarkDataset
        """
        pass
    
    def load(self, config: Optional[DataLoaderConfig] = None) -> BenchmarkDataset:
        """
        Load the dataset with retry logic.
        
        Args:
            config: Optional loader configuration (overrides instance config)
            
        Returns:
            BenchmarkDataset with normalized data
        """
        cfg = config or self.config
        
        last_error = None
        for attempt in range(cfg.max_retries):
            try:
                logger.info(f"Loading {self.name} (attempt {attempt + 1}/{cfg.max_retries})")
                raw_data = self._load_raw()
                dataset = self._normalize(raw_data)
                logger.info(f"Successfully loaded {self.name}: {len(dataset.pages)} pages")
                return dataset
            except Exception as e:
                last_error = e
                delay = cfg.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Load attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        
        raise RuntimeError(f"Failed to load {self.name} after {cfg.max_retries} attempts: {last_error}")
    
    def is_cached(self, config: Optional[DataLoaderConfig] = None) -> bool:
        """
        Check if dataset is already cached.
        
        Args:
            config: Optional loader configuration
            
        Returns:
            True if cached data exists
        """
        cfg = config or self.config
        cache_path = cfg.get_cache_path(self.name)
        return cache_path.exists()


def with_retry(func, max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        func: Function to wrap
        max_retries: Maximum retry attempts
        delay: Initial delay between retries
        
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                wait = delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")
    return wrapper

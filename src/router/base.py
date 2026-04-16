"""
Base router interface and configuration.

Defines the protocol for all router implementations and
common configuration options.

Requirements: 1.5, 1.7
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from src.models import Page, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """
    Configuration for router behavior.
    
    Attributes:
        text_threshold: Threshold for text density (above = text-heavy)
        image_threshold: Threshold for image ratio (above = visual-critical)
        min_confidence: Minimum confidence for classification
        fallback_modality: Default when classification fails
        batch_size: Batch size for parallel processing
    """
    text_threshold: float = 0.6
    image_threshold: float = 0.3
    min_confidence: float = 0.5
    fallback_modality: Literal["text-heavy", "visual-critical"] = "visual-critical"
    batch_size: int = 32
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError(f"text_threshold must be in [0, 1], got {self.text_threshold}")
        if not 0.0 <= self.image_threshold <= 1.0:
            raise ValueError(f"image_threshold must be in [0, 1], got {self.image_threshold}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")


@runtime_checkable
class RouterInterface(Protocol):
    """
    Protocol for page classification routers.
    
    All router implementations must provide these methods.
    """
    
    def classify(self, page: Page) -> ClassificationResult:
        """
        Classify a single page.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult with modality and confidence
        """
        ...
    
    def classify_batch(self, pages: list[Page]) -> list[ClassificationResult]:
        """
        Classify multiple pages.
        
        Args:
            pages: List of pages to classify
            
        Returns:
            List of ClassificationResults in same order
        """
        ...


class BaseRouter(ABC):
    """
    Abstract base class for routers.
    
    Provides common functionality for error handling and
    fallback behavior.
    """
    
    def __init__(self, config: RouterConfig = None):
        """Initialize router with configuration."""
        self.config = config or RouterConfig()
    
    @abstractmethod
    def _classify_impl(self, page: Page) -> ClassificationResult:
        """
        Internal classification implementation.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult
        """
        pass
    
    def classify(self, page: Page) -> ClassificationResult:
        """
        Classify a single page with error handling.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult with modality and confidence
        """
        try:
            result = self._classify_impl(page)
            
            # Apply minimum confidence threshold
            if result.confidence < self.config.min_confidence:
                logger.warning(
                    f"Low confidence ({result.confidence:.2f}) for page "
                    f"{page.page_number} of {page.source_document}, using fallback"
                )
                return ClassificationResult(
                    modality=self.config.fallback_modality,
                    confidence=result.confidence,
                    features=result.features,
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed for page {page.page_number}: {e}")
            # Fallback to visual-critical (safer for accuracy)
            return ClassificationResult(
                modality=self.config.fallback_modality,
                confidence=0.0,
                features={"error": 1.0},
            )
    
    def classify_batch(self, pages: list[Page]) -> list[ClassificationResult]:
        """
        Classify multiple pages.
        
        Default implementation processes sequentially.
        Subclasses can override for parallel processing.
        
        Args:
            pages: List of pages to classify
            
        Returns:
            List of ClassificationResults in same order
        """
        return [self.classify(page) for page in pages]

"""
Heuristic-based router for page classification.

Uses visual features (text density, image ratio, edge density)
to classify pages without requiring ML models.

Requirements: 1.1, 1.3, 1.4, 1.6
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.models import Page, ClassificationResult
from .base import BaseRouter, RouterConfig

logger = logging.getLogger(__name__)


@dataclass
class VisualFeatures:
    """
    Visual features extracted from a page image.
    
    Attributes:
        text_density: Estimated text coverage (0-1)
        image_ratio: Ratio of non-text visual content (0-1)
        edge_density: Edge pixel density (indicates diagrams)
        color_variance: Color variation (high = images/diagrams)
        white_ratio: Ratio of white/background pixels
    """
    text_density: float
    image_ratio: float
    edge_density: float
    color_variance: float
    white_ratio: float
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "text_density": self.text_density,
            "image_ratio": self.image_ratio,
            "edge_density": self.edge_density,
            "color_variance": self.color_variance,
            "white_ratio": self.white_ratio,
        }


class HeuristicRouter(BaseRouter):
    """
    Heuristic router using visual feature analysis.
    
    Classifies pages based on:
    - Text density (high = text-heavy)
    - Image/diagram ratio (high = visual-critical)
    - Edge density (high = diagrams/schematics)
    
    This approach is fast (<10ms per page) and doesn't require
    any ML model loading.
    """
    
    def __init__(self, config: RouterConfig = None):
        """Initialize heuristic router."""
        super().__init__(config)
        
        # Feature weights for classification
        # Adjusted to handle dark backgrounds better
        self._weights = {
            "text_density": 0.5,      # Increased: text density is key signal
            "image_ratio": -0.4,      # Increased: penalize image content more
            "edge_density": -0.15,    # Reduced: edges can be text too
            "color_variance": -0.05,  # Reduced: color variance less important
        }
    
    def _classify_impl(self, page: Page) -> ClassificationResult:
        """
        Classify page using heuristic features.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult
        """
        # Extract visual features
        features = self._extract_features(page.image)
        
        # Compute classification score
        score = self._compute_score(features)
        
        # Determine modality based on score
        if score >= self.config.text_threshold:
            modality = "text-heavy"
            confidence = min(1.0, score)
        else:
            modality = "visual-critical"
            confidence = min(1.0, 1.0 - score)
        
        return ClassificationResult(
            modality=modality,
            confidence=confidence,
            features=features.to_dict(),
        )
    
    def _extract_features(self, image: np.ndarray) -> VisualFeatures:
        """
        Extract visual features from page image.
        
        Args:
            image: RGB image array (HxWx3)
            
        Returns:
            VisualFeatures
        """
        # Convert to grayscale for analysis
        gray = np.mean(image, axis=2).astype(np.uint8)
        
        # Text density: estimate based on dark pixel ratio in typical text regions
        text_density = self._estimate_text_density(gray)
        
        # Image ratio: estimate non-text visual content
        image_ratio = self._estimate_image_ratio(image, gray)
        
        # Edge density: detect lines and shapes (diagrams)
        edge_density = self._compute_edge_density(gray)
        
        # Color variance: high variance suggests images/diagrams
        color_variance = self._compute_color_variance(image)
        
        # White ratio: background coverage
        white_ratio = self._compute_white_ratio(gray)
        
        return VisualFeatures(
            text_density=text_density,
            image_ratio=image_ratio,
            edge_density=edge_density,
            color_variance=color_variance,
            white_ratio=white_ratio,
        )
    
    def _estimate_text_density(self, gray: np.ndarray) -> float:
        """
        Estimate text density from grayscale image.
        
        Text typically appears as dark pixels on light background
        with specific spatial patterns.
        """
        # Threshold for dark pixels (text)
        dark_threshold = 128
        dark_pixels = np.sum(gray < dark_threshold)
        total_pixels = gray.size
        
        # Raw dark ratio
        dark_ratio = dark_pixels / total_pixels
        
        # Text typically has moderate dark ratio (0.1-0.4)
        # Very high dark ratio suggests images, not text
        if dark_ratio > 0.5:
            # Likely an image or dark background
            return max(0.0, 1.0 - dark_ratio)
        
        # Normalize to 0-1 range
        return min(1.0, dark_ratio * 2.5)
    
    def _estimate_image_ratio(self, image: np.ndarray, gray: np.ndarray) -> float:
        """
        Estimate ratio of image/diagram content.
        
        Images and diagrams typically have:
        - Color variation
        - Large contiguous regions
        - Non-text patterns
        """
        # Check for color content (non-grayscale)
        color_diff = np.std(image, axis=2)
        color_pixels = np.sum(color_diff > 10)
        color_ratio = color_pixels / gray.size
        
        # Check for large uniform regions (typical in diagrams)
        # Using simple block analysis
        block_size = 32
        h, w = gray.shape
        uniform_blocks = 0
        total_blocks = 0
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                if np.std(block) < 20:  # Uniform block
                    uniform_blocks += 1
                total_blocks += 1
        
        uniform_ratio = uniform_blocks / max(1, total_blocks)
        
        # Combine signals
        image_ratio = 0.6 * color_ratio + 0.4 * (1.0 - uniform_ratio)
        return min(1.0, max(0.0, image_ratio))
    
    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """
        Compute edge density using simple gradient.
        
        High edge density suggests diagrams, schematics, or charts.
        """
        # Simple Sobel-like gradient
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0))
        
        # Edge threshold
        edge_threshold = 30
        edge_pixels_x = np.sum(gx > edge_threshold)
        edge_pixels_y = np.sum(gy > edge_threshold)
        
        total_pixels = gray.size
        edge_density = (edge_pixels_x + edge_pixels_y) / (2 * total_pixels)
        
        return min(1.0, edge_density * 5)  # Scale up
    
    def _compute_color_variance(self, image: np.ndarray) -> float:
        """
        Compute color variance across the image.
        
        High variance suggests colorful images or diagrams.
        """
        # Variance across color channels
        var_r = np.var(image[:, :, 0])
        var_g = np.var(image[:, :, 1])
        var_b = np.var(image[:, :, 2])
        
        avg_var = (var_r + var_g + var_b) / 3
        
        # Normalize (typical variance range 0-5000)
        return min(1.0, avg_var / 5000)
    
    def _compute_white_ratio(self, gray: np.ndarray) -> float:
        """
        Compute ratio of white/background pixels.
        
        Text documents typically have high white ratio.
        """
        white_threshold = 240
        white_pixels = np.sum(gray > white_threshold)
        return white_pixels / gray.size
    
    def _compute_score(self, features: VisualFeatures) -> float:
        """
        Compute classification score from features.
        
        Higher score = more likely text-heavy.
        """
        score = 0.5  # Base score
        
        # Apply weighted features
        score += self._weights["text_density"] * features.text_density
        score += self._weights["image_ratio"] * features.image_ratio
        score += self._weights["edge_density"] * features.edge_density
        score += self._weights["color_variance"] * features.color_variance
        
        # White ratio bonus for text - but don't penalize dark backgrounds too much
        # Many scanned documents have dark/colored backgrounds
        if features.white_ratio > 0.7:
            score += 0.1 * features.white_ratio
        
        return max(0.0, min(1.0, score))

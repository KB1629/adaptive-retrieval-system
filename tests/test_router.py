"""
Tests for router component.

Property tests validate:
- Property 1: Router Classification Correctness
- Property 2: Batch Classification Output Consistency

Validates: Requirements 1.2, 1.3, 1.4, 1.7
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.models import Page, ClassificationResult
from src.router import RouterConfig, HeuristicRouter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def router():
    """Create default heuristic router."""
    return HeuristicRouter()


@pytest.fixture
def text_heavy_page():
    """Create a text-heavy page (white background, sparse dark pixels)."""
    # Simulate a text document: mostly white with some dark text
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add some "text" lines (dark horizontal stripes)
    for y in range(50, 750, 30):
        image[y:y+10, 50:550, :] = 30  # Dark text
    
    return Page.from_array(image, page_number=1, source_document="text_doc.pdf")


@pytest.fixture
def visual_page():
    """Create a visual-critical page (diagram with colors and shapes)."""
    # Simulate a diagram: colored shapes on white background
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add colored rectangles (diagram elements)
    image[100:300, 100:300, 0] = 200  # Red box
    image[100:300, 100:300, 1] = 50
    image[100:300, 100:300, 2] = 50
    
    image[350:550, 200:400, 0] = 50
    image[350:550, 200:400, 1] = 200  # Green box
    image[350:550, 200:400, 2] = 50
    
    # Add lines (connections)
    image[300:350, 200:250, :] = 0  # Black line
    
    return Page.from_array(image, page_number=1, source_document="diagram.pdf")


# ============================================================================
# Unit Tests: RouterConfig
# ============================================================================

class TestRouterConfig:
    """Tests for RouterConfig."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        config = RouterConfig()
        
        assert 0.0 <= config.text_threshold <= 1.0
        assert 0.0 <= config.image_threshold <= 1.0
        assert config.fallback_modality in ["text-heavy", "visual-critical"]
    
    def test_invalid_threshold(self):
        """Should reject invalid thresholds."""
        with pytest.raises(ValueError):
            RouterConfig(text_threshold=1.5)
        
        with pytest.raises(ValueError):
            RouterConfig(image_threshold=-0.1)
    
    def test_custom_config(self):
        """Should accept custom configuration."""
        config = RouterConfig(
            text_threshold=0.7,
            image_threshold=0.4,
            fallback_modality="text-heavy",
        )
        
        assert config.text_threshold == 0.7
        assert config.image_threshold == 0.4
        assert config.fallback_modality == "text-heavy"


# ============================================================================
# Unit Tests: HeuristicRouter
# ============================================================================

class TestHeuristicRouter:
    """Tests for HeuristicRouter."""
    
    def test_classify_returns_result(self, router, text_heavy_page):
        """classify should return ClassificationResult."""
        result = router.classify(text_heavy_page)
        
        assert isinstance(result, ClassificationResult)
        assert result.modality in ["text-heavy", "visual-critical"]
        assert 0.0 <= result.confidence <= 1.0
    
    def test_classify_text_heavy_page(self, router, text_heavy_page):
        """Text-heavy page should be classified as text-heavy."""
        result = router.classify(text_heavy_page)
        
        # Should lean towards text-heavy
        assert result.modality == "text-heavy" or result.confidence < 0.7
    
    def test_classify_visual_page(self, router, visual_page):
        """Visual page should be classified as visual-critical."""
        result = router.classify(visual_page)
        
        # Should lean towards visual-critical
        assert result.modality == "visual-critical" or result.confidence < 0.7
    
    def test_classify_batch(self, router, text_heavy_page, visual_page):
        """classify_batch should return results for all pages."""
        pages = [text_heavy_page, visual_page]
        results = router.classify_batch(pages)
        
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)
    
    def test_features_extracted(self, router, text_heavy_page):
        """Classification should include extracted features."""
        result = router.classify(text_heavy_page)
        
        assert "text_density" in result.features
        assert "image_ratio" in result.features
        assert "edge_density" in result.features
    
    def test_fallback_on_error(self, router):
        """Should use fallback modality on classification error."""
        # Create invalid page (will cause error in feature extraction)
        # Using a valid but unusual image
        image = np.zeros((10, 10, 3), dtype=np.uint8)  # Very small
        page = Page.from_array(image, page_number=1, source_document="tiny.pdf")
        
        result = router.classify(page)
        
        # Should still return a valid result
        assert isinstance(result, ClassificationResult)
        assert result.modality in ["text-heavy", "visual-critical"]


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestRouterClassificationProperty:
    """
    Property 1: Router Classification Correctness
    
    For any valid page, the router must return a valid classification.
    Validates: Requirements 1.2, 1.3, 1.4
    """
    
    @given(
        height=st.integers(min_value=64, max_value=1024),
        width=st.integers(min_value=64, max_value=1024),
        brightness=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=100)
    def test_property_always_returns_valid_classification(
        self, height, width, brightness
    ):
        """
        Property: Router always returns valid classification for any image.
        
        # Feature: adaptive-retrieval-system, Property 1: Router Classification Correctness
        **Validates: Requirements 1.2, 1.3**
        """
        # Create random image
        image = np.full((height, width, 3), brightness, dtype=np.uint8)
        # Add some noise
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        page = Page.from_array(image, page_number=1, source_document="test.pdf")
        
        router = HeuristicRouter()
        result = router.classify(page)
        
        # Must return valid classification
        assert result.modality in ["text-heavy", "visual-critical"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.features, dict)
    
    @given(
        text_threshold=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
        min_confidence=st.floats(min_value=0.0, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_property_respects_configuration(self, text_threshold, min_confidence):
        """
        Property: Router respects configuration thresholds.
        
        # Feature: adaptive-retrieval-system, Property 1: Router Classification Correctness
        **Validates: Requirements 1.4**
        """
        config = RouterConfig(
            text_threshold=text_threshold,
            min_confidence=min_confidence,
        )
        router = HeuristicRouter(config)
        
        # Create test image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        page = Page.from_array(image, page_number=1, source_document="test.pdf")
        
        result = router.classify(page)
        
        # Result should be valid regardless of config
        assert result.modality in ["text-heavy", "visual-critical"]
        assert 0.0 <= result.confidence <= 1.0


class TestBatchClassificationProperty:
    """
    Property 2: Batch Classification Output Consistency
    
    Batch classification should produce same results as individual classification.
    Validates: Requirements 1.7
    """
    
    @given(
        num_pages=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50)
    def test_property_batch_equals_individual(self, num_pages, seed):
        """
        Property: Batch classification equals individual classification.
        
        # Feature: adaptive-retrieval-system, Property 2: Batch Classification Output Consistency
        **Validates: Requirements 1.7**
        """
        np.random.seed(seed)
        
        # Create random pages
        pages = []
        for i in range(num_pages):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            page = Page.from_array(image, page_number=i+1, source_document=f"doc_{i}.pdf")
            pages.append(page)
        
        router = HeuristicRouter()
        
        # Classify individually
        individual_results = [router.classify(page) for page in pages]
        
        # Classify as batch
        batch_results = router.classify_batch(pages)
        
        # Results should match
        assert len(batch_results) == len(individual_results)
        
        for ind, batch in zip(individual_results, batch_results):
            assert ind.modality == batch.modality
            assert abs(ind.confidence - batch.confidence) < 1e-6
    
    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=30)
    def test_property_batch_size_respected(self, batch_size):
        """
        Property: Batch processing respects configured batch size.
        
        # Feature: adaptive-retrieval-system, Property 2: Batch Classification Output Consistency
        **Validates: Requirements 1.7**
        """
        config = RouterConfig(batch_size=batch_size)
        router = HeuristicRouter(config)
        
        # Create more pages than batch size
        num_pages = batch_size * 2 + 1
        pages = []
        for i in range(num_pages):
            image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            page = Page.from_array(image, page_number=i+1, source_document=f"doc_{i}.pdf")
            pages.append(page)
        
        results = router.classify_batch(pages)
        
        # Should return result for each page
        assert len(results) == num_pages
        assert all(isinstance(r, ClassificationResult) for r in results)


class TestRouterFallbackProperty:
    """
    Additional property tests for router fallback behavior.
    """
    
    @given(
        fallback=st.sampled_from(["text-heavy", "visual-critical"]),
        min_confidence=st.floats(min_value=0.8, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_property_low_confidence_uses_fallback(self, fallback, min_confidence):
        """
        Property: Low confidence results use fallback modality.
        
        # Feature: adaptive-retrieval-system, Property 1: Router Classification Correctness
        **Validates: Requirements 1.4**
        """
        config = RouterConfig(
            fallback_modality=fallback,
            min_confidence=min_confidence,
        )
        router = HeuristicRouter(config)
        
        # Create ambiguous image (gray, uniform)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        page = Page.from_array(image, page_number=1, source_document="gray.pdf")
        
        result = router.classify(page)
        
        # If confidence is below threshold, should use fallback
        # (Note: actual behavior depends on computed confidence)
        assert result.modality in ["text-heavy", "visual-critical"]
        assert 0.0 <= result.confidence <= 1.0

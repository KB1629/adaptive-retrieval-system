"""
Tests for data models and serialization.

Property tests validate round-trip serialization of all dataclasses.
Validates: Requirements 4.2, 10.3
"""

import pytest
import numpy as np
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume

from src.models import (
    Page,
    Document,
    EmbeddingResult,
    BenchmarkDataset,
    ClassificationResult,
    SearchResult,
    QueryResult,
    MetricsResult,
    LatencyResult,
    ExperimentConfig,
    ExperimentResult,
)


# Custom strategies for generating test data
@st.composite
def valid_embedding_vectors(draw, min_dim=64, max_dim=1024):
    """Generate valid embedding vectors."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    return np.random.randn(dim).astype(np.float32)


@st.composite
def valid_page_images(draw, min_size=32, max_size=512):
    """Generate valid page images."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


@st.composite
def valid_metrics(draw):
    """Generate valid metrics in [0, 1] range."""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))


class TestPageModel:
    """Tests for Page dataclass."""
    
    def test_page_creation(self, sample_page_image):
        """Page should be created with valid data."""
        page = Page.from_array(sample_page_image, page_number=1, source_document="test.pdf")
        
        assert page.page_number == 1
        assert page.source_document == "test.pdf"
        assert page.width == sample_page_image.shape[1]
        assert page.height == sample_page_image.shape[0]
    
    def test_page_invalid_page_number(self, sample_page_image):
        """Page should reject invalid page numbers."""
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            Page.from_array(sample_page_image, page_number=0, source_document="test.pdf")
    
    def test_page_to_dict(self, sample_page_image):
        """Page should serialize to dict without image."""
        page = Page.from_array(sample_page_image, page_number=1, source_document="test.pdf")
        data = page.to_dict()
        
        assert "page_number" in data
        assert "source_document" in data
        assert "width" in data
        assert "height" in data
        assert "image" not in data  # Image should not be serialized


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""
    
    def test_embedding_result_creation(self, sample_embedding):
        """EmbeddingResult should be created with valid data."""
        result = EmbeddingResult(
            vector=sample_embedding,
            modality="text-heavy",
            processing_time_ms=50.0,
            model_name="nomic-embed-text",
        )
        
        assert result.dimensions == len(sample_embedding)
        assert result.modality == "text-heavy"
    
    def test_embedding_result_round_trip(self, sample_embedding):
        """EmbeddingResult should survive serialization round-trip."""
        original = EmbeddingResult(
            vector=sample_embedding,
            modality="visual-critical",
            processing_time_ms=100.0,
            model_name="colpali",
            extracted_text="Sample text",
        )
        
        data = original.to_dict()
        restored = EmbeddingResult.from_dict(data)
        
        np.testing.assert_array_almost_equal(original.vector, restored.vector)
        assert original.modality == restored.modality
        assert original.processing_time_ms == restored.processing_time_ms
        assert original.model_name == restored.model_name
        assert original.extracted_text == restored.extracted_text


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""
    
    def test_classification_result_creation(self):
        """ClassificationResult should be created with valid data."""
        result = ClassificationResult(
            modality="text-heavy",
            confidence=0.95,
            features={"text_density": 0.8, "image_ratio": 0.1},
        )
        
        assert result.is_text_heavy
        assert not result.is_visual_critical
    
    def test_classification_result_invalid_confidence(self):
        """ClassificationResult should reject invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            ClassificationResult(modality="text-heavy", confidence=1.5)
    
    def test_classification_result_round_trip(self):
        """ClassificationResult should survive serialization round-trip."""
        original = ClassificationResult(
            modality="visual-critical",
            confidence=0.85,
            features={"text_density": 0.3, "image_ratio": 0.6},
        )
        
        data = original.to_dict()
        restored = ClassificationResult.from_dict(data)
        
        assert original.modality == restored.modality
        assert original.confidence == restored.confidence
        assert original.features == restored.features


class TestMetricsResult:
    """Tests for MetricsResult dataclass."""
    
    def test_metrics_result_creation(self):
        """MetricsResult should be created with valid data."""
        metrics = MetricsResult(
            recall_at_1=0.8,
            recall_at_5=0.9,
            recall_at_10=0.95,
            mrr=0.85,
            ndcg=0.88,
        )
        
        assert metrics.recall_at_10 >= metrics.recall_at_5 >= metrics.recall_at_1
    
    def test_metrics_result_invalid_range(self):
        """MetricsResult should reject values outside [0, 1]."""
        with pytest.raises(ValueError):
            MetricsResult(recall_at_1=1.5)
    
    def test_metrics_result_round_trip(self):
        """MetricsResult should survive serialization round-trip."""
        original = MetricsResult(
            recall_at_1=0.75,
            recall_at_5=0.88,
            recall_at_10=0.92,
            mrr=0.82,
            ndcg=0.85,
        )
        
        data = original.to_dict()
        restored = MetricsResult.from_dict(data)
        
        assert original.recall_at_1 == restored.recall_at_1
        assert original.recall_at_5 == restored.recall_at_5
        assert original.recall_at_10 == restored.recall_at_10
        assert original.mrr == restored.mrr
        assert original.ndcg == restored.ndcg


class TestLatencyResult:
    """Tests for LatencyResult dataclass."""
    
    def test_latency_result_from_measurements(self):
        """LatencyResult should compute statistics from measurements."""
        measurements = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = LatencyResult.from_measurements(measurements)
        
        assert result.mean_ms == 30.0
        assert result.median_ms == 30.0
        assert result.min_ms == 10.0
        assert result.max_ms == 50.0
        assert result.num_samples == 5
    
    def test_latency_result_empty_measurements(self):
        """LatencyResult should handle empty measurements."""
        result = LatencyResult.from_measurements([])
        assert result.num_samples == 0
    
    def test_latency_result_round_trip(self):
        """LatencyResult should survive serialization round-trip."""
        original = LatencyResult.from_measurements([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        data = original.to_dict()
        restored = LatencyResult.from_dict(data)
        
        assert original.mean_ms == restored.mean_ms
        assert original.median_ms == restored.median_ms
        assert original.p95_ms == restored.p95_ms
        assert original.num_samples == restored.num_samples


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""
    
    def test_experiment_config_creation(self):
        """ExperimentConfig should be created with valid data."""
        config = ExperimentConfig(
            experiment_id="exp-001",
            router_type="heuristic",
            vision_model="colpali",
            text_model="nomic-embed-text",
        )
        
        assert config.experiment_id == "exp-001"
        assert config.random_seed == 42  # Default
    
    def test_experiment_config_round_trip(self):
        """ExperimentConfig should survive serialization round-trip."""
        original = ExperimentConfig(
            experiment_id="exp-002",
            router_type="ml",
            vision_model="siglip",
            text_model="nomic-embed-text",
            lora_weights_path="/path/to/weights",
            batch_size=8,
            random_seed=123,
        )
        
        data = original.to_dict()
        restored = ExperimentConfig.from_dict(data)
        
        assert original.experiment_id == restored.experiment_id
        assert original.router_type == restored.router_type
        assert original.vision_model == restored.vision_model
        assert original.lora_weights_path == restored.lora_weights_path
        assert original.batch_size == restored.batch_size
        assert original.random_seed == restored.random_seed


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""
    
    def test_experiment_result_creation(self):
        """ExperimentResult should be created with valid data."""
        config = ExperimentConfig(
            experiment_id="exp-003",
            router_type="heuristic",
            vision_model="colpali",
            text_model="nomic-embed-text",
        )
        
        result = ExperimentResult(
            config=config,
            metrics=MetricsResult(recall_at_10=0.9),
            throughput_pages_per_sec=15.0,
            router_accuracy=0.95,
        )
        
        assert result.config.experiment_id == "exp-003"
        assert result.metrics.recall_at_10 == 0.9
    
    def test_experiment_result_round_trip(self):
        """ExperimentResult should survive serialization round-trip."""
        config = ExperimentConfig(
            experiment_id="exp-004",
            router_type="heuristic",
            vision_model="colpali",
            text_model="nomic-embed-text",
        )
        
        original = ExperimentResult(
            config=config,
            metrics=MetricsResult(recall_at_1=0.8, recall_at_5=0.9, recall_at_10=0.95),
            latency=LatencyResult.from_measurements([50, 60, 70, 80, 90]),
            throughput_pages_per_sec=12.5,
            router_accuracy=0.92,
            dataset_name="REAL-MM-RAG_TechReport",
            notes="Test experiment",
        )
        
        data = original.to_dict()
        restored = ExperimentResult.from_dict(data)
        
        assert original.config.experiment_id == restored.config.experiment_id
        assert original.metrics.recall_at_10 == restored.metrics.recall_at_10
        assert original.throughput_pages_per_sec == restored.throughput_pages_per_sec
        assert original.router_accuracy == restored.router_accuracy
        assert original.dataset_name == restored.dataset_name


# Property-Based Tests
# Feature: adaptive-retrieval-system, Property: Data Model Serialization

class TestDataModelSerializationProperty:
    """
    Property tests for data model serialization.
    
    Validates: Requirements 4.2, 10.3
    """
    
    @given(
        modality=st.sampled_from(["text-heavy", "visual-critical"]),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_property_classification_result_round_trip(self, modality, confidence):
        """
        Property: ClassificationResult should survive serialization round-trip.
        
        # Feature: adaptive-retrieval-system, Property: Data Model Serialization
        """
        original = ClassificationResult(
            modality=modality,
            confidence=confidence,
            features={"test": 0.5},
        )
        
        data = original.to_dict()
        restored = ClassificationResult.from_dict(data)
        
        assert original.modality == restored.modality
        assert abs(original.confidence - restored.confidence) < 1e-6
    
    @given(
        recall_at_1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        recall_at_5=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        recall_at_10=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        mrr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ndcg=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_property_metrics_result_round_trip(
        self, recall_at_1, recall_at_5, recall_at_10, mrr, ndcg
    ):
        """
        Property: MetricsResult should survive serialization round-trip.
        
        # Feature: adaptive-retrieval-system, Property: Data Model Serialization
        """
        original = MetricsResult(
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg=ndcg,
        )
        
        data = original.to_dict()
        restored = MetricsResult.from_dict(data)
        
        assert abs(original.recall_at_1 - restored.recall_at_1) < 1e-6
        assert abs(original.recall_at_10 - restored.recall_at_10) < 1e-6
        assert abs(original.mrr - restored.mrr) < 1e-6
    
    @given(
        measurements=st.lists(
            st.floats(min_value=0.1, max_value=10000.0, allow_nan=False),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_property_latency_result_from_measurements(self, measurements):
        """
        Property: LatencyResult statistics should be consistent.
        
        # Feature: adaptive-retrieval-system, Property: Data Model Serialization
        """
        result = LatencyResult.from_measurements(measurements)
        
        # Use epsilon for floating-point comparison (numpy mean can have tiny precision errors)
        eps = 1e-9
        
        # Validate statistical properties with epsilon tolerance
        assert result.min_ms <= result.mean_ms + eps
        assert result.mean_ms <= result.max_ms + eps
        assert result.min_ms <= result.median_ms + eps
        assert result.median_ms <= result.max_ms + eps
        assert result.median_ms <= result.p95_ms + eps
        assert result.p95_ms <= result.max_ms + eps
        assert result.num_samples == len(measurements)
    
    @given(
        experiment_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Pd"))),
        router_type=st.sampled_from(["heuristic", "ml"]),
        batch_size=st.integers(min_value=1, max_value=128),
        random_seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_property_experiment_config_round_trip(
        self, experiment_id, router_type, batch_size, random_seed
    ):
        """
        Property: ExperimentConfig should survive serialization round-trip.
        
        # Feature: adaptive-retrieval-system, Property: Data Model Serialization
        """
        assume(len(experiment_id.strip()) > 0)  # Skip empty strings
        
        original = ExperimentConfig(
            experiment_id=experiment_id,
            router_type=router_type,
            vision_model="colpali",
            text_model="nomic-embed-text",
            batch_size=batch_size,
            random_seed=random_seed,
        )
        
        data = original.to_dict()
        restored = ExperimentConfig.from_dict(data)
        
        assert original.experiment_id == restored.experiment_id
        assert original.router_type == restored.router_type
        assert original.batch_size == restored.batch_size
        assert original.random_seed == restored.random_seed
    
    @given(
        dim=st.integers(min_value=64, max_value=1024),
        modality=st.sampled_from(["text-heavy", "visual-critical"]),
        processing_time=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_property_embedding_result_round_trip(self, dim, modality, processing_time):
        """
        Property: EmbeddingResult should survive serialization round-trip.
        
        # Feature: adaptive-retrieval-system, Property: Data Model Serialization
        """
        vector = np.random.randn(dim).astype(np.float32)
        
        original = EmbeddingResult(
            vector=vector,
            modality=modality,
            processing_time_ms=processing_time,
            model_name="test-model",
        )
        
        data = original.to_dict()
        restored = EmbeddingResult.from_dict(data)
        
        np.testing.assert_array_almost_equal(original.vector, restored.vector, decimal=5)
        assert original.modality == restored.modality
        assert original.dimensions == restored.dimensions

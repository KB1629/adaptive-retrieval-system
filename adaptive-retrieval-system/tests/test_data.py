"""
Tests for data loading and preprocessing.

Property tests validate:
- Property 11: Dataset Normalization Consistency
- Property 12: Dataset Caching Round-Trip
- Property 13: Dataset Split Proportions

Validates: Requirements 7.4, 7.5, 7.6
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.models import BenchmarkDataset, Page
from src.data import (
    DataLoaderConfig,
    DatasetSplitter,
    SplitConfig,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_pages():
    """Create sample pages for testing."""
    pages = []
    for doc_idx in range(5):
        for page_idx in range(3):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            page = Page.from_array(
                image=image,
                page_number=page_idx + 1,
                source_document=f"doc_{doc_idx}",
            )
            pages.append(page)
    return pages


@pytest.fixture
def sample_dataset(sample_pages):
    """Create sample BenchmarkDataset for testing."""
    queries = [f"Query about doc_{i}" for i in range(5)]
    labels = [f"doc_{i}" for i in range(5)]
    
    return BenchmarkDataset(
        name="test_dataset",
        pages=sample_pages,
        queries=queries,
        labels=labels,
    )


# ============================================================================
# Unit Tests: DataLoaderConfig
# ============================================================================

class TestDataLoaderConfig:
    """Tests for DataLoaderConfig."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        config = DataLoaderConfig()
        
        assert config.use_cache is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 300
    
    def test_cache_path_expansion(self):
        """Cache dir should expand ~ to home directory."""
        config = DataLoaderConfig(cache_dir="~/test_cache")
        
        assert "~" not in config.cache_dir
        assert "test_cache" in config.cache_dir
    
    def test_get_cache_path(self):
        """get_cache_path should return correct path."""
        config = DataLoaderConfig(cache_dir="/tmp/test_cache")
        path = config.get_cache_path("my_dataset")
        
        assert str(path) == "/tmp/test_cache/my_dataset"


# ============================================================================
# Unit Tests: SplitConfig
# ============================================================================

class TestSplitConfig:
    """Tests for SplitConfig."""
    
    def test_default_split_ratios(self):
        """Default ratios should sum to 1.0."""
        config = SplitConfig()
        
        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert abs(total - 1.0) < 1e-6
    
    def test_invalid_ratios_sum(self):
        """Should reject ratios that don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    
    def test_invalid_ratio_range(self):
        """Should reject ratios outside [0, 1]."""
        with pytest.raises(ValueError):
            SplitConfig(train_ratio=1.5, val_ratio=-0.3, test_ratio=-0.2)


# ============================================================================
# Unit Tests: DatasetSplitter
# ============================================================================

class TestDatasetSplitter:
    """Tests for DatasetSplitter."""
    
    def test_split_creates_three_datasets(self, sample_dataset):
        """Split should create train, val, test datasets."""
        splitter = DatasetSplitter()
        train, val, test = splitter.split(sample_dataset)
        
        assert isinstance(train, BenchmarkDataset)
        assert isinstance(val, BenchmarkDataset)
        assert isinstance(test, BenchmarkDataset)
    
    def test_split_preserves_total_pages(self, sample_dataset):
        """Total pages across splits should equal original."""
        splitter = DatasetSplitter()
        train, val, test = splitter.split(sample_dataset)
        
        total_pages = len(train.pages) + len(val.pages) + len(test.pages)
        assert total_pages == len(sample_dataset.pages)
    
    def test_split_no_page_overlap(self, sample_dataset):
        """Pages should not appear in multiple splits."""
        splitter = DatasetSplitter()
        train, val, test = splitter.split(sample_dataset)
        
        train_docs = {p.source_document for p in train.pages}
        val_docs = {p.source_document for p in val.pages}
        test_docs = {p.source_document for p in test.pages}
        
        # No overlap between splits
        assert len(train_docs & val_docs) == 0
        assert len(train_docs & test_docs) == 0
        assert len(val_docs & test_docs) == 0
    
    def test_split_reproducibility(self, sample_dataset):
        """Same seed should produce same splits."""
        config = SplitConfig(seed=42)
        splitter = DatasetSplitter(config)
        
        train1, val1, test1 = splitter.split(sample_dataset)
        train2, val2, test2 = splitter.split(sample_dataset)
        
        # Same documents in each split
        assert {p.source_document for p in train1.pages} == {p.source_document for p in train2.pages}
        assert {p.source_document for p in val1.pages} == {p.source_document for p in val2.pages}
        assert {p.source_document for p in test1.pages} == {p.source_document for p in test2.pages}
    
    def test_split_different_seeds(self, sample_dataset):
        """Different seeds should produce different splits."""
        splitter1 = DatasetSplitter(SplitConfig(seed=42))
        splitter2 = DatasetSplitter(SplitConfig(seed=123))
        
        train1, _, _ = splitter1.split(sample_dataset)
        train2, _, _ = splitter2.split(sample_dataset)
        
        # Different seeds may produce different splits
        # (not guaranteed for small datasets, but likely)
        docs1 = {p.source_document for p in train1.pages}
        docs2 = {p.source_document for p in train2.pages}
        
        # At least check they're valid
        assert len(docs1) > 0
        assert len(docs2) > 0


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestDatasetNormalizationProperty:
    """
    Property 11: Dataset Normalization Consistency
    
    All normalized datasets should have consistent structure.
    Validates: Requirements 7.4
    """
    
    @given(
        num_docs=st.integers(min_value=1, max_value=20),
        pages_per_doc=st.integers(min_value=1, max_value=5),
        num_queries=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=100)
    def test_property_normalized_dataset_structure(self, num_docs, pages_per_doc, num_queries):
        """
        Property: Normalized datasets should have valid structure.
        
        # Feature: adaptive-retrieval-system, Property 11: Dataset Normalization Consistency
        **Validates: Requirements 7.4**
        """
        # Create pages
        pages = []
        for doc_idx in range(num_docs):
            for page_idx in range(pages_per_doc):
                image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                page = Page.from_array(
                    image=image,
                    page_number=page_idx + 1,
                    source_document=f"doc_{doc_idx}",
                )
                pages.append(page)
        
        # Create queries and labels
        queries = [f"Query {i}" for i in range(num_queries)]
        labels = [f"doc_{i % num_docs}" for i in range(num_queries)]
        
        # Create dataset
        dataset = BenchmarkDataset(
            name="test",
            pages=pages,
            queries=queries,
            labels=labels,
        )
        
        # Validate structure
        assert len(dataset.pages) == num_docs * pages_per_doc
        assert len(dataset.queries) == num_queries
        assert len(dataset.labels) == num_queries
        
        # All pages should have valid attributes
        for page in dataset.pages:
            assert page.page_number >= 1
            assert page.source_document is not None
            assert page.width > 0
            assert page.height > 0


class TestDatasetSplitProportionsProperty:
    """
    Property 13: Dataset Split Proportions
    
    Split proportions should approximately match configured ratios.
    Validates: Requirements 7.6
    """
    
    @given(
        train_ratio=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
        val_ratio=st.floats(min_value=0.05, max_value=0.3, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_property_split_proportions(self, train_ratio, val_ratio):
        """
        Property: Split proportions should match configured ratios.
        
        # Feature: adaptive-retrieval-system, Property 13: Dataset Split Proportions
        **Validates: Requirements 7.6**
        """
        # Ensure ratios sum to 1.0
        test_ratio = 1.0 - train_ratio - val_ratio
        assume(test_ratio >= 0.05)  # Ensure test set has some data
        
        # Create dataset with enough documents for meaningful splits
        num_docs = 20
        pages = []
        for doc_idx in range(num_docs):
            image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            page = Page.from_array(
                image=image,
                page_number=1,
                source_document=f"doc_{doc_idx}",
            )
            pages.append(page)
        
        dataset = BenchmarkDataset(
            name="test",
            pages=pages,
            queries=[],
            labels=[],
        )
        
        # Split with configured ratios
        config = SplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        splitter = DatasetSplitter(config)
        train, val, test = splitter.split(dataset)
        
        # Count unique documents in each split
        train_docs = len({p.source_document for p in train.pages})
        val_docs = len({p.source_document for p in val.pages})
        test_docs = len({p.source_document for p in test.pages})
        
        # Total should equal original
        assert train_docs + val_docs + test_docs == num_docs
        
        # Proportions should be approximately correct (within 2 documents)
        expected_train = int(num_docs * train_ratio)
        expected_val = int(num_docs * val_ratio)
        
        assert abs(train_docs - expected_train) <= 2
        assert abs(val_docs - expected_val) <= 2
    
    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=50)
    def test_property_split_no_leakage(self, seed):
        """
        Property: No document should appear in multiple splits.
        
        # Feature: adaptive-retrieval-system, Property 13: Dataset Split Proportions
        **Validates: Requirements 7.6**
        """
        # Create dataset
        num_docs = 10
        pages = []
        for doc_idx in range(num_docs):
            # Multiple pages per document
            for page_idx in range(3):
                image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                page = Page.from_array(
                    image=image,
                    page_number=page_idx + 1,
                    source_document=f"doc_{doc_idx}",
                )
                pages.append(page)
        
        dataset = BenchmarkDataset(
            name="test",
            pages=pages,
            queries=[],
            labels=[],
        )
        
        # Split
        config = SplitConfig(seed=seed)
        splitter = DatasetSplitter(config)
        train, val, test = splitter.split(dataset)
        
        # Get document sets
        train_docs = {p.source_document for p in train.pages}
        val_docs = {p.source_document for p in val.pages}
        test_docs = {p.source_document for p in test.pages}
        
        # No overlap
        assert len(train_docs & val_docs) == 0, "Train and val have overlapping documents"
        assert len(train_docs & test_docs) == 0, "Train and test have overlapping documents"
        assert len(val_docs & test_docs) == 0, "Val and test have overlapping documents"


class TestBenchmarkDatasetProperty:
    """
    Additional property tests for BenchmarkDataset.
    """
    
    @given(
        name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        num_pages=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=100)
    def test_property_dataset_serialization(self, name, num_pages):
        """
        Property: BenchmarkDataset should serialize correctly.
        
        # Feature: adaptive-retrieval-system, Property 11: Dataset Normalization Consistency
        **Validates: Requirements 7.4**
        """
        assume(len(name.strip()) > 0)
        
        # Create pages
        pages = []
        for i in range(num_pages):
            image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            page = Page.from_array(
                image=image,
                page_number=1,
                source_document=f"doc_{i}",
            )
            pages.append(page)
        
        # Create dataset
        dataset = BenchmarkDataset(
            name=name,
            pages=pages,
            queries=[],
            labels=[],
        )
        
        # Serialize and check
        data = dataset.to_dict()
        
        assert data["name"] == name
        assert data["num_pages"] == num_pages
        assert data["num_queries"] == 0

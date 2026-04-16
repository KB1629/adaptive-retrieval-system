"""Data models and core types."""

from .data import (
    Page,
    Document,
    EmbeddingResult,
    BenchmarkDataset,
)
from .results import (
    ClassificationResult,
    SearchResult,
    QueryResult,
    MetricsResult,
    LatencyResult,
)
from .config import (
    ExperimentConfig,
    ExperimentResult,
)

__all__ = [
    # Data structures
    "Page",
    "Document",
    "EmbeddingResult",
    "BenchmarkDataset",
    # Results
    "ClassificationResult",
    "SearchResult",
    "QueryResult",
    "MetricsResult",
    "LatencyResult",
    # Config
    "ExperimentConfig",
    "ExperimentResult",
]

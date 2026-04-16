"""
Result data structures for classification, search, and queries.

These dataclasses represent outputs from various pipeline components:
- ClassificationResult: Router output
- SearchResult: Vector database search result
- QueryResult: Final retrieval result
- MetricsResult: Benchmark metrics
- LatencyResult: Latency statistics

Requirements: 4.2, 5.3, 6.1, 6.2
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
import json


@dataclass
class ClassificationResult:
    """
    Result from page classification by the Router.
    
    Attributes:
        modality: Classification result ("text-heavy" or "visual-critical")
        confidence: Confidence score (0.0 to 1.0)
        features: Feature values used for classification
    """
    modality: Literal["text-heavy", "visual-critical"]
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate classification result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    @property
    def is_text_heavy(self) -> bool:
        """Check if classified as text-heavy."""
        return self.modality == "text-heavy"
    
    @property
    def is_visual_critical(self) -> bool:
        """Check if classified as visual-critical."""
        return self.modality == "visual-critical"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "modality": self.modality,
            "confidence": self.confidence,
            "features": self.features,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationResult":
        """Create from dictionary."""
        return cls(
            modality=data["modality"],
            confidence=data["confidence"],
            features=data.get("features", {}),
        )


@dataclass
class SearchResult:
    """
    Single result from vector database search.
    
    Attributes:
        doc_id: Document identifier
        page_number: Page number within document
        score: Relevance/similarity score
        modality: Embedding modality used
        metadata: Additional metadata
    """
    doc_id: str
    page_number: int
    score: float
    modality: str
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "score": self.score,
            "modality": self.modality,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            page_number=data["page_number"],
            score=data["score"],
            modality=data["modality"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueryResult:
    """
    Complete result from a retrieval query.
    
    Attributes:
        query: Original query text
        results: List of search results
        query_latency_ms: Time taken for query
        total_searched: Number of documents searched
    """
    query: str
    results: list[SearchResult] = field(default_factory=list)
    query_latency_ms: float = 0.0
    total_searched: int = 0
    
    @property
    def num_results(self) -> int:
        """Return number of results."""
        return len(self.results)
    
    @property
    def top_result(self) -> Optional[SearchResult]:
        """Return top result if available."""
        return self.results[0] if self.results else None
    
    def get_doc_ids(self) -> list[str]:
        """Get list of document IDs in result order."""
        return [r.doc_id for r in self.results]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "query_latency_ms": self.query_latency_ms,
            "total_searched": self.total_searched,
            "num_results": self.num_results,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            results=[SearchResult.from_dict(r) for r in data["results"]],
            query_latency_ms=data["query_latency_ms"],
            total_searched=data["total_searched"],
        )


@dataclass
class MetricsResult:
    """
    Retrieval evaluation metrics.
    
    Attributes:
        recall_at_1: Recall@1 score
        recall_at_5: Recall@5 score
        recall_at_10: Recall@10 score
        mrr: Mean Reciprocal Rank
        ndcg: Normalized Discounted Cumulative Gain
    """
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    
    def __post_init__(self):
        """Validate metrics are in valid range."""
        for name, value in [
            ("recall_at_1", self.recall_at_1),
            ("recall_at_5", self.recall_at_5),
            ("recall_at_10", self.recall_at_10),
            ("mrr", self.mrr),
            ("ndcg", self.ndcg),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MetricsResult":
        """Create from dictionary."""
        return cls(
            recall_at_1=data.get("recall_at_1", 0.0),
            recall_at_5=data.get("recall_at_5", 0.0),
            recall_at_10=data.get("recall_at_10", 0.0),
            mrr=data.get("mrr", 0.0),
            ndcg=data.get("ndcg", 0.0),
        )
    
    def to_latex_row(self, name: str) -> str:
        """Generate LaTeX table row."""
        return (
            f"{name} & {self.recall_at_1:.3f} & {self.recall_at_5:.3f} & "
            f"{self.recall_at_10:.3f} & {self.mrr:.3f} & {self.ndcg:.3f} \\\\"
        )


@dataclass
class LatencyResult:
    """
    Latency measurement statistics.
    
    Attributes:
        mean_ms: Mean latency in milliseconds
        median_ms: Median latency
        p95_ms: 95th percentile latency
        std_ms: Standard deviation
        min_ms: Minimum latency
        max_ms: Maximum latency
        num_samples: Number of measurements
    """
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    num_samples: int = 0
    
    def __post_init__(self):
        """Validate latency statistics."""
        if self.num_samples > 0:
            # Use small epsilon for floating-point comparison
            eps = 1e-9
            # Validate ordering: min <= median <= max, min <= mean <= max
            if not (self.min_ms <= self.mean_ms + eps and self.mean_ms <= self.max_ms + eps):
                raise ValueError(
                    f"Invalid: min ({self.min_ms}) <= mean ({self.mean_ms}) <= max ({self.max_ms})"
                )
            if not (self.min_ms <= self.median_ms + eps and self.median_ms <= self.max_ms + eps):
                raise ValueError(
                    f"Invalid: min ({self.min_ms}) <= median ({self.median_ms}) <= max ({self.max_ms})"
                )
            if not (self.median_ms <= self.p95_ms + eps and self.p95_ms <= self.max_ms + eps):
                raise ValueError(
                    f"Invalid: median ({self.median_ms}) <= p95 ({self.p95_ms}) <= max ({self.max_ms})"
                )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "num_samples": self.num_samples,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LatencyResult":
        """Create from dictionary."""
        return cls(
            mean_ms=data.get("mean_ms", 0.0),
            median_ms=data.get("median_ms", 0.0),
            p95_ms=data.get("p95_ms", 0.0),
            std_ms=data.get("std_ms", 0.0),
            min_ms=data.get("min_ms", 0.0),
            max_ms=data.get("max_ms", 0.0),
            num_samples=data.get("num_samples", 0),
        )
    
    @classmethod
    def from_measurements(cls, measurements: list[float]) -> "LatencyResult":
        """Create from list of latency measurements."""
        import numpy as np
        
        if not measurements:
            return cls()
        
        arr = np.array(measurements)
        return cls(
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            num_samples=len(measurements),
        )

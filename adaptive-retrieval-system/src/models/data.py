"""
Core data structures for the Adaptive Retrieval System.

These dataclasses represent the fundamental data types used throughout the pipeline:
- Page: A single document page
- Document: A complete document with multiple pages
- EmbeddingResult: Result from embedding generation

Requirements: 4.2, 5.3, 10.1
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
import json
import numpy as np


@dataclass
class Page:
    """
    Represents a single document page.
    
    Attributes:
        image: Page rendered as numpy array (RGB, HxWx3)
        page_number: 1-indexed page number within document
        source_document: Identifier of the parent document
        width: Image width in pixels
        height: Image height in pixels
    """
    image: np.ndarray
    page_number: int
    source_document: str
    width: int
    height: int
    
    def __post_init__(self):
        """Validate page data after initialization."""
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Image must be HxWx3 RGB, got shape {self.image.shape}")
    
    @classmethod
    def from_array(cls, image: np.ndarray, page_number: int, source_document: str) -> "Page":
        """Create Page from numpy array, inferring dimensions."""
        height, width = image.shape[:2]
        return cls(
            image=image,
            page_number=page_number,
            source_document=source_document,
            width=width,
            height=height,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without image data for serialization)."""
        return {
            "page_number": self.page_number,
            "source_document": self.source_document,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class EmbeddingResult:
    """
    Result from embedding generation.
    
    Attributes:
        vector: Embedding vector as numpy array
        modality: Which path generated this embedding
        processing_time_ms: Time taken to generate embedding
        model_name: Name of the model used
        extracted_text: Text content (only for text path)
    """
    vector: np.ndarray
    modality: Literal["text-heavy", "visual-critical"]
    processing_time_ms: float
    model_name: str
    extracted_text: Optional[str] = None
    
    def __post_init__(self):
        """Validate embedding result."""
        if self.vector.ndim != 1:
            raise ValueError(f"Vector must be 1D, got shape {self.vector.shape}")
        if self.processing_time_ms < 0:
            raise ValueError(f"processing_time_ms must be >= 0, got {self.processing_time_ms}")
    
    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return len(self.vector)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vector": self.vector.tolist(),
            "modality": self.modality,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "extracted_text": self.extracted_text,
            "dimensions": self.dimensions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingResult":
        """Create from dictionary."""
        return cls(
            vector=np.array(data["vector"], dtype=np.float32),
            modality=data["modality"],
            processing_time_ms=data["processing_time_ms"],
            model_name=data["model_name"],
            extracted_text=data.get("extracted_text"),
        )


@dataclass
class Document:
    """
    Represents a complete document with multiple pages.
    
    Attributes:
        doc_id: Unique document identifier
        source_path: Original file path
        pages: List of Page objects
        total_pages: Total number of pages
        processed_at: When the document was processed
    """
    doc_id: str
    source_path: str
    pages: list[Page] = field(default_factory=list)
    total_pages: int = 0
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Update total_pages if not set."""
        if self.total_pages == 0 and self.pages:
            self.total_pages = len(self.pages)
    
    def add_page(self, page: Page) -> None:
        """Add a page to the document."""
        self.pages.append(page)
        self.total_pages = len(self.pages)
    
    def get_page(self, page_number: int) -> Optional[Page]:
        """Get page by number (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "total_pages": self.total_pages,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "pages": [p.to_dict() for p in self.pages],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create from dictionary (without page images)."""
        processed_at = None
        if data.get("processed_at"):
            processed_at = datetime.fromisoformat(data["processed_at"])
        
        return cls(
            doc_id=data["doc_id"],
            source_path=data["source_path"],
            total_pages=data["total_pages"],
            processed_at=processed_at,
            pages=[],  # Pages need images, can't deserialize fully
        )


@dataclass
class BenchmarkDataset:
    """
    Dataset for benchmarking retrieval performance.
    
    Supports two modes:
    1. Page-based: Direct list of pages (simpler, used by loaders)
    2. Document-based: List of Document objects (structured)
    
    Attributes:
        name: Dataset name
        pages: List of pages (flat structure)
        documents: List of documents (hierarchical structure)
        queries: List of query strings
        labels: List of relevance labels (parallel to queries)
        ground_truth: Mapping from query to list of relevant doc IDs
    """
    name: str
    pages: list[Page] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    ground_truth: dict[str, list[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate dataset."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
    
    @property
    def num_documents(self) -> int:
        """Return number of unique documents."""
        if self.documents:
            return len(self.documents)
        # Count unique source documents from pages
        return len({p.source_document for p in self.pages})
    
    @property
    def num_pages(self) -> int:
        """Return total number of pages."""
        if self.pages:
            return len(self.pages)
        return sum(doc.total_pages for doc in self.documents)
    
    @property
    def num_queries(self) -> int:
        """Return number of queries."""
        return len(self.queries)
    
    @property
    def total_pages(self) -> int:
        """Return total pages (alias for num_pages)."""
        return self.num_pages
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "num_documents": self.num_documents,
            "num_pages": self.num_pages,
            "num_queries": self.num_queries,
            "queries": self.queries,
            "labels": self.labels,
            "ground_truth": self.ground_truth,
        }
    
    def is_valid(self) -> bool:
        """Check if dataset is valid for benchmarking."""
        has_data = (len(self.pages) > 0 or len(self.documents) > 0)
        has_queries = len(self.queries) > 0
        has_labels = len(self.labels) > 0 or len(self.ground_truth) > 0
        return has_data and has_queries and has_labels

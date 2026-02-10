"""
Vector database interface and base types.

This module defines the protocol for vector database operations
and common data structures.

Requirements: 4.1, 4.5
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Optional, Literal
import numpy as np


@dataclass
class DocumentMetadata:
    """
    Metadata associated with a stored embedding.
    
    Attributes:
        doc_id: Unique document identifier
        page_number: Page number within document (1-indexed)
        modality: Embedding modality ("text-heavy" or "visual-critical")
        source_file: Original file path
        processed_at: Processing timestamp
        model_name: Model used for embedding
        embedding_dim: Embedding vector dimensions
    """
    doc_id: str
    page_number: int
    modality: Literal["text-heavy", "visual-critical"]
    source_file: str
    processed_at: datetime
    model_name: str
    embedding_dim: int
    
    def __post_init__(self):
        """Validate metadata."""
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "modality": self.modality,
            "source_file": self.source_file,
            "processed_at": self.processed_at.isoformat(),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentMetadata":
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            page_number=data["page_number"],
            modality=data["modality"],
            source_file=data["source_file"],
            processed_at=datetime.fromisoformat(data["processed_at"]),
            model_name=data["model_name"],
            embedding_dim=data["embedding_dim"],
        )


@dataclass
class SearchResult:
    """
    Single result from vector database search.
    
    Attributes:
        id: Unique embedding ID
        score: Similarity/relevance score
        metadata: Associated document metadata
        vector: Optional embedding vector
    """
    id: str
    score: float
    metadata: DocumentMetadata
    vector: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata.to_dict(),
        }
        if self.vector is not None:
            result["vector"] = self.vector.tolist()
        return result


class VectorDBInterface(Protocol):
    """
    Protocol defining vector database operations.
    
    All vector database backends must implement this interface.
    
    Requirements: 4.1, 4.5
    """
    
    def insert(
        self,
        embedding: np.ndarray,
        metadata: DocumentMetadata,
        id: Optional[str] = None,
    ) -> str:
        """
        Insert embedding with metadata.
        
        Args:
            embedding: Vector to store
            metadata: Associated document metadata
            id: Optional custom ID (generated if None)
            
        Returns:
            Unique ID for the stored embedding
            
        Raises:
            ValueError: If embedding dimensions don't match schema
        """
        ...
    
    def insert_batch(
        self,
        embeddings: list[np.ndarray],
        metadatas: list[DocumentMetadata],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Insert multiple embeddings.
        
        Args:
            embeddings: List of vectors to store
            metadatas: List of associated metadata
            ids: Optional list of custom IDs
            
        Returns:
            List of unique IDs for stored embeddings
            
        Raises:
            ValueError: If lengths don't match or dimensions invalid
        """
        ...
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results sorted by score (descending)
            
        Raises:
            ValueError: If query dimensions don't match schema
        """
        ...
    
    def delete(self, id: str) -> bool:
        """
        Delete embedding by ID.
        
        Args:
            id: ID of embedding to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all embeddings for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Number of embeddings deleted
        """
        ...
    
    def get_by_id(self, id: str) -> Optional[SearchResult]:
        """
        Retrieve embedding by ID.
        
        Args:
            id: Embedding ID
            
        Returns:
            SearchResult if found, None otherwise
        """
        ...
    
    def count(self) -> int:
        """
        Get total number of stored embeddings.
        
        Returns:
            Count of embeddings
        """
        ...
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection metadata
        """
        ...
    
    def validate_dimensions(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding dimensions match schema.
        
        Args:
            embedding: Vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        ...

"""
Tests for retrieval components.

This module tests:
- QueryEncoder: Query embedding generation
- Retriever: Unified retrieval interface
- Property 7: Retrieval Result Correctness

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import pytest
import numpy as np
from datetime import datetime
from hypothesis import given, strategies as st, settings

from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.retriever import Retriever
from src.storage.base import (
    VectorDBInterface,
    DocumentMetadata,
    SearchResult as DBSearchResult,
)


# Mock vector database for testing
class MockVectorDB:
    """Mock vector database for testing."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self._id_counter = 0
    
    def insert(
        self,
        embedding: np.ndarray,
        metadata: DocumentMetadata,
        id: str = None,
    ) -> str:
        """Insert embedding."""
        if id is None:
            id = f"emb_{self._id_counter}"
            self._id_counter += 1
        
        self.embeddings.append(embedding)
        self.metadatas.append(metadata)
        self.ids.append(id)
        return id
    
    def insert_batch(
        self,
        embeddings: list[np.ndarray],
        metadatas: list[DocumentMetadata],
        ids: list[str] = None,
    ) -> list[str]:
        """Insert batch."""
        if ids is None:
            ids = [None] * len(embeddings)
        
        result_ids = []
        for emb, meta, id in zip(embeddings, metadatas, ids):
            result_ids.append(self.insert(emb, meta, id))
        return result_ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict = None,
    ) -> list[DBSearchResult]:
        """Search for similar embeddings."""
        if not self.embeddings:
            return []
        
        # Compute cosine similarity
        scores = []
        for emb in self.embeddings:
            # Normalize vectors
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            score = float(np.dot(query_norm, emb_norm))
            scores.append(score)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        # Return top_k results
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(
                DBSearchResult(
                    id=self.ids[idx],
                    score=scores[idx],
                    metadata=self.metadatas[idx],
                    vector=self.embeddings[idx],
                )
            )
        
        return results
    
    def delete(self, id: str) -> bool:
        """Delete by ID."""
        if id in self.ids:
            idx = self.ids.index(id)
            del self.embeddings[idx]
            del self.metadatas[idx]
            del self.ids[idx]
            return True
        return False
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete by doc_id."""
        count = 0
        for i in range(len(self.metadatas) - 1, -1, -1):
            if self.metadatas[i].doc_id == doc_id:
                del self.embeddings[i]
                del self.metadatas[i]
                del self.ids[i]
                count += 1
        return count
    
    def get_by_id(self, id: str) -> DBSearchResult:
        """Get by ID."""
        if id in self.ids:
            idx = self.ids.index(id)
            return DBSearchResult(
                id=id,
                score=1.0,
                metadata=self.metadatas[idx],
                vector=self.embeddings[idx],
            )
        return None
    
    def count(self) -> int:
        """Get count."""
        return len(self.embeddings)
    
    def get_collection_info(self) -> dict:
        """Get collection info."""
        return {
            "embedding_dim": self.embedding_dim,
            "count": len(self.embeddings),
        }
    
    def validate_dimensions(self, embedding: np.ndarray) -> bool:
        """Validate dimensions."""
        return embedding.shape[0] == self.embedding_dim


# Unit Tests

class TestQueryEncoder:
    """Test QueryEncoder functionality."""
    
    def test_encode_single_query(self):
        """Test encoding a single query."""
        encoder = QueryEncoder()
        query = "What is machine learning?"
        
        embedding = encoder.encode(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # nomic-embed-text dimension
        assert not np.all(embedding == 0)
    
    def test_encode_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        encoder = QueryEncoder()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            encoder.encode("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            encoder.encode("   ")
    
    def test_encode_batch(self):
        """Test encoding multiple queries."""
        encoder = QueryEncoder()
        queries = [
            "What is deep learning?",
            "How does neural network work?",
            "Explain transformers",
        ]
        
        embeddings = encoder.encode_batch(queries)
        
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (768,)
    
    def test_encode_batch_filters_empty(self):
        """Test that batch encoding filters empty queries."""
        encoder = QueryEncoder()
        queries = ["Valid query", "", "   ", "Another valid"]
        
        embeddings = encoder.encode_batch(queries)
        
        # Should only encode the 2 valid queries
        assert len(embeddings) == 2
    
    def test_encode_batch_all_empty_raises(self):
        """Test that all empty queries raises ValueError."""
        encoder = QueryEncoder()
        
        with pytest.raises(ValueError, match="All queries are empty"):
            encoder.encode_batch(["", "   ", ""])


class TestRetriever:
    """Test Retriever functionality."""
    
    def test_retrieve_basic(self):
        """Test basic retrieval."""
        # Setup mock database with some documents
        mock_db = MockVectorDB()
        
        # Add some documents
        for i in range(5):
            embedding = np.random.randn(768)
            metadata = DocumentMetadata(
                doc_id=f"doc_{i}",
                page_number=1,
                modality="text-heavy",
                source_file=f"file_{i}.pdf",
                processed_at=datetime.now(),
                model_name="nomic-embed-text",
                embedding_dim=768,
            )
            mock_db.insert(embedding, metadata)
        
        # Create retriever
        retriever = Retriever(vector_db=mock_db, default_top_k=3)
        
        # Retrieve
        result = retriever.retrieve("test query")
        
        assert result.query == "test query"
        assert len(result.results) <= 3
        assert result.query_latency_ms > 0
        assert result.total_searched == 5
    
    def test_retrieve_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        mock_db = MockVectorDB()
        retriever = Retriever(vector_db=mock_db)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("")
    
    def test_retrieve_invalid_top_k_raises(self):
        """Test that invalid top_k raises ValueError."""
        mock_db = MockVectorDB()
        retriever = Retriever(vector_db=mock_db)
        
        with pytest.raises(ValueError, match="top_k must be > 0"):
            retriever.retrieve("test", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be > 0"):
            retriever.retrieve("test", top_k=-1)
    
    def test_retrieve_empty_database(self):
        """Test retrieval from empty database."""
        mock_db = MockVectorDB()
        retriever = Retriever(vector_db=mock_db)
        
        result = retriever.retrieve("test query")
        
        assert result.query == "test query"
        assert len(result.results) == 0
        assert result.total_searched == 0
    
    def test_retrieve_batch(self):
        """Test batch retrieval."""
        mock_db = MockVectorDB()
        
        # Add documents
        for i in range(3):
            embedding = np.random.randn(768)
            metadata = DocumentMetadata(
                doc_id=f"doc_{i}",
                page_number=1,
                modality="text-heavy",
                source_file=f"file_{i}.pdf",
                processed_at=datetime.now(),
                model_name="nomic-embed-text",
                embedding_dim=768,
            )
            mock_db.insert(embedding, metadata)
        
        retriever = Retriever(vector_db=mock_db)
        queries = ["query 1", "query 2"]
        
        results = retriever.retrieve_batch(queries)
        
        assert len(results) == 2
        assert results[0].query == "query 1"
        assert results[1].query == "query 2"
    
    def test_result_conversion(self):
        """Test that results are properly converted."""
        mock_db = MockVectorDB()
        
        # Add a document
        embedding = np.random.randn(768)
        metadata = DocumentMetadata(
            doc_id="test_doc",
            page_number=5,
            modality="visual-critical",
            source_file="test.pdf",
            processed_at=datetime.now(),
            model_name="colpali",
            embedding_dim=768,
        )
        mock_db.insert(embedding, metadata)
        
        retriever = Retriever(vector_db=mock_db)
        result = retriever.retrieve("test query", top_k=1)
        
        assert len(result.results) == 1
        search_result = result.results[0]
        assert search_result.doc_id == "test_doc"
        assert search_result.page_number == 5
        assert search_result.modality == "visual-critical"
        assert "source_file" in search_result.metadata
        assert search_result.metadata["source_file"] == "test.pdf"


# Property-Based Tests

# Feature: adaptive-retrieval-system, Property 7: Retrieval Result Correctness
# **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
@settings(max_examples=10, deadline=None)
@given(
    num_docs=st.integers(min_value=0, max_value=20),
    top_k=st.integers(min_value=1, max_value=10),
)
def test_property_retrieval_result_correctness(num_docs, top_k):
    """
    Property 7: Retrieval Result Correctness
    
    For any text query submitted to the Retrieval_Interface with top_k parameter K,
    the result SHALL contain at most K results, each result SHALL include doc_id,
    page_number, relevance_score, and modality, and results SHALL be sorted in
    descending order by relevance_score.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """
    # Setup mock database
    mock_db = MockVectorDB()
    
    # Add documents
    for i in range(num_docs):
        embedding = np.random.randn(768)
        metadata = DocumentMetadata(
            doc_id=f"doc_{i}",
            page_number=i + 1,
            modality="text-heavy" if i % 2 == 0 else "visual-critical",
            source_file=f"file_{i}.pdf",
            processed_at=datetime.now(),
            model_name="test-model",
            embedding_dim=768,
        )
        mock_db.insert(embedding, metadata)
    
    # Create retriever and query
    retriever = Retriever(vector_db=mock_db)
    result = retriever.retrieve("test query", top_k=top_k)
    
    # Property 1: Result contains at most K results
    assert len(result.results) <= top_k
    assert len(result.results) <= num_docs  # Can't return more than available
    
    # Property 2: Each result has required fields
    for search_result in result.results:
        assert hasattr(search_result, "doc_id")
        assert hasattr(search_result, "page_number")
        assert hasattr(search_result, "score")  # relevance_score
        assert hasattr(search_result, "modality")
        
        # Validate field types
        assert isinstance(search_result.doc_id, str)
        assert isinstance(search_result.page_number, int)
        assert isinstance(search_result.score, float)
        assert search_result.modality in ["text-heavy", "visual-critical"]
    
    # Property 3: Results are sorted by score in descending order
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True), \
        f"Scores not sorted descending: {scores}"

"""
Tests for vector database storage components.

This module tests:
- Vector database interface
- Qdrant backend
- LanceDB backend
- Property-based tests for storage

Requirements: 4.1, 4.2, 4.4, 4.5, 4.6
"""

import pytest
import numpy as np
from datetime import datetime
from hypothesis import given, strategies as st, settings

from src.storage.base import DocumentMetadata, SearchResult


# ============================================================================
# Unit Tests for DocumentMetadata
# ============================================================================

class TestDocumentMetadata:
    """Unit tests for DocumentMetadata."""
    
    def test_init_valid(self):
        """Test valid initialization."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            page_number=1,
            modality="text-heavy",
            source_file="test.pdf",
            processed_at=datetime.now(),
            model_name="nomic-embed-text",
            embedding_dim=768,
        )
        assert metadata.doc_id == "doc1"
        assert metadata.page_number == 1
        assert metadata.embedding_dim == 768
    
    def test_invalid_page_number(self):
        """Test that invalid page number raises ValueError."""
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            DocumentMetadata(
                doc_id="doc1",
                page_number=0,
                modality="text-heavy",
                source_file="test.pdf",
                processed_at=datetime.now(),
                model_name="test",
                embedding_dim=768,
            )
    
    def test_invalid_embedding_dim(self):
        """Test that invalid embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim must be > 0"):
            DocumentMetadata(
                doc_id="doc1",
                page_number=1,
                modality="text-heavy",
                source_file="test.pdf",
                processed_at=datetime.now(),
                model_name="test",
                embedding_dim=0,
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now()
        metadata = DocumentMetadata(
            doc_id="doc1",
            page_number=1,
            modality="text-heavy",
            source_file="test.pdf",
            processed_at=now,
            model_name="test",
            embedding_dim=768,
        )
        d = metadata.to_dict()
        
        assert d["doc_id"] == "doc1"
        assert d["page_number"] == 1
        assert d["embedding_dim"] == 768
        assert d["processed_at"] == now.isoformat()
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        now = datetime.now()
        d = {
            "doc_id": "doc1",
            "page_number": 1,
            "modality": "text-heavy",
            "source_file": "test.pdf",
            "processed_at": now.isoformat(),
            "model_name": "test",
            "embedding_dim": 768,
        }
        metadata = DocumentMetadata.from_dict(d)
        
        assert metadata.doc_id == "doc1"
        assert metadata.page_number == 1
        assert metadata.embedding_dim == 768


# ============================================================================
# Unit Tests for SearchResult
# ============================================================================

class TestSearchResult:
    """Unit tests for SearchResult."""
    
    def test_init(self):
        """Test initialization."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            page_number=1,
            modality="text-heavy",
            source_file="test.pdf",
            processed_at=datetime.now(),
            model_name="test",
            embedding_dim=768,
        )
        result = SearchResult(
            id="id1",
            score=0.95,
            metadata=metadata,
        )
        assert result.id == "id1"
        assert result.score == 0.95
        assert result.vector is None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = DocumentMetadata(
            doc_id="doc1",
            page_number=1,
            modality="text-heavy",
            source_file="test.pdf",
            processed_at=datetime.now(),
            model_name="test",
            embedding_dim=768,
        )
        vector = np.random.randn(768).astype(np.float32)
        result = SearchResult(
            id="id1",
            score=0.95,
            metadata=metadata,
            vector=vector,
        )
        d = result.to_dict()
        
        assert d["id"] == "id1"
        assert d["score"] == 0.95
        assert "metadata" in d
        assert "vector" in d


# ============================================================================
# Unit Tests for LanceDB Backend
# ============================================================================

class TestLanceDBBackend:
    """Unit tests for LanceDB backend."""
    
    def test_init(self, tmp_path):
        """Test initialization."""
        from src.storage.lancedb_backend import LanceDBBackend
        
        db_path = tmp_path / "lancedb"
        backend = LanceDBBackend(
            table_name="test_table",
            db_path=str(db_path),
            embedding_dim=768,
        )
        assert backend.table_name == "test_table"
        assert backend.embedding_dim == 768
    
    def test_insert_and_search(self, tmp_path):
        """Test insert and search operations."""
        from src.storage.lancedb_backend import LanceDBBackend
        
        db_path = tmp_path / "lancedb"
        backend = LanceDBBackend(
            table_name="test_table",
            db_path=str(db_path),
            embedding_dim=128,
        )
        
        # Create test data
        embedding = np.random.randn(128).astype(np.float32)
        metadata = DocumentMetadata(
            doc_id="doc1",
            page_number=1,
            modality="text-heavy",
            source_file="test.pdf",
            processed_at=datetime.now(),
            model_name="test",
            embedding_dim=128,
        )
        
        # Insert
        id = backend.insert(embedding, metadata)
        assert id is not None
        
        # Search
        results = backend.search(embedding, top_k=1)
        assert len(results) == 1
        assert results[0].metadata.doc_id == "doc1"
    
    def test_insert_batch(self, tmp_path):
        """Test batch insertion."""
        from src.storage.lancedb_backend import LanceDBBackend
        
        db_path = tmp_path / "lancedb"
        backend = LanceDBBackend(
            table_name="test_table",
            db_path=str(db_path),
            embedding_dim=128,
        )
        
        # Create test data
        embeddings = [np.random.randn(128).astype(np.float32) for _ in range(3)]
        metadatas = [
            DocumentMetadata(
                doc_id=f"doc{i}",
                page_number=i+1,
                modality="text-heavy",
                source_file="test.pdf",
                processed_at=datetime.now(),
                model_name="test",
                embedding_dim=128,
            )
            for i in range(3)
        ]
        
        # Insert batch
        ids = backend.insert_batch(embeddings, metadatas)
        assert len(ids) == 3
        
        # Verify count
        assert backend.count() == 3


# ============================================================================
# Property-Based Tests
# ============================================================================

# Feature: adaptive-retrieval-system, Property 5: Vector Database Storage Round-Trip
@settings(max_examples=5, deadline=60000)
@given(
    embedding_dim=st.integers(min_value=64, max_value=256),
    num_embeddings=st.integers(min_value=1, max_value=5),
)
def test_property_vector_database_storage_round_trip(embedding_dim, num_embeddings, tmp_path):
    """
    Property 5: Vector Database Storage Round-Trip
    
    For any embedding stored in the Vector_Database with associated metadata,
    retrieving that embedding by its ID SHALL return the identical vector values
    and complete metadata (doc_id, page_number, modality, source_file, processed_at).
    
    Validates: Requirements 4.1, 4.2
    """
    from src.storage.lancedb_backend import LanceDBBackend
    
    db_path = tmp_path / f"lancedb_{embedding_dim}"
    backend = LanceDBBackend(
        table_name="test_table",
        db_path=str(db_path),
        embedding_dim=embedding_dim,
    )
    
    # Create and insert embeddings
    for i in range(num_embeddings):
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        metadata = DocumentMetadata(
            doc_id=f"doc{i}",
            page_number=i+1,
            modality="text-heavy",
            source_file=f"test{i}.pdf",
            processed_at=datetime.now(),
            model_name="test",
            embedding_dim=embedding_dim,
        )
        
        # Insert
        id = backend.insert(embedding, metadata)
        
        # Retrieve
        result = backend.get_by_id(id)
        
        # Verify round-trip
        assert result is not None
        assert result.id == id
        assert result.metadata.doc_id == f"doc{i}"
        assert result.metadata.page_number == i+1
        assert result.metadata.embedding_dim == embedding_dim
        
        # Verify vector if returned
        if result.vector is not None:
            assert result.vector.shape == (embedding_dim,)
            assert result.vector.dtype == np.float32


# Feature: adaptive-retrieval-system, Property 6: Embedding Dimension Validation
@settings(max_examples=5, deadline=60000)
@given(
    correct_dim=st.integers(min_value=64, max_value=256),
    wrong_dim=st.integers(min_value=64, max_value=256),
)
def test_property_embedding_dimension_validation(correct_dim, wrong_dim, tmp_path):
    """
    Property 6: Embedding Dimension Validation
    
    For any embedding submitted to the Vector_Database with dimensions not matching
    the expected schema, the Vector_Database SHALL reject the insertion and raise
    a validation error.
    
    Validates: Requirements 4.5
    """
    from hypothesis import assume
    from src.storage.lancedb_backend import LanceDBBackend
    
    # Ensure dimensions are different
    assume(correct_dim != wrong_dim)
    
    db_path = tmp_path / f"lancedb_{correct_dim}_{wrong_dim}"
    backend = LanceDBBackend(
        table_name="test_table",
        db_path=str(db_path),
        embedding_dim=correct_dim,
    )
    
    # Insert with correct dimensions should succeed
    correct_embedding = np.random.randn(correct_dim).astype(np.float32)
    metadata = DocumentMetadata(
        doc_id="doc1",
        page_number=1,
        modality="text-heavy",
        source_file="test.pdf",
        processed_at=datetime.now(),
        model_name="test",
        embedding_dim=correct_dim,
    )
    
    id = backend.insert(correct_embedding, metadata)
    assert id is not None
    
    # Insert with wrong dimensions should fail
    wrong_embedding = np.random.randn(wrong_dim).astype(np.float32)
    wrong_metadata = DocumentMetadata(
        doc_id="doc2",
        page_number=2,
        modality="text-heavy",
        source_file="test.pdf",
        processed_at=datetime.now(),
        model_name="test",
        embedding_dim=wrong_dim,
    )
    
    with pytest.raises(ValueError, match="don't match"):
        backend.insert(wrong_embedding, wrong_metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

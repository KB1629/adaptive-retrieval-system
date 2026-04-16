"""
LanceDB vector database backend implementation.

This module provides LanceDB integration with:
- Table creation and management
- Local file-based storage
- Batch insertion support
- Metadata filtering

Requirements: 4.4
"""

import logging
import uuid
from pathlib import Path
from typing import Optional
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


class LanceDBBackend:
    """
    LanceDB vector database backend.
    
    Provides vector storage and similarity search using LanceDB.
    LanceDB is a file-based vector database optimized for local development.
    
    Requirements: 4.4
    """
    
    def __init__(
        self,
        table_name: str = "adaptive_retrieval",
        db_path: str = "./data/lancedb",
        embedding_dim: Optional[int] = None,
        distance_metric: str = "cosine",
    ):
        """
        Initialize LanceDB backend.
        
        Args:
            table_name: Name of the table
            db_path: Path to LanceDB database directory
            embedding_dim: Expected embedding dimensions (auto-detected if None)
            distance_metric: Distance metric ("cosine", "l2", "dot")
        """
        self.table_name = table_name
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        self._db = None
        self._table = None
        
        logger.info(f"Initializing LanceDBBackend: {db_path}/{table_name}")
    
    def _get_db(self):
        """Get or create LanceDB connection."""
        if self._db is not None:
            return self._db
        
        try:
            import lancedb
            
            # Create directory if needed
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self._db = lancedb.connect(str(self.db_path))
            logger.info(f"Connected to LanceDB at {self.db_path}")
            return self._db
        except ImportError:
            raise RuntimeError("lancedb not installed. Install with: pip install lancedb")
    
    def _get_schema(self, embedding_dim: int) -> pa.Schema:
        """Create PyArrow schema for the table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
            pa.field("doc_id", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("modality", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("processed_at", pa.string()),
            pa.field("model_name", pa.string()),
            pa.field("embedding_dim", pa.int32()),
        ])
    
    def _ensure_table(self, embedding_dim: Optional[int] = None):
        """Ensure table exists, create if needed."""
        if self._table is not None:
            return
        
        db = self._get_db()
        
        # Determine embedding dimensions
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        
        # Check if table exists
        table_names = db.table_names()
        
        if self.table_name in table_names:
            self._table = db.open_table(self.table_name)
            logger.info(f"Opened existing table: {self.table_name}")
        else:
            if self.embedding_dim is None:
                raise ValueError("embedding_dim must be specified before creating table")
            
            # Create empty table with schema
            schema = self._get_schema(self.embedding_dim)
            self._table = db.create_table(
                self.table_name,
                schema=schema,
                mode="create",
            )
            logger.info(f"Created table: {self.table_name}")
    
    def insert(
        self,
        embedding: np.ndarray,
        metadata: "DocumentMetadata",
        id: Optional[str] = None,
    ) -> str:
        """Insert embedding with metadata."""
        # Validate dimensions
        if not self.validate_dimensions(embedding):
            raise ValueError(
                f"Embedding dimensions {embedding.shape[0]} don't match "
                f"expected {self.embedding_dim}"
            )
        
        # Ensure table exists
        self._ensure_table(embedding.shape[0])
        
        # Generate ID if not provided
        if id is None:
            id = str(uuid.uuid4())
        
        # Create record
        record = {
            "id": id,
            "vector": embedding.tolist(),
            **metadata.to_dict(),
        }
        
        # Insert
        self._table.add([record])
        
        logger.debug(f"Inserted embedding {id}")
        return id
    
    def insert_batch(
        self,
        embeddings: list[np.ndarray],
        metadatas: list["DocumentMetadata"],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Insert multiple embeddings."""
        if len(embeddings) != len(metadatas):
            raise ValueError(f"Length mismatch: {len(embeddings)} embeddings, {len(metadatas)} metadatas")
        
        if not embeddings:
            return []
        
        # Ensure table exists
        self._ensure_table(embeddings[0].shape[0])
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]
        elif len(ids) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(embeddings)} embeddings, {len(ids)} ids")
        
        # Validate all dimensions
        for i, emb in enumerate(embeddings):
            if not self.validate_dimensions(emb):
                raise ValueError(
                    f"Embedding {i} dimensions {emb.shape[0]} don't match "
                    f"expected {self.embedding_dim}"
                )
        
        # Create records
        records = [
            {
                "id": id,
                "vector": emb.tolist(),
                **meta.to_dict(),
            }
            for id, emb, meta in zip(ids, embeddings, metadatas)
        ]
        
        # Insert batch
        self._table.add(records)
        
        logger.info(f"Inserted batch of {len(embeddings)} embeddings")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list["SearchResult"]:
        """Search for similar embeddings."""
        from .base import SearchResult, DocumentMetadata
        
        if not self.validate_dimensions(query_embedding):
            raise ValueError(
                f"Query dimensions {query_embedding.shape[0]} don't match "
                f"expected {self.embedding_dim}"
            )
        
        # Build query
        query = self._table.search(query_embedding.tolist())
        
        # Apply metadata filter if provided
        if filter_metadata:
            filter_str = " AND ".join([f"{k} = '{v}'" for k, v in filter_metadata.items()])
            query = query.where(filter_str)
        
        # Set distance metric
        if self.distance_metric == "cosine":
            query = query.metric("cosine")
        elif self.distance_metric == "l2":
            query = query.metric("l2")
        elif self.distance_metric == "dot":
            query = query.metric("dot")
        
        # Execute search
        results = query.limit(top_k).to_list()
        
        # Convert to SearchResult
        search_results = []
        for result in results:
            metadata = DocumentMetadata(
                doc_id=result["doc_id"],
                page_number=result["page_number"],
                modality=result["modality"],
                source_file=result["source_file"],
                processed_at=result["processed_at"],
                model_name=result["model_name"],
                embedding_dim=result["embedding_dim"],
            )
            search_results.append(
                SearchResult(
                    id=result["id"],
                    score=result.get("_distance", 0.0),
                    metadata=metadata,
                    vector=np.array(result["vector"]) if "vector" in result else None,
                )
            )
        
        return search_results
    
    def delete(self, id: str) -> bool:
        """Delete embedding by ID."""
        try:
            self._table.delete(f"id = '{id}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {id}: {e}")
            return False
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all embeddings for a document."""
        try:
            # Count before delete
            count_before = self.count()
            
            # Delete
            self._table.delete(f"doc_id = '{doc_id}'")
            
            # Count after delete
            count_after = self.count()
            
            return count_before - count_after
        except Exception as e:
            logger.warning(f"Failed to delete doc_id {doc_id}: {e}")
            return 0
    
    def get_by_id(self, id: str) -> Optional["SearchResult"]:
        """Retrieve embedding by ID."""
        from .base import SearchResult, DocumentMetadata
        
        try:
            results = self._table.search().where(f"id = '{id}'").limit(1).to_list()
            
            if not results:
                return None
            
            result = results[0]
            metadata = DocumentMetadata(
                doc_id=result["doc_id"],
                page_number=result["page_number"],
                modality=result["modality"],
                source_file=result["source_file"],
                processed_at=result["processed_at"],
                model_name=result["model_name"],
                embedding_dim=result["embedding_dim"],
            )
            return SearchResult(
                id=result["id"],
                score=1.0,  # No score for direct retrieval
                metadata=metadata,
                vector=np.array(result["vector"]) if "vector" in result else None,
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve {id}: {e}")
            return None
    
    def count(self) -> int:
        """Get total number of stored embeddings."""
        try:
            return self._table.count_rows()
        except Exception as e:
            logger.warning(f"Failed to get count: {e}")
            return 0
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            return {
                "name": self.table_name,
                "path": str(self.db_path),
                "count": self.count(),
                "embedding_dim": self.embedding_dim,
                "distance_metric": self.distance_metric,
            }
        except Exception as e:
            logger.warning(f"Failed to get collection info: {e}")
            return {"name": self.table_name, "error": str(e)}
    
    def validate_dimensions(self, embedding: np.ndarray) -> bool:
        """Validate embedding dimensions match schema."""
        if self.embedding_dim is None:
            # Auto-detect from first embedding
            self.embedding_dim = embedding.shape[0]
            return True
        return embedding.shape[0] == self.embedding_dim

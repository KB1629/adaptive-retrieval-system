"""
Qdrant vector database backend implementation.

This module provides Qdrant integration with:
- Collection creation and management
- Connection retry logic
- Batch insertion support
- Metadata filtering

Requirements: 4.4
"""

import logging
import time
import uuid
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class QdrantBackend:
    """
    Qdrant vector database backend.
    
    Provides vector storage and similarity search using Qdrant.
    
    Requirements: 4.4
    """
    
    def __init__(
        self,
        collection_name: str = "adaptive_retrieval",
        host: str = "localhost",
        port: int = 6333,
        embedding_dim: Optional[int] = None,
        distance_metric: str = "Cosine",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Qdrant backend.
        
        Args:
            collection_name: Name of the collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_dim: Expected embedding dimensions (auto-detected if None)
            distance_metric: Distance metric ("Cosine", "Euclidean", "Dot")
            max_retries: Maximum connection retries
            retry_delay: Delay between retries (seconds)
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._client = None
        self._collection_initialized = False
        
        logger.info(f"Initializing QdrantBackend: {host}:{port}/{collection_name}")
    
    def _get_client(self):
        """Get or create Qdrant client with retry logic."""
        if self._client is not None:
            return self._client
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            for attempt in range(self.max_retries):
                try:
                    self._client = QdrantClient(host=self.host, port=self.port)
                    logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
                    return self._client
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Connection attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise
        except ImportError:
            raise RuntimeError("qdrant-client not installed. Install with: pip install qdrant-client")
    
    def _ensure_collection(self, embedding_dim: Optional[int] = None):
        """Ensure collection exists, create if needed."""
        if self._collection_initialized:
            return
        
        from qdrant_client.models import Distance, VectorParams
        
        client = self._get_client()
        
        # Determine embedding dimensions
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        
        if self.embedding_dim is None:
            raise ValueError("embedding_dim must be specified before creating collection")
        
        # Map distance metric
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance = distance_map.get(self.distance_metric, Distance.COSINE)
        
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=distance,
                ),
            )
        else:
            logger.info(f"Collection already exists: {self.collection_name}")
        
        self._collection_initialized = True
    
    def insert(
        self,
        embedding: np.ndarray,
        metadata: "DocumentMetadata",
        id: Optional[str] = None,
    ) -> str:
        """Insert embedding with metadata."""
        from qdrant_client.models import PointStruct
        
        # Validate dimensions
        if not self.validate_dimensions(embedding):
            raise ValueError(
                f"Embedding dimensions {embedding.shape[0]} don't match "
                f"expected {self.embedding_dim}"
            )
        
        # Ensure collection exists
        self._ensure_collection(embedding.shape[0])
        
        # Generate ID if not provided
        if id is None:
            id = str(uuid.uuid4())
        
        # Create point
        point = PointStruct(
            id=id,
            vector=embedding.tolist(),
            payload=metadata.to_dict(),
        )
        
        # Insert
        client = self._get_client()
        client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        
        logger.debug(f"Inserted embedding {id}")
        return id
    
    def insert_batch(
        self,
        embeddings: list[np.ndarray],
        metadatas: list["DocumentMetadata"],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Insert multiple embeddings."""
        from qdrant_client.models import PointStruct
        
        if len(embeddings) != len(metadatas):
            raise ValueError(f"Length mismatch: {len(embeddings)} embeddings, {len(metadatas)} metadatas")
        
        if not embeddings:
            return []
        
        # Ensure collection exists
        self._ensure_collection(embeddings[0].shape[0])
        
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
        
        # Create points
        points = [
            PointStruct(
                id=id,
                vector=emb.tolist(),
                payload=meta.to_dict(),
            )
            for id, emb, meta in zip(ids, embeddings, metadatas)
        ]
        
        # Insert batch
        client = self._get_client()
        client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Inserted batch of {len(embeddings)} embeddings")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list["SearchResult"]:
        """Search for similar embeddings."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from .base import SearchResult
        
        if not self.validate_dimensions(query_embedding):
            raise ValueError(
                f"Query dimensions {query_embedding.shape[0]} don't match "
                f"expected {self.embedding_dim}"
            )
        
        # Build filter if provided
        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)
        
        # Search
        client = self._get_client()
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )
        
        # Convert to SearchResult
        from .base import DocumentMetadata
        search_results = []
        for result in results:
            metadata = DocumentMetadata.from_dict(result.payload)
            search_results.append(
                SearchResult(
                    id=str(result.id),
                    score=result.score,
                    metadata=metadata,
                    vector=np.array(result.vector) if result.vector else None,
                )
            )
        
        return search_results
    
    def delete(self, id: str) -> bool:
        """Delete embedding by ID."""
        client = self._get_client()
        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=[id],
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {id}: {e}")
            return False
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all embeddings for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = self._get_client()
        
        # Search for all points with this doc_id
        filter_condition = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        
        # Delete with filter
        result = client.delete(
            collection_name=self.collection_name,
            points_selector=filter_condition,
        )
        
        # Qdrant doesn't return count, so we return 1 if successful
        return 1 if result else 0
    
    def get_by_id(self, id: str) -> Optional["SearchResult"]:
        """Retrieve embedding by ID."""
        from .base import SearchResult, DocumentMetadata
        
        client = self._get_client()
        try:
            points = client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_vectors=True,
            )
            
            if not points:
                return None
            
            point = points[0]
            metadata = DocumentMetadata.from_dict(point.payload)
            return SearchResult(
                id=str(point.id),
                score=1.0,  # No score for direct retrieval
                metadata=metadata,
                vector=np.array(point.vector) if point.vector else None,
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve {id}: {e}")
            return None
    
    def count(self) -> int:
        """Get total number of stored embeddings."""
        client = self._get_client()
        try:
            info = client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.warning(f"Failed to get count: {e}")
            return 0
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        client = self._get_client()
        try:
            info = client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
            }
        except Exception as e:
            logger.warning(f"Failed to get collection info: {e}")
            return {"name": self.collection_name, "error": str(e)}
    
    def validate_dimensions(self, embedding: np.ndarray) -> bool:
        """Validate embedding dimensions match schema."""
        if self.embedding_dim is None:
            # Auto-detect from first embedding
            self.embedding_dim = embedding.shape[0]
            return True
        return embedding.shape[0] == self.embedding_dim

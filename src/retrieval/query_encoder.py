"""
Query encoder for text queries.

This module provides query embedding generation using the same
text embedding model as the text path for consistency.

Requirements: 5.1
"""

import logging
from typing import Optional
import numpy as np

from ..embedding.text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


class QueryEncoder:
    """
    Encodes text queries into embedding vectors.
    
    Uses nomic-embed-text for consistency with the text embedding path.
    
    Requirements: 5.1
    """
    
    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize query encoder.
        
        Args:
            embedder: Text embedder (creates default if None)
            device: Device to use ("mps", "cuda", "cpu", or None for auto)
        """
        if embedder is None:
            self.embedder = TextEmbedder(device=device)
        else:
            self.embedder = embedder
        
        logger.info("QueryEncoder initialized")
    
    def encode(self, query: str) -> np.ndarray:
        """
        Encode a text query into an embedding vector.
        
        Args:
            query: Text query string
            
        Returns:
            Query embedding vector
            
        Raises:
            ValueError: If query is empty or encoding fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            embedding = self.embedder.embed(query)
            logger.debug(f"Encoded query: '{query[:50]}...' -> {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            raise ValueError(f"Query encoding failed: {e}")
    
    def encode_batch(self, queries: list[str]) -> list[np.ndarray]:
        """
        Encode multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of query embedding vectors
            
        Raises:
            ValueError: If queries is empty or encoding fails
        """
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        # Filter out empty queries
        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            raise ValueError("All queries are empty")
        
        if len(valid_queries) < len(queries):
            logger.warning(f"Filtered out {len(queries) - len(valid_queries)} empty queries")
        
        try:
            embeddings = self.embedder.embed_batch(valid_queries)
            logger.debug(f"Encoded {len(embeddings)} queries")
            return embeddings
        except Exception as e:
            logger.error(f"Batch query encoding failed: {e}")
            raise ValueError(f"Batch query encoding failed: {e}")
    
    @property
    def embedding_dimensions(self) -> int:
        """Return embedding dimensions."""
        return self.embedder.embedding_dimensions
    
    def get_encoder_info(self) -> dict:
        """Get encoder information."""
        return {
            "model_name": self.embedder.model_name,
            "embedding_dim": self.embedder.embedding_dimensions,
            "device": self.embedder.device,
        }

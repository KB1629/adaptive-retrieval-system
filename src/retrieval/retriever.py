"""
Unified retrieval interface.

This module provides the main retrieval interface that integrates
with vector databases to perform similarity search.

Requirements: 5.1, 5.2, 5.4
"""

import logging
import time
from typing import Optional
import numpy as np

from ..storage.base import VectorDBInterface, SearchResult as DBSearchResult
from ..models.results import SearchResult, QueryResult
from .query_encoder import QueryEncoder

logger = logging.getLogger(__name__)


class Retriever:
    """
    Unified retrieval interface for similarity search.
    
    Integrates query encoding with vector database search to provide
    end-to-end retrieval functionality.
    
    Requirements: 5.1, 5.2, 5.4
    """
    
    def __init__(
        self,
        vector_db: VectorDBInterface,
        query_encoder: Optional[QueryEncoder] = None,
        default_top_k: int = 10,
    ):
        """
        Initialize retriever.
        
        Args:
            vector_db: Vector database backend
            query_encoder: Query encoder (creates default if None)
            default_top_k: Default number of results to return
        """
        self.vector_db = vector_db
        self.query_encoder = query_encoder or QueryEncoder()
        self.default_top_k = default_top_k
        
        logger.info(f"Retriever initialized with top_k={default_top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> QueryResult:
        """
        Retrieve documents for a query.
        
        Args:
            query: Text query string
            top_k: Number of results to return (uses default if None)
            filter_metadata: Optional metadata filters
            
        Returns:
            QueryResult with search results and metadata
            
        Raises:
            ValueError: If query is invalid or retrieval fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            raise ValueError(f"top_k must be > 0, got {k}")
        
        # Measure query latency
        start_time = time.perf_counter()
        
        try:
            # Encode query
            query_embedding = self.query_encoder.encode(query)
            
            # Search vector database
            db_results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=k,
                filter_metadata=filter_metadata,
            )
            
            # Convert to SearchResult format
            results = self._convert_results(db_results)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Get total count
            total_searched = self.vector_db.count()
            
            logger.info(
                f"Retrieved {len(results)} results for query '{query[:50]}...' "
                f"in {latency_ms:.2f}ms"
            )
            
            return QueryResult(
                query=query,
                results=results,
                query_latency_ms=latency_ms,
                total_searched=total_searched,
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
            raise ValueError(f"Retrieval failed: {e}")
    
    def retrieve_batch(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[QueryResult]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            filter_metadata: Optional metadata filters
            
        Returns:
            List of QueryResult objects
            
        Raises:
            ValueError: If queries is empty or retrieval fails
        """
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        results = []
        for query in queries:
            try:
                result = self.retrieve(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to retrieve for query '{query[:50]}...': {e}")
                # Add empty result for failed query
                results.append(QueryResult(query=query))
        
        logger.info(f"Batch retrieval completed for {len(queries)} queries")
        return results
    
    def _convert_results(self, db_results: list[DBSearchResult]) -> list[SearchResult]:
        """
        Convert database search results to SearchResult format.
        
        Args:
            db_results: Results from vector database
            
        Returns:
            List of SearchResult objects
        """
        results = []
        for db_result in db_results:
            result = SearchResult(
                doc_id=db_result.metadata.doc_id,
                page_number=db_result.metadata.page_number,
                score=db_result.score,
                modality=db_result.metadata.modality,
                metadata={
                    "source_file": db_result.metadata.source_file,
                    "model_name": db_result.metadata.model_name,
                    "processed_at": db_result.metadata.processed_at.isoformat(),
                },
            )
            results.append(result)
        
        return results
    
    def get_retriever_info(self) -> dict:
        """
        Get retriever configuration information.
        
        Returns:
            Dictionary with retriever metadata
        """
        return {
            "default_top_k": self.default_top_k,
            "encoder_info": self.query_encoder.get_encoder_info(),
            "vector_db_info": self.vector_db.get_collection_info(),
        }

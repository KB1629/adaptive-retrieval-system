"""
Text embedding path pipeline combining extraction and embedding.

This module provides the complete text processing pipeline:
1. Extract text from page using PyMuPDF
2. Generate embedding using nomic-embed-text
3. Handle errors with escalation to vision path

Requirements: 2.1, 2.5
"""

import logging
import time
from typing import Optional
import numpy as np

from ..models.data import EmbeddingResult
from .text_extractor import TextExtractor
from .text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


class TextEmbeddingPath:
    """
    Complete text embedding pipeline.
    
    Combines text extraction and embedding generation with:
    - Error handling and fallback
    - Performance tracking
    - Escalation to vision path on failure
    
    Requirements: 2.1, 2.5
    """
    
    def __init__(
        self,
        extractor: Optional[TextExtractor] = None,
        embedder: Optional[TextEmbedder] = None,
        escalate_on_failure: bool = True,
    ):
        """
        Initialize text embedding path.
        
        Args:
            extractor: Text extractor (creates default if None)
            embedder: Text embedder (creates default if None)
            escalate_on_failure: Whether to raise exception on failure
        """
        self.extractor = extractor or TextExtractor()
        self.embedder = embedder or TextEmbedder()
        self.escalate_on_failure = escalate_on_failure
        
        logger.info("TextEmbeddingPath initialized")
    
    def process_page(self, page_image: np.ndarray) -> EmbeddingResult:
        """
        Process a page through the text embedding path.
        
        Args:
            page_image: Page rendered as numpy array (RGB, HxWx3)
            
        Returns:
            EmbeddingResult with embedding and metadata
            
        Raises:
            ValueError: If processing fails and escalate_on_failure is True
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract text
            try:
                text = self.extractor.extract_from_image(page_image)
                logger.debug(f"Extracted {len(text)} characters of text")
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                if self.escalate_on_failure:
                    raise ValueError(f"Text extraction failed, escalate to vision path: {e}")
                raise
            
            # Step 2: Generate embedding
            try:
                embedding = self.embedder.embed(text)
                logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                if self.escalate_on_failure:
                    raise ValueError(f"Embedding generation failed, escalate to vision path: {e}")
                raise
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = EmbeddingResult(
                vector=embedding,
                modality="text-heavy",
                processing_time_ms=processing_time_ms,
                model_name=self.embedder.model_name,
                extracted_text=text,
            )
            
            logger.info(f"Text embedding completed in {processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Text embedding path failed after {processing_time_ms:.2f}ms: {e}")
            raise
    
    def process_batch(self, page_images: list[np.ndarray]) -> list[EmbeddingResult]:
        """
        Process multiple pages through the text embedding path.
        
        Args:
            page_images: List of page images
            
        Returns:
            List of EmbeddingResult objects
            
        Raises:
            ValueError: If batch is empty or processing fails
        """
        if not page_images:
            raise ValueError("Page images list cannot be empty")
        
        logger.info(f"Processing batch of {len(page_images)} pages")
        start_time = time.time()
        
        # Extract text from all pages
        texts = []
        failed_indices = []
        
        for i, page_image in enumerate(page_images):
            try:
                text = self.extractor.extract_from_image(page_image)
                texts.append(text)
            except Exception as e:
                logger.warning(f"Text extraction failed for page {i}: {e}")
                texts.append("")  # Placeholder
                failed_indices.append(i)
        
        # Generate embeddings for successfully extracted texts
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            raise ValueError("All text extractions failed, escalate to vision path")
        
        try:
            embeddings = self.embedder.embed_batch(valid_texts)
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise ValueError(f"Batch embedding failed, escalate to vision path: {e}")
        
        # Create results
        results = []
        embedding_idx = 0
        processing_time_ms = (time.time() - start_time) * 1000
        avg_time_per_page = processing_time_ms / len(page_images)
        
        for i, text in enumerate(texts):
            if i in failed_indices:
                # Create error result for failed pages
                logger.warning(f"Page {i} failed extraction, should escalate to vision path")
                continue
            
            result = EmbeddingResult(
                vector=embeddings[embedding_idx],
                modality="text-heavy",
                processing_time_ms=avg_time_per_page,
                model_name=self.embedder.model_name,
                extracted_text=text,
            )
            results.append(result)
            embedding_idx += 1
        
        logger.info(
            f"Batch processing completed: {len(results)}/{len(page_images)} successful "
            f"in {processing_time_ms:.2f}ms"
        )
        
        return results
    
    def get_pipeline_info(self) -> dict:
        """
        Get pipeline configuration information.
        
        Returns:
            Dictionary with pipeline details
        """
        return {
            "extractor": {
                "preserve_structure": self.extractor.preserve_structure,
                "min_text_length": self.extractor.min_text_length,
            },
            "embedder": self.embedder.get_model_info(),
            "escalate_on_failure": self.escalate_on_failure,
        }

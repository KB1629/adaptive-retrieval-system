"""
Vision embedding path pipeline for visual-critical pages.

This module provides the complete vision processing pipeline:
1. Preprocess page image
2. Generate embedding using ColPali/SigLIP
3. Handle OOM errors with batch size reduction
4. CPU fallback for memory issues

Requirements: 3.1, 3.6
"""

import logging
import time
from typing import Optional
import numpy as np

from ..models.data import EmbeddingResult
from .vision_embedder import VisionEmbedder

logger = logging.getLogger(__name__)


class VisionEmbeddingPath:
    """
    Complete vision embedding pipeline.
    
    Handles visual-critical pages with:
    - ColPali/SigLIP embedding generation
    - OOM handling with batch size reduction
    - CPU fallback for memory issues
    - Performance tracking
    
    Requirements: 3.1, 3.6
    """
    
    def __init__(
        self,
        embedder: Optional[VisionEmbedder] = None,
        lora_weights_path: Optional[str] = None,
        fallback_to_cpu: bool = True,
    ):
        """
        Initialize vision embedding path.
        
        Args:
            embedder: Vision embedder (creates default if None)
            lora_weights_path: Optional path to LoRA weights
            fallback_to_cpu: Whether to fall back to CPU on OOM
        """
        self.embedder = embedder or VisionEmbedder()
        self.fallback_to_cpu = fallback_to_cpu
        self._cpu_fallback_embedder: Optional[VisionEmbedder] = None
        
        # Load LoRA weights if provided
        if lora_weights_path:
            try:
                self.embedder.load_lora_weights(lora_weights_path)
                logger.info(f"Loaded LoRA weights from {lora_weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}, using base model")
        
        logger.info("VisionEmbeddingPath initialized")
    
    def process_page(self, page_image: np.ndarray) -> EmbeddingResult:
        """
        Process a page through the vision embedding path.
        
        Args:
            page_image: Page rendered as numpy array (RGB, HxWx3)
            
        Returns:
            EmbeddingResult with embedding and metadata
            
        Raises:
            ValueError: If processing fails after all fallbacks
        """
        start_time = time.time()
        
        try:
            embedding = self._embed_with_fallback(page_image)
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = EmbeddingResult(
                vector=embedding,
                modality="visual-critical",
                processing_time_ms=processing_time_ms,
                model_name=self.embedder.model_name,
                extracted_text=None,
            )
            
            logger.info(f"Vision embedding completed in {processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Vision embedding path failed after {processing_time_ms:.2f}ms: {e}")
            raise ValueError(f"Vision embedding failed: {e}")
    
    def _embed_with_fallback(self, page_image: np.ndarray) -> np.ndarray:
        """Embed with OOM fallback to CPU."""
        try:
            return self.embedder.embed(page_image)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.fallback_to_cpu:
                logger.warning("OOM error, falling back to CPU")
                return self._embed_on_cpu(page_image)
            raise
    
    def _embed_on_cpu(self, page_image: np.ndarray) -> np.ndarray:
        """Embed using CPU fallback."""
        if self._cpu_fallback_embedder is None:
            logger.info("Creating CPU fallback embedder")
            self._cpu_fallback_embedder = VisionEmbedder(
                model_name=self.embedder.model_name,
                device="cpu",
                batch_size=1,
            )
        
        return self._cpu_fallback_embedder.embed(page_image)
    
    def process_batch(self, page_images: list[np.ndarray]) -> list[EmbeddingResult]:
        """
        Process multiple pages through the vision embedding path.
        
        Args:
            page_images: List of page images
            
        Returns:
            List of EmbeddingResult objects
        """
        if not page_images:
            raise ValueError("Page images list cannot be empty")
        
        logger.info(f"Processing batch of {len(page_images)} pages")
        start_time = time.time()
        
        try:
            embeddings = self._embed_batch_with_fallback(page_images)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fall back to individual processing
            logger.info("Falling back to individual processing")
            embeddings = []
            for img in page_images:
                try:
                    emb = self._embed_with_fallback(img)
                    embeddings.append(emb)
                except Exception as inner_e:
                    logger.error(f"Individual embedding failed: {inner_e}")
                    raise ValueError(f"Vision embedding failed: {inner_e}")
        
        processing_time_ms = (time.time() - start_time) * 1000
        avg_time_per_page = processing_time_ms / len(page_images)
        
        results = [
            EmbeddingResult(
                vector=emb,
                modality="visual-critical",
                processing_time_ms=avg_time_per_page,
                model_name=self.embedder.model_name,
                extracted_text=None,
            )
            for emb in embeddings
        ]
        
        logger.info(
            f"Batch processing completed: {len(results)} pages "
            f"in {processing_time_ms:.2f}ms ({avg_time_per_page:.2f}ms/page)"
        )
        
        return results
    
    def _embed_batch_with_fallback(self, page_images: list[np.ndarray]) -> list[np.ndarray]:
        """Embed batch with OOM fallback."""
        try:
            return self.embedder.embed_batch(page_images)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM on batch, reducing batch size")
                return self._embed_batch_reduced(page_images)
            raise
    
    def _embed_batch_reduced(self, page_images: list[np.ndarray]) -> list[np.ndarray]:
        """Embed with reduced batch size on OOM."""
        results = []
        reduced_batch_size = max(1, self.embedder.batch_size // 2)
        
        for i in range(0, len(page_images), reduced_batch_size):
            batch = page_images[i:i + reduced_batch_size]
            try:
                if reduced_batch_size == 1:
                    # Single image, use individual embed
                    for img in batch:
                        results.append(self._embed_with_fallback(img))
                else:
                    results.extend(self.embedder.embed_batch(batch))
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.fallback_to_cpu:
                    # Fall back to CPU for this batch
                    for img in batch:
                        results.append(self._embed_on_cpu(img))
                else:
                    raise
        
        return results
    
    def get_pipeline_info(self) -> dict:
        """Get pipeline configuration information."""
        return {
            "embedder": self.embedder.get_model_info(),
            "fallback_to_cpu": self.fallback_to_cpu,
            "has_cpu_fallback": self._cpu_fallback_embedder is not None,
        }

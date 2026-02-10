"""
Text embedding generation using nomic-embed-text.

This module provides efficient text embedding with:
- Batch processing for throughput optimization
- Token limit handling with truncation
- Hardware-aware device selection

Requirements: 2.1, 2.4, 2.6
"""

import logging
from typing import Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Generates embeddings for text using nomic-embed-text model.
    
    Features:
    - 768-dimensional embeddings
    - Batch processing support
    - Automatic token truncation
    - Hardware-aware (MPS/CUDA/CPU)
    
    Requirements: 2.1, 2.4, 2.6
    """
    
    MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIM = 768
    MAX_TOKENS = 8192
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize text embedder.
        
        Args:
            model_name: Model name (defaults to nomic-embed-text-v1.5)
            device: Device to use ("mps", "cuda", "cpu", or None for auto)
            batch_size: Batch size for processing
            max_tokens: Maximum tokens per text (defaults to 8192)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.batch_size = batch_size
        self.max_tokens = max_tokens or self.MAX_TOKENS
        
        # Auto-detect device if not specified
        if device is None:
            device = self._detect_device()
        self.device = device
        
        logger.info(f"Initializing TextEmbedder with model={self.model_name}, device={self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            self.model.max_seq_length = self.max_tokens
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load text embedding model: {e}")
    
    def _detect_device(self) -> str:
        """
        Auto-detect available hardware.
        
        Returns:
            Device string ("mps", "cuda", or "cpu")
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector (768 dimensions)
            
        Raises:
            ValueError: If text is empty or embedding fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Truncate if needed (model handles this internally, but we log it)
            if len(text) > self.max_tokens * 4:  # Rough estimate (4 chars per token)
                logger.warning(f"Text length {len(text)} may exceed token limit, will be truncated")
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            
            # Validate dimensions
            if embedding.shape[0] != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {self.EMBEDDING_DIM} dimensions, got {embedding.shape[0]}"
                )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ValueError(f"Embedding generation failed: {e}")
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is empty or embedding fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        if len(valid_texts) < len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 100,
                normalize_embeddings=True,
            )
            
            # Validate dimensions
            if embeddings.shape[1] != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {self.EMBEDDING_DIM} dimensions, got {embeddings.shape[1]}"
                )
            
            # Convert to list of arrays
            return [emb.astype(np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise ValueError(f"Batch embedding generation failed: {e}")
    
    @property
    def embedding_dimensions(self) -> int:
        """Return embedding dimensions."""
        return self.EMBEDDING_DIM
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.EMBEDDING_DIM,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "batch_size": self.batch_size,
        }

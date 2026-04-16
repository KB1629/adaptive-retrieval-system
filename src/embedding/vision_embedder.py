"""
Vision embedding generation using ColPali model.

This module provides vision-based document embedding with:
- ColPali model loading and inference
- Image preprocessing
- Batch processing with memory management
- LoRA weight loading support

Requirements: 3.1, 3.4, 3.5
"""

import logging
from typing import Optional
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VisionEmbedder:
    """
    Generates embeddings for document images using ColPali.
    
    ColPali produces multi-vector embeddings (~1000 vectors per page)
    that capture spatial relationships in documents.
    
    For this implementation, we use a simplified approach that:
    1. Uses the ColPali processor for image preprocessing
    2. Generates embeddings via the vision encoder
    3. Pools to a single vector for compatibility with text embeddings
    
    Requirements: 3.1, 3.4, 3.5
    """
    
    MODEL_NAME = "vidore/colpali-v1.2"
    EMBEDDING_DIM = 128  # ColPali embedding dimension per patch
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
        use_pooled: bool = True,
    ):
        """
        Initialize vision embedder.
        
        Args:
            model_name: Model name (defaults to ColPali v1.2)
            device: Device to use ("mps", "cuda", "cpu", or None for auto)
            batch_size: Batch size for processing
            use_pooled: Whether to pool embeddings to single vector
        """
        self.model_name = model_name or self.MODEL_NAME
        self.batch_size = batch_size
        self.use_pooled = use_pooled
        self._lora_loaded = False
        self._lora_path: Optional[str] = None
        
        # Auto-detect device if not specified
        if device is None:
            device = self._detect_device()
        self.device = device
        
        logger.info(f"Initializing VisionEmbedder with model={self.model_name}, device={self.device}")
        
        # Lazy loading - model loaded on first use
        self._model = None
        self._processor = None
    
    def _detect_device(self) -> str:
        """Auto-detect available hardware."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load model and processor lazily."""
        if self._model is not None:
            return
        
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
            
            logger.info(f"Loading ColPali model: {self.model_name}")
            
            # Load model
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=self.device if self.device != "mps" else None,
            )
            
            if self.device == "mps":
                self._model = self._model.to(self.device)
            
            self._model.eval()
            
            # Load processor - use ColPaliProcessor instead of BaseVisualRetrieverProcessor
            self._processor = ColPaliProcessor.from_pretrained(self.model_name)
            
            logger.info(f"ColPali model loaded successfully on {self.device}")
            
        except ImportError:
            logger.warning("colpali_engine not installed, using fallback SigLIP model")
            self._load_fallback_model()
        except Exception as e:
            logger.warning(f"Failed to load ColPali: {e}, using fallback")
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load fallback SigLIP model if ColPali unavailable."""
        from transformers import AutoProcessor, AutoModel
        
        fallback_model = "google/siglip-base-patch16-224"
        logger.info(f"Loading fallback model: {fallback_model}")
        
        self._processor = AutoProcessor.from_pretrained(fallback_model)
        self._model = AutoModel.from_pretrained(
            fallback_model,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        self._model.eval()
        
        # Update model name to reflect fallback
        self.model_name = fallback_model
        logger.info(f"Fallback model loaded on {self.device}")
    
    def load_lora_weights(self, weights_path: str) -> None:
        """
        Load LoRA fine-tuned weights.
        
        Args:
            weights_path: Path to LoRA weights file
            
        Raises:
            ValueError: If weights are incompatible
        """
        self._load_model()
        
        try:
            from peft import PeftModel
            
            logger.info(f"Loading LoRA weights from: {weights_path}")
            self._model = PeftModel.from_pretrained(self._model, weights_path)
            self._model.eval()
            self._lora_loaded = True
            self._lora_path = weights_path
            logger.info("LoRA weights loaded successfully")
            
        except ImportError:
            logger.error("peft library not installed, cannot load LoRA weights")
            raise ValueError("peft library required for LoRA weights")
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            raise ValueError(f"Incompatible LoRA weights: {e}")
    
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: Image as numpy array (RGB, HxWx3)
            
        Returns:
            Embedding vector
            
        Raises:
            ValueError: If embedding fails
        """
        self._load_model()
        
        try:
            # Convert numpy to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
            else:
                pil_image = image
            
            # Process image
            with torch.no_grad():
                if hasattr(self._processor, 'process_images'):
                    # ColPali processor
                    inputs = self._processor.process_images([pil_image]).to(self.device)
                    outputs = self._model(**inputs)
                    
                    # ColPali returns multi-vector embeddings
                    # Pool to single vector for compatibility
                    if self.use_pooled:
                        embedding = outputs.mean(dim=1).squeeze().cpu().numpy()
                    else:
                        embedding = outputs.squeeze().cpu().numpy()
                else:
                    # Fallback SigLIP processor
                    inputs = self._processor(images=pil_image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self._model.get_image_features(**inputs)
                    embedding = outputs.squeeze().cpu().numpy()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Vision embedding failed: {e}")
            raise ValueError(f"Vision embedding failed: {e}")

    def encode_text_query(self, query: str) -> np.ndarray:
        """
        Encode a text query using ColPali's query encoder.

        ColPali uses a UNIFIED embedding space — both document images and text
        queries are projected into the same 128-dim space. This method uses
        ColPali's native `process_queries()` so the resulting vector can be
        directly compared (cosine similarity) against image embeddings stored
        in the vision DB.

        Args:
            query: Text query string

        Returns:
            128-dim query embedding in the same space as image embeddings.

        Raises:
            ValueError: If encoding fails
        """
        self._load_model()

        try:
            with torch.no_grad():
                if hasattr(self._processor, 'process_queries'):
                    # ── ColPali native text query encoding ──────────────────
                    inputs = self._processor.process_queries([query]).to(self.device)
                    outputs = self._model(**inputs)
                    # Pool across sequence length → (128,)
                    embedding = outputs.mean(dim=1).squeeze().cpu().numpy()
                else:
                    # ── Fallback: SigLIP text encoder ───────────────────────
                    inputs = self._processor(text=query, return_tensors="pt",
                                             padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self._model.get_text_features(**inputs)
                    embedding = outputs.squeeze().cpu().numpy()

            embedding = embedding.astype(np.float32)
            # L2-normalise so cosine similarity == dot product
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            return embedding

        except Exception as e:
            logger.error(f"Text query encoding failed: {e}")
            raise ValueError(f"Text query encoding failed: {e}")

    def encode_text_queries_batch(self, queries: list[str]) -> list[np.ndarray]:
        """
        Encode a batch of text queries using ColPali's query encoder.

        Args:
            queries: List of query strings

        Returns:
            List of 128-dim query embeddings
        """
        self._load_model()

        try:
            with torch.no_grad():
                if hasattr(self._processor, 'process_queries'):
                    inputs = self._processor.process_queries(queries).to(self.device)
                    outputs = self._model(**inputs)
                    embeddings = outputs.mean(dim=1).cpu().numpy()
                else:
                    inputs = self._processor(text=queries, return_tensors="pt",
                                             padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self._model.get_text_features(**inputs)
                    embeddings = outputs.cpu().numpy()

            result = []
            for emb in embeddings:
                emb = emb.astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm
                result.append(emb)
            return result

        except Exception as e:
            logger.error(f"Batch text query encoding failed: {e}")
            raise ValueError(f"Batch text query encoding failed: {e}")


    
    def embed_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If batch embedding fails
        """
        if not images:
            raise ValueError("Images list cannot be empty")
        
        self._load_model()
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            try:
                batch_embeddings = self._embed_batch_internal(batch)
                results.extend(batch_embeddings)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch size {len(batch)}, processing individually")
                    # Fall back to individual processing
                    for img in batch:
                        results.append(self.embed(img))
                else:
                    raise
        
        return results
    
    def _embed_batch_internal(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Internal batch embedding without memory fallback."""
        # Convert to PIL images
        pil_images = [
            Image.fromarray(img.astype('uint8'), 'RGB') if isinstance(img, np.ndarray) else img
            for img in images
        ]
        
        with torch.no_grad():
            if hasattr(self._processor, 'process_images'):
                # ColPali processor
                inputs = self._processor.process_images(pil_images).to(self.device)
                outputs = self._model(**inputs)
                
                if self.use_pooled:
                    embeddings = outputs.mean(dim=1).cpu().numpy()
                else:
                    embeddings = outputs.cpu().numpy()
            else:
                # Fallback SigLIP processor
                inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()
        
        return [emb.astype(np.float32) for emb in embeddings]
    
    @property
    def embedding_dimensions(self) -> int:
        """Return embedding dimensions."""
        self._load_model()
        # Get actual dimension from a test embedding
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        emb = self.embed(test_img)
        return emb.shape[-1]
    
    @property
    def has_lora(self) -> bool:
        """Check if LoRA weights are loaded."""
        return self._lora_loaded
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "use_pooled": self.use_pooled,
            "lora_loaded": self._lora_loaded,
            "lora_path": self._lora_path,
        }

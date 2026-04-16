"""
ML-based router using image classification.

Uses a lightweight vision model to classify pages.
Falls back to heuristic router if model unavailable.

Requirements: 1.6
"""

import logging
from typing import Optional

import numpy as np

from src.models import Page, ClassificationResult
from src.utils.hardware import detect_device, get_device_string
from .base import BaseRouter, RouterConfig
from .heuristic import HeuristicRouter

logger = logging.getLogger(__name__)


class MLRouter(BaseRouter):
    """
    ML-based router using a vision classifier.
    
    Uses a lightweight model (e.g., MobileNet or custom trained)
    to classify pages. Falls back to heuristic if model loading fails.
    """
    
    def __init__(
        self,
        config: RouterConfig = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize ML router.
        
        Args:
            config: Router configuration
            model_path: Path to trained model weights (optional)
        """
        super().__init__(config)
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._device = None
        self._fallback = HeuristicRouter(config)
        self._model_loaded = False
        
        # Try to load model
        self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """
        Attempt to load the ML model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Detect device
            device_type = detect_device()
            self._device = get_device_string(device_type)
            
            # Use a lightweight pretrained model for document classification
            # In production, this would be a custom-trained model
            model_name = self.model_path or "microsoft/resnet-18"
            
            logger.info(f"Loading ML router model: {model_name}")
            
            self._processor = AutoImageProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()
            
            self._model_loaded = True
            logger.info(f"ML router model loaded on {self._device}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load ML router model: {e}. Using heuristic fallback.")
            self._model_loaded = False
            return False
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if ML model is loaded."""
        return self._model_loaded
    
    def _classify_impl(self, page: Page) -> ClassificationResult:
        """
        Classify page using ML model or fallback.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult
        """
        if not self._model_loaded:
            return self._fallback._classify_impl(page)
        
        try:
            return self._classify_with_model(page)
        except Exception as e:
            logger.warning(f"ML classification failed: {e}. Using fallback.")
            return self._fallback._classify_impl(page)
    
    def _classify_with_model(self, page: Page) -> ClassificationResult:
        """
        Classify using the loaded ML model.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult
        """
        import torch
        
        # Preprocess image
        inputs = self._processor(
            images=page.image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        # For a generic model, we use heuristics on the output
        # A custom-trained model would have text-heavy/visual-critical classes
        confidence = float(probs.max())
        
        # Use output features to determine modality
        # This is a simplified approach - real implementation would
        # use a model trained specifically for this task
        features = self._extract_model_features(page, probs)
        
        # Determine modality based on features
        if features.get("text_score", 0.5) > 0.5:
            modality = "text-heavy"
        else:
            modality = "visual-critical"
        
        return ClassificationResult(
            modality=modality,
            confidence=confidence,
            features=features,
        )
    
    def _extract_model_features(
        self, 
        page: Page, 
        probs: "torch.Tensor"
    ) -> dict[str, float]:
        """
        Extract features from model output.
        
        For a generic pretrained model, we combine model confidence
        with heuristic features.
        """
        # Get heuristic features as baseline
        heuristic_result = self._fallback._classify_impl(page)
        features = heuristic_result.features.copy()
        
        # Add model confidence
        features["model_confidence"] = float(probs.max())
        features["model_entropy"] = float(-torch.sum(probs * torch.log(probs + 1e-10)))
        
        # Compute combined text score
        text_score = (
            0.6 * features.get("text_density", 0.5) +
            0.2 * (1.0 - features.get("image_ratio", 0.5)) +
            0.2 * features.get("white_ratio", 0.5)
        )
        features["text_score"] = text_score
        
        return features
    
    def classify_batch(self, pages: list[Page]) -> list[ClassificationResult]:
        """
        Classify multiple pages with batched inference.
        
        Args:
            pages: List of pages to classify
            
        Returns:
            List of ClassificationResults
        """
        if not self._model_loaded:
            return [self._fallback.classify(page) for page in pages]
        
        # Process in batches
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            batch_results = self._classify_batch_impl(batch)
            results.extend(batch_results)
        
        return results
    
    def _classify_batch_impl(self, pages: list[Page]) -> list[ClassificationResult]:
        """
        Classify a batch of pages.
        
        Args:
            pages: Batch of pages
            
        Returns:
            List of ClassificationResults
        """
        try:
            import torch
            
            # Preprocess all images
            images = [page.image for page in pages]
            inputs = self._processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Run batched inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Process each result
            results = []
            for idx, page in enumerate(pages):
                page_probs = probs[idx:idx+1]
                features = self._extract_model_features(page, page_probs)
                
                if features.get("text_score", 0.5) > 0.5:
                    modality = "text-heavy"
                else:
                    modality = "visual-critical"
                
                results.append(ClassificationResult(
                    modality=modality,
                    confidence=float(page_probs.max()),
                    features=features,
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"Batch ML classification failed: {e}. Using fallback.")
            return [self._fallback.classify(page) for page in pages]

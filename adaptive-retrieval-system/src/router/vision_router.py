"""
Vision-based router using Microsoft DiT for document classification.

Uses Document Image Transformer (DiT) trained on RVL-CDIP dataset
to accurately classify document pages as text-heavy or visual-critical.

Requirements: 1.1, 1.3, 1.4, 1.6
"""

import logging
from typing import Optional

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.models import Page, ClassificationResult
from src.utils.hardware import detect_device, get_device_string
from .base import BaseRouter, RouterConfig

logger = logging.getLogger(__name__)

# Router optimization: resize input images to reduce classification time
MAX_ROUTER_SIZE = 768  # DiT doesn't need full resolution (adjust 768-1024)


def _resize_for_router(img: np.ndarray) -> np.ndarray:
    """
    Resize image for router classification.
    
    DiT can make accurate predictions on smaller images since it's
    classifying document type/layout, not reading fine details.
    
    Args:
        img: Input image as numpy array (HxWxC)
        
    Returns:
        Resized image (or original if already small enough)
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    
    if long_side <= MAX_ROUTER_SIZE:
        return img
    
    # Calculate new dimensions maintaining aspect ratio
    scale = MAX_ROUTER_SIZE / long_side
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Use INTER_AREA for downscaling (best quality)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    logger.debug(f"Resized router input from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")
    return resized


class VisionRouter(BaseRouter):
    """
    Vision-based router using Microsoft DiT for document classification.
    
    Uses DiT (Document Image Transformer) fine-tuned on RVL-CDIP dataset
    to classify pages based on document layout understanding.
    
    DiT classifies into 16 document classes, which we map to 2 classes:
    - Text-heavy: letter, form, email, handwriting, memo, invoice, budget
    - Visual-critical: advertisement, presentation, scientific report, 
                      scientific publication, specification, file folder,
                      news article, questionnaire, resume
    
    Performance:
    - Accuracy: 92.69% on RVL-CDIP
    - Speed: ~50-100ms per page on CPU, <10ms on GPU
    - Model size: ~350MB
    
    Requirements: 1.1, 1.3, 1.4, 1.6
    """
    
    # DiT class mapping to our binary classification
    # RVL-CDIP has 16 classes (0-15)
    TEXT_HEAVY_CLASSES = {
        0: "letter",           # Typically text-heavy
        1: "form",             # Structured text
        2: "email",            # Text-heavy
        3: "handwriting",      # Text-heavy
        4: "invoice",          # Structured text
        6: "budget",           # Structured text/tables
        7: "memo",             # Text-heavy
    }
    
    VISUAL_CRITICAL_CLASSES = {
        5: "advertisement",           # High visual content
        8: "news_article",            # Mixed but often visual
        9: "presentation",            # Charts, diagrams
        10: "scientific_report",      # Figures, charts
        11: "scientific_publication", # Figures, equations
        12: "specification",          # Diagrams, schematics
        13: "file_folder",            # Visual structure
        14: "questionnaire",          # Forms with structure
        15: "resume",                 # Visual layout important
    }
    
    def __init__(self, config: RouterConfig = None, model_name: str = "microsoft/dit-base-finetuned-rvlcdip"):
        """
        Initialize vision router with DiT model.
        
        Args:
            config: Router configuration
            model_name: HuggingFace model identifier
        """
        super().__init__(config)
        
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        
        # Lazy load model
        logger.info(f"VisionRouter initialized (model will load on first use)")
    
    def _load_model(self):
        """Lazy load DiT model and processor."""
        if self._model is not None:
            return
        
        logger.info(f"Loading DiT model: {self.model_name}")
        
        try:
            # Detect device
            device_type = detect_device()
            self._device = get_device_string(device_type)
            logger.info(f"Using device: {self._device}")
            
            # Load processor and model
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Move to device and set to eval mode
            self._model.to(self._device)
            self._model.eval()
            
            logger.info(f"✓ DiT model loaded successfully on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load DiT model: {e}")
            raise RuntimeError(f"Failed to load DiT model: {e}")
    
    def _classify_impl(self, page: Page) -> ClassificationResult:
        """
        Classify page using DiT model.
        
        Args:
            page: Page to classify
            
        Returns:
            ClassificationResult with modality and confidence
        """
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(page.image, np.ndarray):
                # Ensure uint8
                if page.image.dtype != np.uint8:
                    image_arr = (page.image * 255).astype(np.uint8) if page.image.max() <= 1.0 else page.image.astype(np.uint8)
                else:
                    image_arr = page.image
                
                # Convert to PIL
                pil_image = Image.fromarray(image_arr)
            else:
                pil_image = page.image
            
            # OPTIMIZATION: Resize for router (DiT doesn't need full resolution)
            # Convert back to numpy for resize, then to PIL
            if isinstance(pil_image, Image.Image):
                img_array = np.array(pil_image)
                img_array = _resize_for_router(img_array)
                pil_image = Image.fromarray(img_array)
            
            # Preprocess image
            inputs = self._processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Get predicted class and probabilities
            predicted_class = logits.argmax(-1).item()
            class_prob = probs[0][predicted_class].item()
            
            # Aggregate probabilities for text-heavy vs visual-critical
            text_heavy_indices = list(self.TEXT_HEAVY_CLASSES.keys())
            visual_critical_indices = list(self.VISUAL_CRITICAL_CLASSES.keys())
            
            text_prob = probs[0][text_heavy_indices].sum().item()
            visual_prob = probs[0][visual_critical_indices].sum().item()
            
            # Determine final classification
            if predicted_class in self.TEXT_HEAVY_CLASSES:
                modality = "text-heavy"
                confidence = text_prob
                dit_class = self.TEXT_HEAVY_CLASSES[predicted_class]
            else:
                modality = "visual-critical"
                confidence = visual_prob
                dit_class = self.VISUAL_CRITICAL_CLASSES.get(predicted_class, "unknown")
            
            # Create feature dictionary for debugging
            features = {
                "dit_class": dit_class,
                "dit_class_id": predicted_class,
                "dit_class_prob": class_prob,
                "text_prob": text_prob,
                "visual_prob": visual_prob,
            }
            
            logger.debug(
                f"DiT classification: {dit_class} (class {predicted_class}) "
                f"→ {modality} (confidence: {confidence:.3f})"
            )
            
            return ClassificationResult(
                modality=modality,
                confidence=confidence,
                features=features,
            )
            
        except Exception as e:
            logger.error(f"Vision classification failed: {e}")
            raise
    
    def classify_batch(self, pages: list[Page]) -> list[ClassificationResult]:
        """
        Classify multiple pages in batch for efficiency.
        
        Args:
            pages: List of pages to classify
            
        Returns:
            List of ClassificationResults in same order
        """
        if not pages:
            return []
        
        # Ensure model is loaded
        self._load_model()
        
        logger.info(f"Classifying batch of {len(pages)} pages")
        
        try:
            # Convert all pages to PIL images
            pil_images = []
            for page in pages:
                if isinstance(page.image, np.ndarray):
                    if page.image.dtype != np.uint8:
                        image_arr = (page.image * 255).astype(np.uint8) if page.image.max() <= 1.0 else page.image.astype(np.uint8)
                    else:
                        image_arr = page.image
                    pil_image = Image.fromarray(image_arr)
                else:
                    pil_image = page.image
                
                # OPTIMIZATION: Resize for router
                img_array = np.array(pil_image) if isinstance(pil_image, Image.Image) else pil_image
                img_array = _resize_for_router(img_array)
                pil_image = Image.fromarray(img_array)
                
                pil_images.append(pil_image)
            
            # Batch preprocess
            inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Batch inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Process results
            results = []
            text_heavy_indices = list(self.TEXT_HEAVY_CLASSES.keys())
            visual_critical_indices = list(self.VISUAL_CRITICAL_CLASSES.keys())
            
            for i in range(len(pages)):
                predicted_class = logits[i].argmax(-1).item()
                class_prob = probs[i][predicted_class].item()
                
                text_prob = probs[i][text_heavy_indices].sum().item()
                visual_prob = probs[i][visual_critical_indices].sum().item()
                
                if predicted_class in self.TEXT_HEAVY_CLASSES:
                    modality = "text-heavy"
                    confidence = text_prob
                    dit_class = self.TEXT_HEAVY_CLASSES[predicted_class]
                else:
                    modality = "visual-critical"
                    confidence = visual_prob
                    dit_class = self.VISUAL_CRITICAL_CLASSES.get(predicted_class, "unknown")
                
                features = {
                    "dit_class": dit_class,
                    "dit_class_id": predicted_class,
                    "dit_class_prob": class_prob,
                    "text_prob": text_prob,
                    "visual_prob": visual_prob,
                }
                
                results.append(ClassificationResult(
                    modality=modality,
                    confidence=confidence,
                    features=features,
                ))
            
            logger.info(f"Batch classification complete: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            return [self.classify(page) for page in pages]

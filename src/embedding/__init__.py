"""
Embedding generation components for text and vision paths.

This module provides:
- Text extraction and embedding (PyMuPDF + nomic-embed-text)
- Vision embedding (ColPali/SigLIP)
- Unified embedding path interfaces

Requirements: 2.1, 3.1
"""

from .text_extractor import TextExtractor
from .text_embedder import TextEmbedder
from .text_path import TextEmbeddingPath
from .vision_embedder import VisionEmbedder
from .vision_path import VisionEmbeddingPath

__all__ = [
    "TextExtractor",
    "TextEmbedder",
    "TextEmbeddingPath",
    "VisionEmbedder",
    "VisionEmbeddingPath",
]


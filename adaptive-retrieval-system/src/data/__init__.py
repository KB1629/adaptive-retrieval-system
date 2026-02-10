"""
Data loading and preprocessing module.

This module provides dataset loaders for:
- REAL-MM-RAG (TechReport, TechSlides)
- DocVQA, InfographicVQA
- ViDoRe benchmark

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from .loader import DataLoader, DataLoaderConfig, BaseDataLoader
from .splitter import DatasetSplitter, SplitConfig
from .real_mm_rag import RealMMRAGLoader, load_techreport, load_techslides
from .docvqa import DocVQALoader, InfographicVQALoader, load_docvqa, load_infographicvqa
from .vidore import ViDoReLoader, load_vidore, load_all_vidore, get_available_subsets

__all__ = [
    # Base classes
    "DataLoader",
    "DataLoaderConfig",
    "BaseDataLoader",
    # Splitter
    "DatasetSplitter",
    "SplitConfig",
    # REAL-MM-RAG
    "RealMMRAGLoader",
    "load_techreport",
    "load_techslides",
    # DocVQA
    "DocVQALoader",
    "InfographicVQALoader",
    "load_docvqa",
    "load_infographicvqa",
    # ViDoRe
    "ViDoReLoader",
    "load_vidore",
    "load_all_vidore",
    "get_available_subsets",
]

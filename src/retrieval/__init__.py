"""
Unified retrieval interface components.

This module provides:
- Query encoding for text queries
- Unified retrieval interface
- Result formatting and ranking

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from .query_encoder import QueryEncoder
from .retriever import Retriever

__all__ = [
    "QueryEncoder",
    "Retriever",
]

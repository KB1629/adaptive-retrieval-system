"""
Vector database storage components.

This module provides:
- Vector database interface (protocol)
- Qdrant backend implementation
- LanceDB backend implementation
- Incremental indexing support

Requirements: 4.1, 4.4, 4.5, 4.6
"""

from .base import VectorDBInterface, DocumentMetadata
from .qdrant_backend import QdrantBackend
from .lancedb_backend import LanceDBBackend

__all__ = [
    "VectorDBInterface",
    "DocumentMetadata",
    "QdrantBackend",
    "LanceDBBackend",
]

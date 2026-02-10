"""
Router module for page classification.

The router classifies document pages as either:
- "text-heavy": Pages with primarily text content (80% of pages)
- "visual-critical": Pages with diagrams, schematics, charts (20% of pages)

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""

from .base import RouterInterface, RouterConfig, BaseRouter
from .heuristic import HeuristicRouter
from .ml_router import MLRouter
from .vision_router import VisionRouter

__all__ = [
    "RouterInterface",
    "RouterConfig",
    "BaseRouter",
    "HeuristicRouter",
    "MLRouter",
    "VisionRouter",
]

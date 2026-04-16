"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from pathlib import Path
from hypothesis import settings, Verbosity

# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100)
settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("thorough", max_examples=500)
settings.load_profile("default")


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_page_image() -> np.ndarray:
    """Create a sample page image for testing."""
    # Create a simple 224x224 RGB image
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_text_page_image() -> np.ndarray:
    """Create a sample text-heavy page image (mostly white with some text-like patterns)."""
    # White background with some dark regions simulating text
    image = np.ones((448, 448, 3), dtype=np.uint8) * 255
    # Add some "text" regions (dark horizontal lines)
    for y in range(50, 400, 20):
        image[y:y+10, 50:400, :] = 30
    return image


@pytest.fixture
def sample_visual_page_image() -> np.ndarray:
    """Create a sample visual-critical page image (with diagram-like patterns)."""
    image = np.ones((448, 448, 3), dtype=np.uint8) * 255
    # Add some "diagram" elements (boxes, lines)
    # Box 1
    image[50:150, 50:150, :] = [100, 150, 200]
    # Box 2
    image[200:300, 250:350, :] = [200, 100, 150]
    # Connecting line
    image[100:105, 150:250, :] = 0
    return image


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Create a sample embedding vector."""
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def temp_config(tmp_path) -> Path:
    """Create a temporary config file for testing."""
    config_content = """
router:
  type: heuristic
  text_density_threshold: 0.7
  image_area_threshold: 0.3

hardware:
  device: cpu
  max_memory_gb: 4

experiment:
  random_seed: 42
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path

"""
Configuration loading and management.

Loads YAML configuration files and provides typed access to settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class RouterConfig:
    """Router configuration."""
    type: str = "heuristic"
    text_density_threshold: float = 0.7
    image_area_threshold: float = 0.3
    ml_model_path: Optional[str] = None
    timeout_ms: int = 50


@dataclass
class TextEmbeddingConfig:
    """Text embedding path configuration."""
    model: str = "nomic-embed-text-v1.5"
    max_tokens: int = 8192
    batch_size: int = 32
    normalize: bool = True


@dataclass
class VisionEmbeddingConfig:
    """Vision embedding path configuration."""
    model: str = "vidore/colpali-v1.2"
    lora_weights: Optional[str] = None
    batch_size: int = 4
    image_size: int = 448


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    backend: str = "qdrant"
    collection_name: str = "adaptive_retrieval"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    lancedb_uri: str = "./data/lancedb"


@dataclass
class HardwareConfigSettings:
    """Hardware configuration settings."""
    device: str = "auto"
    max_memory_gb: float = 8.0
    mixed_precision: bool = True


@dataclass
class ExperimentConfigSettings:
    """Experiment configuration settings."""
    random_seed: int = 42
    checkpoint_interval: int = 100
    log_level: str = "INFO"


@dataclass
class Config:
    """Complete application configuration."""
    router: RouterConfig = field(default_factory=RouterConfig)
    text_embedding: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)
    vision_embedding: VisionEmbeddingConfig = field(default_factory=VisionEmbeddingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    hardware: HardwareConfigSettings = field(default_factory=HardwareConfigSettings)
    experiment: ExperimentConfigSettings = field(default_factory=ExperimentConfigSettings)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml.
    
    Returns:
        Parsed Config object.
    """
    if config_path is None:
        # Look for config.yaml in project root
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return Config()
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    return _parse_config(raw_config)


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw YAML dict into Config object."""
    config = Config()
    
    if "router" in raw:
        r = raw["router"]
        config.router = RouterConfig(
            type=r.get("type", "heuristic"),
            text_density_threshold=r.get("text_density_threshold", 0.7),
            image_area_threshold=r.get("image_area_threshold", 0.3),
            ml_model_path=r.get("ml_model_path"),
            timeout_ms=r.get("timeout_ms", 50),
        )
    
    if "text_embedding" in raw:
        t = raw["text_embedding"]
        config.text_embedding = TextEmbeddingConfig(
            model=t.get("model", "nomic-embed-text-v1.5"),
            max_tokens=t.get("max_tokens", 8192),
            batch_size=t.get("batch_size", 32),
            normalize=t.get("normalize", True),
        )
    
    if "vision_embedding" in raw:
        v = raw["vision_embedding"]
        config.vision_embedding = VisionEmbeddingConfig(
            model=v.get("model", "vidore/colpali-v1.2"),
            lora_weights=v.get("lora_weights"),
            batch_size=v.get("batch_size", 4),
            image_size=v.get("image_size", 448),
        )
    
    if "vector_db" in raw:
        vdb = raw["vector_db"]
        config.vector_db = VectorDBConfig(
            backend=vdb.get("backend", "qdrant"),
            collection_name=vdb.get("collection_name", "adaptive_retrieval"),
            qdrant_host=vdb.get("qdrant", {}).get("host", "localhost"),
            qdrant_port=vdb.get("qdrant", {}).get("port", 6333),
            lancedb_uri=vdb.get("lancedb", {}).get("uri", "./data/lancedb"),
        )
    
    if "hardware" in raw:
        h = raw["hardware"]
        config.hardware = HardwareConfigSettings(
            device=h.get("device", "auto"),
            max_memory_gb=h.get("max_memory_gb", 8.0),
            mixed_precision=h.get("mixed_precision", True),
        )
    
    if "experiment" in raw:
        e = raw["experiment"]
        config.experiment = ExperimentConfigSettings(
            random_seed=e.get("random_seed", 42),
            checkpoint_interval=e.get("checkpoint_interval", 100),
            log_level=e.get("log_level", "INFO"),
        )
    
    return config


def get_env_override(key: str, default: Any = None) -> Any:
    """
    Get configuration value from environment variable.
    
    Environment variables override config file values.
    Format: ARS_SECTION_KEY (e.g., ARS_HARDWARE_DEVICE)
    
    Args:
        key: Config key in format "section.key".
        default: Default value if not set.
    
    Returns:
        Environment value or default.
    """
    env_key = "ARS_" + key.upper().replace(".", "_")
    return os.environ.get(env_key, default)

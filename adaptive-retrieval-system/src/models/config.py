"""
Experiment configuration and result data structures.

These dataclasses track experiment settings and results for reproducibility:
- ExperimentConfig: Configuration for an experiment run
- ExperimentResult: Complete results from an experiment

Requirements: 10.1, 10.3
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
import json
from pathlib import Path

from .results import MetricsResult, LatencyResult


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment run.
    
    Attributes:
        experiment_id: Unique identifier for this experiment
        router_type: Type of router used
        vision_model: Vision model name
        text_model: Text embedding model name
        lora_weights_path: Path to LoRA weights (if used)
        vector_db_backend: Vector database backend
        batch_size: Batch size for processing
        random_seed: Random seed for reproducibility
        created_at: When the experiment was created
    """
    experiment_id: str
    router_type: Literal["heuristic", "ml"]
    vision_model: str
    text_model: str
    lora_weights_path: Optional[str] = None
    vector_db_backend: Literal["qdrant", "lancedb"] = "qdrant"
    batch_size: int = 4
    random_seed: int = 42
    created_at: datetime = field(default_factory=datetime.now)
    
    # Additional hyperparameters
    text_density_threshold: float = 0.7
    image_area_threshold: float = 0.3
    top_k: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "router_type": self.router_type,
            "vision_model": self.vision_model,
            "text_model": self.text_model,
            "lora_weights_path": self.lora_weights_path,
            "vector_db_backend": self.vector_db_backend,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "created_at": self.created_at.isoformat(),
            "text_density_threshold": self.text_density_threshold,
            "image_area_threshold": self.image_area_threshold,
            "top_k": self.top_k,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        return cls(
            experiment_id=data["experiment_id"],
            router_type=data["router_type"],
            vision_model=data["vision_model"],
            text_model=data["text_model"],
            lora_weights_path=data.get("lora_weights_path"),
            vector_db_backend=data.get("vector_db_backend", "qdrant"),
            batch_size=data.get("batch_size", 4),
            random_seed=data.get("random_seed", 42),
            created_at=created_at,
            text_density_threshold=data.get("text_density_threshold", 0.7),
            image_area_threshold=data.get("image_area_threshold", 0.3),
            top_k=data.get("top_k", 10),
        )
    
    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ExperimentResult:
    """
    Complete results from an experiment run.
    
    Attributes:
        config: Experiment configuration
        metrics: Retrieval metrics
        latency: Latency statistics
        throughput_pages_per_sec: Processing throughput
        router_accuracy: Router classification accuracy
        completed_at: When the experiment completed
        dataset_name: Name of the dataset used
        notes: Additional notes or observations
    """
    config: ExperimentConfig
    metrics: MetricsResult = field(default_factory=MetricsResult)
    latency: LatencyResult = field(default_factory=LatencyResult)
    throughput_pages_per_sec: float = 0.0
    router_accuracy: float = 0.0
    completed_at: Optional[datetime] = None
    dataset_name: str = ""
    notes: str = ""
    
    def __post_init__(self):
        """Set completion time if not provided."""
        if self.completed_at is None:
            self.completed_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "latency": self.latency.to_dict(),
            "throughput_pages_per_sec": self.throughput_pages_per_sec,
            "router_accuracy": self.router_accuracy,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dataset_name": self.dataset_name,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Create from dictionary."""
        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        
        return cls(
            config=ExperimentConfig.from_dict(data["config"]),
            metrics=MetricsResult.from_dict(data.get("metrics", {})),
            latency=LatencyResult.from_dict(data.get("latency", {})),
            throughput_pages_per_sec=data.get("throughput_pages_per_sec", 0.0),
            router_accuracy=data.get("router_accuracy", 0.0),
            completed_at=completed_at,
            dataset_name=data.get("dataset_name", ""),
            notes=data.get("notes", ""),
        )
    
    def to_json(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "ExperimentResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_latex_table(self) -> str:
        """Generate LaTeX table for publication."""
        return f"""
\\begin{{table}}[h]
\\centering
\\caption{{Experiment Results: {self.config.experiment_id}}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Recall@1 & {self.metrics.recall_at_1:.3f} \\\\
Recall@5 & {self.metrics.recall_at_5:.3f} \\\\
Recall@10 & {self.metrics.recall_at_10:.3f} \\\\
MRR & {self.metrics.mrr:.3f} \\\\
NDCG & {self.metrics.ndcg:.3f} \\\\
\\midrule
Mean Latency (ms) & {self.latency.mean_ms:.1f} \\\\
P95 Latency (ms) & {self.latency.p95_ms:.1f} \\\\
Throughput (pages/s) & {self.throughput_pages_per_sec:.1f} \\\\
Router Accuracy & {self.router_accuracy:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:{self.config.experiment_id}}}
\\end{{table}}
"""
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Experiment: {self.config.experiment_id}
Dataset: {self.dataset_name}
Router: {self.config.router_type}
Vision Model: {self.config.vision_model}

Metrics:
  Recall@10: {self.metrics.recall_at_10:.3f}
  MRR: {self.metrics.mrr:.3f}
  NDCG: {self.metrics.ndcg:.3f}

Performance:
  Mean Latency: {self.latency.mean_ms:.1f} ms
  P95 Latency: {self.latency.p95_ms:.1f} ms
  Throughput: {self.throughput_pages_per_sec:.1f} pages/sec
  Router Accuracy: {self.router_accuracy:.3f}

Completed: {self.completed_at}
"""

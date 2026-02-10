"""Experiment configuration tracking and persistence."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.config import ExperimentConfig, ExperimentResult


class ExperimentTracker:
    """Tracks experiment configurations and results."""

    def __init__(self, output_dir: str = "outputs/experiments"):
        """Initialize experiment tracker.

        Args:
            output_dir: Directory to save experiment data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        router_type: str,
        vision_model: str,
        text_model: str,
        vector_db_backend: str,
        batch_size: int = 32,
        random_seed: int = 42,
        lora_weights_path: str | None = None,
        **kwargs: Any,
    ) -> ExperimentConfig:
        """Create a new experiment configuration.

        Args:
            router_type: Type of router ("heuristic" or "ml")
            vision_model: Vision model name
            text_model: Text model name
            vector_db_backend: Vector DB backend ("qdrant" or "lancedb")
            batch_size: Batch size for processing
            random_seed: Random seed for reproducibility
            lora_weights_path: Optional path to LoRA weights
            **kwargs: Additional configuration parameters

        Returns:
            ExperimentConfig object
        """
        experiment_id = self._generate_experiment_id()

        config = ExperimentConfig(
            experiment_id=experiment_id,
            router_type=router_type,
            vision_model=vision_model,
            text_model=text_model,
            lora_weights_path=lora_weights_path,
            vector_db_backend=vector_db_backend,
            batch_size=batch_size,
            random_seed=random_seed,
            created_at=datetime.now(),
        )

        # Save configuration
        self.save_config(config)

        return config

    def save_config(self, config: ExperimentConfig) -> Path:
        """Save experiment configuration to JSON.

        Args:
            config: Experiment configuration

        Returns:
            Path to saved configuration file
        """
        config_path = self.output_dir / f"{config.experiment_id}_config.json"

        config_dict = {
            "experiment_id": config.experiment_id,
            "router_type": config.router_type,
            "vision_model": config.vision_model,
            "text_model": config.text_model,
            "lora_weights_path": config.lora_weights_path,
            "vector_db_backend": config.vector_db_backend,
            "batch_size": config.batch_size,
            "random_seed": config.random_seed,
            "created_at": config.created_at.isoformat(),
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return config_path

    def load_config(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment configuration from JSON.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentConfig object

        Raises:
            FileNotFoundError: If configuration file not found
        """
        config_path = self.output_dir / f"{experiment_id}_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        config_dict["created_at"] = datetime.fromisoformat(config_dict["created_at"])

        return ExperimentConfig(**config_dict)

    def save_result(self, result: ExperimentResult) -> Path:
        """Save experiment result to JSON.

        Args:
            result: Experiment result

        Returns:
            Path to saved result file
        """
        result_path = self.output_dir / f"{result.config.experiment_id}_result.json"

        result_dict = {
            "config": {
                "experiment_id": result.config.experiment_id,
                "router_type": result.config.router_type,
                "vision_model": result.config.vision_model,
                "text_model": result.config.text_model,
                "lora_weights_path": result.config.lora_weights_path,
                "vector_db_backend": result.config.vector_db_backend,
                "batch_size": result.config.batch_size,
                "random_seed": result.config.random_seed,
                "created_at": result.config.created_at.isoformat(),
            },
            "metrics": {
                "recall_at_1": result.metrics.recall_at_1,
                "recall_at_5": result.metrics.recall_at_5,
                "recall_at_10": result.metrics.recall_at_10,
                "mrr": result.metrics.mrr,
                "ndcg": result.metrics.ndcg,
            },
            "latency": {
                "mean_ms": result.latency.mean_ms,
                "median_ms": result.latency.median_ms,
                "p95_ms": result.latency.p95_ms,
                "std_ms": result.latency.std_ms,
            },
            "throughput_pages_per_sec": result.throughput_pages_per_sec,
            "router_accuracy": result.router_accuracy,
            "completed_at": result.completed_at.isoformat(),
        }

        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        return result_path

    def load_result(self, experiment_id: str) -> ExperimentResult:
        """Load experiment result from JSON.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentResult object

        Raises:
            FileNotFoundError: If result file not found
        """
        result_path = self.output_dir / f"{experiment_id}_result.json"

        if not result_path.exists():
            raise FileNotFoundError(f"Result not found: {result_path}")

        with open(result_path) as f:
            result_dict = json.load(f)

        # Reconstruct objects
        from src.models.results import (
            MetricsResult,
            LatencyResult,
        )

        config = ExperimentConfig(
            experiment_id=result_dict["config"]["experiment_id"],
            router_type=result_dict["config"]["router_type"],
            vision_model=result_dict["config"]["vision_model"],
            text_model=result_dict["config"]["text_model"],
            lora_weights_path=result_dict["config"]["lora_weights_path"],
            vector_db_backend=result_dict["config"]["vector_db_backend"],
            batch_size=result_dict["config"]["batch_size"],
            random_seed=result_dict["config"]["random_seed"],
            created_at=datetime.fromisoformat(result_dict["config"]["created_at"]),
        )

        metrics = MetricsResult(**result_dict["metrics"])
        latency = LatencyResult(**result_dict["latency"])

        return ExperimentResult(
            config=config,
            metrics=metrics,
            latency=latency,
            throughput_pages_per_sec=result_dict["throughput_pages_per_sec"],
            router_accuracy=result_dict["router_accuracy"],
            completed_at=datetime.fromisoformat(result_dict["completed_at"]),
        )

    def list_experiments(self) -> list[str]:
        """List all experiment IDs.

        Returns:
            List of experiment IDs
        """
        config_files = self.output_dir.glob("*_config.json")
        return [f.stem.replace("_config", "") for f in config_files]

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID.

        Returns:
            Experiment ID in format: exp_YYYYMMDD_HHMMSS
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_experiment_config(config: ExperimentConfig, output_path: str) -> None:
    """Save experiment configuration to file.

    Args:
        config: Experiment configuration
        output_path: Path to save configuration
    """
    tracker = ExperimentTracker(output_dir=str(Path(output_path).parent))
    tracker.save_config(config)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        ExperimentConfig object
    """
    experiment_id = Path(config_path).stem.replace("_config", "")
    tracker = ExperimentTracker(output_dir=str(Path(config_path).parent))
    return tracker.load_config(experiment_id)

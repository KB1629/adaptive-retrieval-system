"""
Benchmark runner for full evaluation.

This module orchestrates complete benchmark runs:
- Evaluation on multiple datasets
- Baseline comparison (ColPali, HPC-ColPali)
- Result aggregation and reporting

Requirements: 6.4, 6.5
"""

import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..models.data import BenchmarkDataset
from ..models.results import MetricsResult, LatencyResult
from .metrics import evaluate_retrieval
from .latency import measure_component_latency, LatencyProfiler
from .throughput import measure_pipeline_throughput, ThroughputResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark run.
    
    Attributes:
        name: Benchmark name
        datasets: List of datasets to evaluate on
        k_values: K values for Recall@K
        measure_latency: Whether to measure component latency
        measure_throughput: Whether to measure throughput
        baseline_name: Name of baseline for comparison
    """
    name: str
    datasets: List[str] = field(default_factory=list)
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    measure_latency: bool = True
    measure_throughput: bool = True
    baseline_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "datasets": self.datasets,
            "k_values": self.k_values,
            "measure_latency": self.measure_latency,
            "measure_throughput": self.measure_throughput,
            "baseline_name": self.baseline_name,
        }


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result.
    
    Attributes:
        config: Benchmark configuration
        metrics: Retrieval metrics
        latency: Latency measurements (optional)
        throughput: Throughput measurements (optional)
        dataset_results: Per-dataset results
        timestamp: When benchmark was run
        notes: Additional notes
    """
    config: BenchmarkConfig
    metrics: MetricsResult
    latency: Optional[Dict[str, LatencyResult]] = None
    throughput: Optional[ThroughputResult] = None
    dataset_results: Dict[str, MetricsResult] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }
        
        if self.latency:
            result["latency"] = {
                name: lat.to_dict() for name, lat in self.latency.items()
            }
        
        if self.throughput:
            result["throughput"] = self.throughput.to_dict()
        
        if self.dataset_results:
            result["dataset_results"] = {
                name: metrics.to_dict() 
                for name, metrics in self.dataset_results.items()
            }
        
        return result
    
    def save(self, filepath: str):
        """Save result to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark result to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "BenchmarkResult":
        """Load result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects (simplified - would need full reconstruction)
        config = BenchmarkConfig(
            name=data["config"]["name"],
            datasets=data["config"]["datasets"],
            k_values=data["config"]["k_values"],
        )
        
        metrics = MetricsResult.from_dict(data["metrics"])
        
        return cls(
            config=config,
            metrics=metrics,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            notes=data.get("notes", ""),
        )


class BenchmarkRunner:
    """
    Runner for executing benchmark evaluations.
    
    Example:
        >>> runner = BenchmarkRunner(retrieval_system)
        >>> config = BenchmarkConfig(name="baseline", datasets=["REAL-MM-RAG"])
        >>> result = runner.run(config, dataset_loader)
        >>> print(result.metrics.recall_at_5)
    """
    
    def __init__(
        self,
        retrieval_func: Callable[[str], List[str]],
        name: str = "adaptive-retrieval",
    ):
        """
        Initialize benchmark runner.
        
        Args:
            retrieval_func: Function that takes a query and returns doc IDs
            name: Name of the system being benchmarked
        """
        self.retrieval_func = retrieval_func
        self.name = name
        self.profiler = LatencyProfiler()
    
    def run(
        self,
        config: BenchmarkConfig,
        datasets: Dict[str, BenchmarkDataset],
    ) -> BenchmarkResult:
        """
        Run complete benchmark evaluation.
        
        Args:
            config: Benchmark configuration
            datasets: Dictionary mapping dataset names to BenchmarkDataset objects
            
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Starting benchmark: {config.name}")
        logger.info(f"Datasets: {config.datasets}")
        
        # Collect predictions and ground truth across all datasets
        all_predictions = []
        all_ground_truth = []
        dataset_results = {}
        
        for dataset_name in config.datasets:
            if dataset_name not in datasets:
                logger.warning(f"Dataset {dataset_name} not found, skipping")
                continue
            
            dataset = datasets[dataset_name]
            logger.info(f"Evaluating on {dataset_name} ({len(dataset.queries)} queries)")
            
            # Run queries
            predictions = []
            ground_truth = []
            
            for query in dataset.queries:
                with self.profiler.profile("query"):
                    pred_docs = self.retrieval_func(query)
                    predictions.append(pred_docs)
                
                # Get ground truth for this query
                gt_docs = dataset.ground_truth.get(query, set())
                ground_truth.append(gt_docs)
            
            # Evaluate this dataset
            dataset_metrics = evaluate_retrieval(
                predictions,
                ground_truth,
                k_values=config.k_values,
            )
            dataset_results[dataset_name] = dataset_metrics
            
            # Add to overall results
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
            
            logger.info(
                f"{dataset_name} results: R@1={dataset_metrics.recall_at_1:.3f}, "
                f"R@5={dataset_metrics.recall_at_5:.3f}, MRR={dataset_metrics.mrr:.3f}"
            )
        
        # Compute overall metrics
        overall_metrics = evaluate_retrieval(
            all_predictions,
            all_ground_truth,
            k_values=config.k_values,
        )
        
        # Get latency statistics
        latency_stats = None
        if config.measure_latency:
            latency_stats = self.profiler.get_statistics()
            logger.info("Latency measurements:")
            for component, stats in latency_stats.items():
                logger.info(f"  {component}: {stats.mean_ms:.2f}ms (median: {stats.median_ms:.2f}ms)")
        
        # Create result
        result = BenchmarkResult(
            config=config,
            metrics=overall_metrics,
            latency=latency_stats,
            dataset_results=dataset_results,
            notes=f"Evaluated on {len(all_predictions)} queries across {len(config.datasets)} datasets",
        )
        
        logger.info(f"Benchmark complete: {config.name}")
        logger.info(f"Overall: R@1={overall_metrics.recall_at_1:.3f}, R@5={overall_metrics.recall_at_5:.3f}, "
                   f"R@10={overall_metrics.recall_at_10:.3f}, MRR={overall_metrics.mrr:.3f}, NDCG={overall_metrics.ndcg:.3f}")
        
        return result
    
    def compare_with_baseline(
        self,
        result: BenchmarkResult,
        baseline_result: BenchmarkResult,
    ) -> Dict[str, float]:
        """
        Compare results with baseline.
        
        Args:
            result: Current benchmark result
            baseline_result: Baseline benchmark result
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "recall_at_1_diff": result.metrics.recall_at_1 - baseline_result.metrics.recall_at_1,
            "recall_at_5_diff": result.metrics.recall_at_5 - baseline_result.metrics.recall_at_5,
            "recall_at_10_diff": result.metrics.recall_at_10 - baseline_result.metrics.recall_at_10,
            "mrr_diff": result.metrics.mrr - baseline_result.metrics.mrr,
            "ndcg_diff": result.metrics.ndcg - baseline_result.metrics.ndcg,
        }
        
        # Latency comparison
        if result.latency and baseline_result.latency:
            for component in result.latency:
                if component in baseline_result.latency:
                    speedup = (baseline_result.latency[component].mean_ms / 
                              result.latency[component].mean_ms)
                    comparison[f"{component}_speedup"] = speedup
        
        logger.info("Comparison with baseline:")
        logger.info(f"  Recall@1: {comparison['recall_at_1_diff']:+.3f}")
        logger.info(f"  Recall@5: {comparison['recall_at_5_diff']:+.3f}")
        logger.info(f"  Recall@10: {comparison['recall_at_10_diff']:+.3f}")
        logger.info(f"  MRR: {comparison['mrr_diff']:+.3f}")
        logger.info(f"  NDCG: {comparison['ndcg_diff']:+.3f}")
        
        return comparison
    
    def print_summary(self, result: BenchmarkResult):
        """
        Print human-readable summary of results.
        
        Args:
            result: Benchmark result to summarize
        """
        print("\n" + "="*70)
        print(f"Benchmark Results: {result.config.name}")
        print("="*70)
        print(f"System: {self.name}")
        print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Datasets: {', '.join(result.config.datasets)}")
        print()
        
        print("Overall Metrics:")
        print(f"  Recall@1:  {result.metrics.recall_at_1:.4f}")
        print(f"  Recall@5:  {result.metrics.recall_at_5:.4f}")
        print(f"  Recall@10: {result.metrics.recall_at_10:.4f}")
        print(f"  MRR:       {result.metrics.mrr:.4f}")
        print(f"  NDCG:      {result.metrics.ndcg:.4f}")
        print()
        
        if result.dataset_results:
            print("Per-Dataset Results:")
            for dataset_name, metrics in result.dataset_results.items():
                print(f"  {dataset_name}:")
                print(f"    R@1={metrics.recall_at_1:.4f}, R@5={metrics.recall_at_5:.4f}, "
                      f"R@10={metrics.recall_at_10:.4f}, MRR={metrics.mrr:.4f}")
            print()
        
        if result.latency:
            print("Latency (mean):")
            for component, stats in result.latency.items():
                print(f"  {component}: {stats.mean_ms:.2f}ms")
            print()
        
        if result.throughput:
            print(f"Throughput: {result.throughput.pages_per_second:.2f} pages/sec")
            print()
        
        if result.notes:
            print(f"Notes: {result.notes}")
        
        print("="*70 + "\n")

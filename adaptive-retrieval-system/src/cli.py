"""Command-line interface for Adaptive Retrieval System."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.pipeline.orchestrator import AdaptiveRetrievalPipeline
from src.benchmark.runner import BenchmarkRunner
from src.experiment.tracker import ExperimentTracker, set_random_seed
from src.utils.logging import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def cmd_index(args: argparse.Namespace) -> None:
    """Index documents command.

    Args:
        args: Command arguments
    """
    config = load_config(args.config)

    # Initialize pipeline
    pipeline = AdaptiveRetrievalPipeline(
        router_type=config["router"]["type"],
        vector_db_backend=config["vector_db"]["backend"],
        vector_db_config=config["vector_db"],
        text_model=config["text_embedding"]["model"],
        vision_model=config["vision_embedding"]["model"],
        lora_weights_path=config["vision_embedding"].get("lora_weights"),
        device=config["hardware"]["device"],
    )

    # TODO: Load documents from args.input_path
    # For now, just print message
    print(f"Indexing documents from: {args.input_path}")
    print(f"Using router: {config['router']['type']}")
    print(f"Vector DB: {config['vector_db']['backend']}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query command.

    Args:
        args: Command arguments
    """
    config = load_config(args.config)

    # Initialize pipeline
    pipeline = AdaptiveRetrievalPipeline(
        router_type=config["router"]["type"],
        vector_db_backend=config["vector_db"]["backend"],
        vector_db_config=config["vector_db"],
        text_model=config["text_embedding"]["model"],
        vision_model=config["vision_embedding"]["model"],
        device=config["hardware"]["device"],
    )

    # Execute query
    results = pipeline.query(args.query, top_k=args.top_k)

    # Display results
    print(f"\nQuery: {args.query}")
    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result.doc_id}, Page: {result.page_number}")
        print(f"   Score: {result.relevance_score:.4f}, Modality: {result.modality}")
        print()


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark command.

    Args:
        args: Command arguments
    """
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_random_seed(config["experiment"]["random_seed"])

    # Initialize pipeline
    pipeline = AdaptiveRetrievalPipeline(
        router_type=config["router"]["type"],
        vector_db_backend=config["vector_db"]["backend"],
        vector_db_config=config["vector_db"],
        text_model=config["text_embedding"]["model"],
        vision_model=config["vision_embedding"]["model"],
        lora_weights_path=config["vision_embedding"].get("lora_weights"),
        device=config["hardware"]["device"],
    )

    # Initialize benchmark runner
    runner = BenchmarkRunner(pipeline=pipeline)

    # TODO: Load dataset and run benchmark
    print(f"Running benchmark on dataset: {args.dataset}")
    print(f"Random seed: {config['experiment']['random_seed']}")


def cmd_experiment(args: argparse.Namespace) -> None:
    """Experiment tracking command.

    Args:
        args: Command arguments
    """
    tracker = ExperimentTracker(output_dir=args.output_dir)

    if args.action == "list":
        experiments = tracker.list_experiments()
        print(f"Found {len(experiments)} experiments:")
        for exp_id in experiments:
            print(f"  - {exp_id}")

    elif args.action == "show":
        result = tracker.load_result(args.experiment_id)
        print(f"\nExperiment: {result.config.experiment_id}")
        print(f"Router: {result.config.router_type}")
        print(f"Vision Model: {result.config.vision_model}")
        print(f"\nMetrics:")
        print(f"  Recall@1:  {result.metrics.recall_at_1:.3f}")
        print(f"  Recall@5:  {result.metrics.recall_at_5:.3f}")
        print(f"  Recall@10: {result.metrics.recall_at_10:.3f}")
        print(f"  MRR:       {result.metrics.mrr:.3f}")
        print(f"  NDCG:      {result.metrics.ndcg:.3f}")
        print(f"\nLatency:")
        print(f"  Mean:   {result.latency.mean_ms:.1f} ms")
        print(f"  Median: {result.latency.median_ms:.1f} ms")
        print(f"  P95:    {result.latency.p95_ms:.1f} ms")
        print(f"\nThroughput: {result.throughput_pages_per_sec:.2f} pages/sec")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive Retrieval System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "input_path",
        type=str,
        help="Path to documents to index",
    )
    index_parser.set_defaults(func=cmd_index)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument(
        "query",
        type=str,
        help="Query text",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return",
    )
    query_parser.set_defaults(func=cmd_query)

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    benchmark_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["real-mm-rag", "docvqa", "vidore"],
        help="Dataset to benchmark on",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    # Experiment command
    experiment_parser = subparsers.add_parser(
        "experiment", help="Experiment tracking"
    )
    experiment_parser.add_argument(
        "action",
        type=str,
        choices=["list", "show"],
        help="Action to perform",
    )
    experiment_parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment ID (for show action)",
    )
    experiment_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Output directory for experiments",
    )
    experiment_parser.set_defaults(func=cmd_experiment)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

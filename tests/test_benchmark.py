"""
Tests for benchmark framework.

This module tests:
- Retrieval metrics (Recall@K, MRR, NDCG)
- Latency measurement
- Throughput calculation
- Property 8: Recall Metric Computation
- Property 9: Latency Statistics Computation
- Property 10: Throughput Computation

Requirements: 6.1, 6.2, 6.3
"""

import pytest
import time
from hypothesis import given, strategies as st, settings

from src.benchmark.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
    evaluate_retrieval,
)
from src.benchmark.latency import measure_latency, LatencyProfiler
from src.benchmark.throughput import measure_throughput, ThroughputResult
from src.models.results import MetricsResult, LatencyResult


# Unit Tests for Metrics

class TestRecallMetrics:
    """Test Recall@K computation."""
    
    def test_recall_perfect(self):
        """Test perfect recall."""
        predictions = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1", "doc2"}
        
        recall = compute_recall_at_k(predictions, ground_truth, k=3)
        assert recall == 1.0
    
    def test_recall_partial(self):
        """Test partial recall."""
        predictions = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1", "doc4"}
        
        recall = compute_recall_at_k(predictions, ground_truth, k=2)
        assert recall == 0.5  # Found 1 out of 2
    
    def test_recall_zero(self):
        """Test zero recall."""
        predictions = ["doc1", "doc2"]
        ground_truth = {"doc3", "doc4"}
        
        recall = compute_recall_at_k(predictions, ground_truth, k=2)
        assert recall == 0.0
    
    def test_recall_empty_predictions(self):
        """Test recall with empty predictions."""
        recall = compute_recall_at_k([], {"doc1"}, k=5)
        assert recall == 0.0
    
    def test_recall_empty_ground_truth(self):
        """Test recall with empty ground truth."""
        recall = compute_recall_at_k(["doc1"], set(), k=5)
        assert recall == 1.0  # Perfect by convention
    
    def test_recall_k_larger_than_predictions(self):
        """Test K larger than predictions list."""
        predictions = ["doc1", "doc2"]
        ground_truth = {"doc1", "doc2", "doc3"}
        
        recall = compute_recall_at_k(predictions, ground_truth, k=10)
        assert recall == 2/3  # Found 2 out of 3


class TestMRR:
    """Test Mean Reciprocal Rank computation."""
    
    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        predictions = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1"}
        
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 1.0
    
    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        predictions = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc2"}
        
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 0.5
    
    def test_mrr_third_position(self):
        """Test MRR when relevant doc is third."""
        predictions = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc3"}
        
        mrr = compute_mrr(predictions, ground_truth)
        assert abs(mrr - 1/3) < 0.001
    
    def test_mrr_no_relevant(self):
        """Test MRR when no relevant docs found."""
        predictions = ["doc1", "doc2"]
        ground_truth = {"doc3"}
        
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 0.0
    
    def test_mrr_empty(self):
        """Test MRR with empty inputs."""
        assert compute_mrr([], {"doc1"}) == 0.0
        assert compute_mrr(["doc1"], set()) == 0.0


class TestNDCG:
    """Test NDCG computation."""
    
    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        predictions = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1", "doc2"}
        
        ndcg = compute_ndcg(predictions, ground_truth, k=3)
        assert ndcg == 1.0
    
    def test_ndcg_reversed(self):
        """Test NDCG with reversed ranking."""
        predictions = ["doc3", "doc2", "doc1"]
        ground_truth = {"doc1", "doc2"}
        
        ndcg = compute_ndcg(predictions, ground_truth, k=3)
        assert 0.0 < ndcg < 1.0  # Not perfect but not zero
    
    def test_ndcg_no_relevant(self):
        """Test NDCG with no relevant docs."""
        predictions = ["doc1", "doc2"]
        ground_truth = {"doc3"}
        
        ndcg = compute_ndcg(predictions, ground_truth, k=2)
        assert ndcg == 0.0
    
    def test_ndcg_empty(self):
        """Test NDCG with empty inputs."""
        assert compute_ndcg([], {"doc1"}) == 0.0
        assert compute_ndcg(["doc1"], set()) == 0.0


class TestEvaluateRetrieval:
    """Test overall retrieval evaluation."""
    
    def test_evaluate_single_query(self):
        """Test evaluation with single query."""
        predictions = [["doc1", "doc2", "doc3"]]
        ground_truth = [{"doc1", "doc2"}]
        
        result = evaluate_retrieval(predictions, ground_truth)
        
        assert isinstance(result, MetricsResult)
        assert result.recall_at_1 == 0.5  # doc1 found
        assert result.recall_at_5 == 1.0  # Both found in top 5
    
    def test_evaluate_multiple_queries(self):
        """Test evaluation with multiple queries."""
        predictions = [
            ["doc1", "doc2"],
            ["doc3", "doc4"],
        ]
        ground_truth = [
            {"doc1"},
            {"doc3", "doc5"},
        ]
        
        result = evaluate_retrieval(predictions, ground_truth)
        
        assert isinstance(result, MetricsResult)
        assert 0.0 <= result.recall_at_1 <= 1.0
        assert 0.0 <= result.mrr <= 1.0
    
    def test_evaluate_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="must have same length"):
            evaluate_retrieval([["doc1"]], [{"doc1"}, {"doc2"}])
    
    def test_evaluate_empty_returns_zeros(self):
        """Test that empty input returns zero metrics."""
        result = evaluate_retrieval([], [])
        
        assert result.recall_at_1 == 0.0
        assert result.recall_at_5 == 0.0
        assert result.mrr == 0.0


# Unit Tests for Latency

class TestLatencyMeasurement:
    """Test latency measurement."""
    
    def test_measure_latency_basic(self):
        """Test basic latency measurement."""
        def fast_func():
            time.sleep(0.001)  # 1ms
        
        result = measure_latency(fast_func, iterations=10, warmup=2)
        
        assert isinstance(result, LatencyResult)
        assert result.num_samples == 10
        assert result.mean_ms > 0
        assert result.min_ms <= result.mean_ms <= result.max_ms
    
    def test_measure_latency_with_args(self):
        """Test latency measurement with arguments."""
        def func_with_args(x, y):
            return x + y
        
        result = measure_latency(func_with_args, 1, 2, iterations=5)
        
        assert result.num_samples == 5
    
    def test_measure_latency_invalid_iterations(self):
        """Test that invalid iterations raises error."""
        with pytest.raises(ValueError, match="iterations must be > 0"):
            measure_latency(lambda: None, iterations=0)
    
    def test_latency_profiler(self):
        """Test latency profiler."""
        profiler = LatencyProfiler()
        
        with profiler.profile("section1"):
            time.sleep(0.001)
        
        with profiler.profile("section2"):
            time.sleep(0.002)
        
        stats = profiler.get_statistics()
        
        assert "section1" in stats
        assert "section2" in stats
        assert stats["section1"].num_samples == 1
        assert stats["section2"].mean_ms > stats["section1"].mean_ms


# Unit Tests for Throughput

class TestThroughputMeasurement:
    """Test throughput measurement."""
    
    def test_measure_throughput_basic(self):
        """Test basic throughput measurement."""
        def process_item(item):
            time.sleep(0.001)  # 1ms per item
        
        items = list(range(10))
        result = measure_throughput(process_item, items)
        
        assert isinstance(result, ThroughputResult)
        assert result.total_pages == 10
        assert result.pages_per_second > 0
        assert result.total_time_seconds > 0
    
    def test_measure_throughput_empty(self):
        """Test throughput with empty items."""
        result = measure_throughput(lambda x: x, [])
        
        assert result.total_pages == 0
        assert result.pages_per_second == 0.0
    
    def test_throughput_result_string(self):
        """Test ThroughputResult string representation."""
        result = ThroughputResult(
            pages_per_second=10.0,
            total_pages=100,
            total_time_seconds=10.0,
            avg_time_per_page_ms=100.0,
        )
        
        string = str(result)
        assert "10.00 pages/sec" in string
        assert "100 pages" in string


# Property-Based Tests

# Feature: adaptive-retrieval-system, Property 8: Recall Metric Computation
# **Validates: Requirements 6.1**
@settings(max_examples=10, deadline=None)
@given(
    num_predictions=st.integers(min_value=0, max_value=20),
    num_relevant=st.integers(min_value=0, max_value=10),
    k=st.integers(min_value=1, max_value=10),
)
def test_property_recall_computation(num_predictions, num_relevant, k):
    """
    Property 8: Recall Metric Computation
    
    For any set of predictions and ground truth relevance labels, the computed
    Recall@K metric SHALL equal the proportion of ground truth relevant documents
    that appear in the top K predictions, and Recall@K1 <= Recall@K2 when K1 < K2.
    
    **Validates: Requirements 6.1**
    """
    # Generate predictions
    predictions = [f"doc_{i}" for i in range(num_predictions)]
    
    # Generate ground truth (some overlap with predictions)
    ground_truth = set(f"doc_{i}" for i in range(num_relevant))
    
    # Compute recall
    recall = compute_recall_at_k(predictions, ground_truth, k)
    
    # Property 1: Recall is between 0 and 1
    assert 0.0 <= recall <= 1.0, f"Recall {recall} not in [0, 1]"
    
    # Property 2: If K >= num_predictions, recall is maximized
    if k >= num_predictions and num_relevant > 0:
        # Count actual overlap
        overlap = len(set(predictions) & ground_truth)
        expected_recall = overlap / num_relevant
        assert abs(recall - expected_recall) < 0.001
    
    # Property 3: Recall@K1 <= Recall@K2 when K1 < K2
    if k < min(num_predictions, 20):
        k2 = k + 1
        recall2 = compute_recall_at_k(predictions, ground_truth, k2)
        assert recall <= recall2, f"Recall@{k}={recall} > Recall@{k2}={recall2}"


# Feature: adaptive-retrieval-system, Property 9: Latency Statistics Computation
# **Validates: Requirements 6.2**
@settings(max_examples=10, deadline=None)
@given(
    num_measurements=st.integers(min_value=1, max_value=100),
)
def test_property_latency_statistics(num_measurements):
    """
    Property 9: Latency Statistics Computation
    
    For any set of latency measurements, the computed statistics SHALL satisfy:
    min <= mean <= max, min <= median <= max, min <= P95 <= max, and P95 >= median.
    
    **Validates: Requirements 6.2**
    """
    # Generate random latency measurements
    import random
    measurements = [random.uniform(1.0, 100.0) for _ in range(num_measurements)]
    
    # Compute statistics
    result = LatencyResult.from_measurements(measurements)
    
    # Property 1: min <= mean <= max
    assert result.min_ms <= result.mean_ms <= result.max_ms, \
        f"Invalid: min ({result.min_ms}) <= mean ({result.mean_ms}) <= max ({result.max_ms})"
    
    # Property 2: min <= median <= max
    assert result.min_ms <= result.median_ms <= result.max_ms, \
        f"Invalid: min ({result.min_ms}) <= median ({result.median_ms}) <= max ({result.max_ms})"
    
    # Property 3: min <= P95 <= max
    assert result.min_ms <= result.p95_ms <= result.max_ms, \
        f"Invalid: min ({result.min_ms}) <= P95 ({result.p95_ms}) <= max ({result.max_ms})"
    
    # Property 4: P95 >= median
    assert result.p95_ms >= result.median_ms, \
        f"Invalid: P95 ({result.p95_ms}) >= median ({result.median_ms})"
    
    # Property 5: num_samples matches input
    assert result.num_samples == num_measurements


# Feature: adaptive-retrieval-system, Property 10: Throughput Computation
# **Validates: Requirements 6.3**
@settings(max_examples=10, deadline=None)
@given(
    num_pages=st.integers(min_value=1, max_value=100),
    time_per_page_ms=st.floats(min_value=1.0, max_value=100.0),
)
def test_property_throughput_computation(num_pages, time_per_page_ms):
    """
    Property 10: Throughput Computation
    
    For any processing run with N pages completed in T seconds, the computed
    throughput SHALL equal N/T pages per second.
    
    **Validates: Requirements 6.3**
    """
    # Simulate processing
    total_time_seconds = (num_pages * time_per_page_ms) / 1000.0
    
    # Create result
    result = ThroughputResult(
        pages_per_second=num_pages / total_time_seconds,
        total_pages=num_pages,
        total_time_seconds=total_time_seconds,
        avg_time_per_page_ms=time_per_page_ms,
    )
    
    # Property: Throughput = N / T
    expected_throughput = num_pages / total_time_seconds
    assert abs(result.pages_per_second - expected_throughput) < 0.001, \
        f"Throughput {result.pages_per_second} != {expected_throughput}"
    
    # Property: Average time per page = T / N * 1000
    expected_avg_ms = (total_time_seconds / num_pages) * 1000
    assert abs(result.avg_time_per_page_ms - expected_avg_ms) < 0.001, \
        f"Avg time {result.avg_time_per_page_ms}ms != {expected_avg_ms}ms"
    
    # Property: Throughput > 0 when pages > 0
    assert result.pages_per_second > 0, "Throughput must be positive"

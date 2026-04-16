"""
Retrieval evaluation metrics.

This module implements standard information retrieval metrics:
- Recall@K: Proportion of relevant documents in top K results
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
- NDCG (Normalized Discounted Cumulative Gain): Ranking quality metric

Requirements: 6.1
"""

import logging
from typing import List, Set
import numpy as np

from ..models.results import MetricsResult

logger = logging.getLogger(__name__)


def compute_recall_at_k(
    predictions: List[str],
    ground_truth: Set[str],
    k: int,
) -> float:
    """
    Compute Recall@K metric.
    
    Recall@K measures the proportion of relevant documents that appear
    in the top K predictions.
    
    Args:
        predictions: List of predicted document IDs (ordered by relevance)
        ground_truth: Set of relevant document IDs
        k: Number of top predictions to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
        
    Examples:
        >>> compute_recall_at_k(["doc1", "doc2", "doc3"], {"doc1", "doc4"}, k=2)
        0.5  # Found 1 out of 2 relevant docs in top 2
        
        >>> compute_recall_at_k(["doc1", "doc2"], {"doc1", "doc2"}, k=2)
        1.0  # Found all relevant docs
    """
    if not ground_truth:
        # No relevant documents - perfect recall by convention
        return 1.0
    
    if not predictions:
        # No predictions - zero recall
        return 0.0
    
    # Consider only top K predictions
    top_k = predictions[:k]
    
    # Count how many relevant docs are in top K
    relevant_in_top_k = len(set(top_k) & ground_truth)
    
    # Recall = (relevant found) / (total relevant)
    recall = relevant_in_top_k / len(ground_truth)
    
    return recall


def compute_mrr(
    predictions: List[str],
    ground_truth: Set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR measures how high the first relevant document appears in the ranking.
    It's the reciprocal of the rank of the first relevant result.
    
    Args:
        predictions: List of predicted document IDs (ordered by relevance)
        ground_truth: Set of relevant document IDs
        
    Returns:
        MRR score (0.0 to 1.0)
        
    Examples:
        >>> compute_mrr(["doc1", "doc2", "doc3"], {"doc2"})
        0.5  # First relevant doc at rank 2, so 1/2 = 0.5
        
        >>> compute_mrr(["doc1", "doc2"], {"doc1"})
        1.0  # First relevant doc at rank 1, so 1/1 = 1.0
        
        >>> compute_mrr(["doc1", "doc2"], {"doc3"})
        0.0  # No relevant docs found
    """
    if not ground_truth or not predictions:
        return 0.0
    
    # Find rank of first relevant document (1-indexed)
    for rank, doc_id in enumerate(predictions, start=1):
        if doc_id in ground_truth:
            return 1.0 / rank
    
    # No relevant document found
    return 0.0


def compute_ndcg(
    predictions: List[str],
    ground_truth: Set[str],
    k: int = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG measures ranking quality by giving higher weight to relevant
    documents that appear earlier in the ranking.
    
    Args:
        predictions: List of predicted document IDs (ordered by relevance)
        ground_truth: Set of relevant document IDs
        k: Number of top predictions to consider (None = all)
        
    Returns:
        NDCG score (0.0 to 1.0)
        
    Note:
        Uses binary relevance (relevant=1, not relevant=0).
        DCG formula: sum(rel_i / log2(i+1)) for i in 1..k
    """
    if not ground_truth or not predictions:
        return 0.0
    
    # Limit to top K if specified
    if k is not None:
        predictions = predictions[:k]
    
    # Compute DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(predictions, start=1):
        relevance = 1.0 if doc_id in ground_truth else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # Compute IDCG (Ideal DCG) - best possible ranking
    # Ideal ranking has all relevant docs first
    num_relevant = min(len(ground_truth), len(predictions))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
    
    # NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


def evaluate_retrieval(
    predictions_list: List[List[str]],
    ground_truth_list: List[Set[str]],
    k_values: List[int] = [1, 5, 10],
) -> MetricsResult:
    """
    Evaluate retrieval performance across multiple queries.
    
    Computes Recall@K (for multiple K), MRR, and NDCG averaged across
    all queries.
    
    Args:
        predictions_list: List of prediction lists (one per query)
        ground_truth_list: List of ground truth sets (one per query)
        k_values: List of K values for Recall@K (default: [1, 5, 10])
        
    Returns:
        MetricsResult with averaged metrics
        
    Raises:
        ValueError: If predictions and ground truth lists have different lengths
        
    Example:
        >>> preds = [["doc1", "doc2"], ["doc3", "doc4"]]
        >>> truth = [{"doc1"}, {"doc3", "doc5"}]
        >>> result = evaluate_retrieval(preds, truth)
        >>> print(result.recall_at_1, result.mrr)
    """
    if len(predictions_list) != len(ground_truth_list):
        raise ValueError(
            f"Predictions ({len(predictions_list)}) and ground truth "
            f"({len(ground_truth_list)}) must have same length"
        )
    
    if not predictions_list:
        logger.warning("Empty predictions list, returning zero metrics")
        return MetricsResult()
    
    num_queries = len(predictions_list)
    
    # Compute metrics for each query
    recall_at_k_scores = {k: [] for k in k_values}
    mrr_scores = []
    ndcg_scores = []
    
    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        # Recall@K for each K value
        for k in k_values:
            recall = compute_recall_at_k(predictions, ground_truth, k)
            recall_at_k_scores[k].append(recall)
        
        # MRR
        mrr = compute_mrr(predictions, ground_truth)
        mrr_scores.append(mrr)
        
        # NDCG@10
        ndcg = compute_ndcg(predictions, ground_truth, k=10)
        ndcg_scores.append(ndcg)
    
    # Average across all queries
    result = MetricsResult(
        recall_at_1=np.mean(recall_at_k_scores.get(1, [0.0])),
        recall_at_5=np.mean(recall_at_k_scores.get(5, [0.0])),
        recall_at_10=np.mean(recall_at_k_scores.get(10, [0.0])),
        mrr=np.mean(mrr_scores),
        ndcg=np.mean(ndcg_scores),
    )
    
    logger.info(
        f"Evaluated {num_queries} queries: "
        f"R@1={result.recall_at_1:.3f}, R@5={result.recall_at_5:.3f}, "
        f"R@10={result.recall_at_10:.3f}, MRR={result.mrr:.3f}, NDCG={result.ndcg:.3f}"
    )
    
    return result

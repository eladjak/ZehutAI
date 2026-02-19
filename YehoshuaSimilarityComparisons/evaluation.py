"""Retrieval quality evaluation metrics.

Standard Information Retrieval metrics for measuring the quality of
document retrieval systems, including Precision@K, Recall@K, Average
Precision, MRR, NDCG@K, and a convenience aggregation function.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K: fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved: Ordered list of retrieved document identifiers
            (most relevant first).
        relevant: Set of ground-truth relevant document identifiers.
        k: Number of top results to consider. Must be >= 1.

    Returns:
        Precision at k as a float in [0.0, 1.0]. Returns 0.0 if k <= 0
        or *retrieved* is empty.

    Examples:
        >>> precision_at_k(["a", "b", "c"], {"a", "c"}, k=2)
        0.5
        >>> precision_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        0.6666666666666666
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-k results.

    Args:
        retrieved: Ordered list of retrieved document identifiers
            (most relevant first).
        relevant: Set of ground-truth relevant document identifiers.
        k: Number of top results to consider. Must be >= 1.

    Returns:
        Recall at k as a float in [0.0, 1.0]. Returns 0.0 if *relevant*
        is empty or k <= 0.

    Examples:
        >>> recall_at_k(["a", "b", "c"], {"a", "c"}, k=2)
        0.5
        >>> recall_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        1.0
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision (AP): mean of precision values at each relevant position.

    AP is the average of Precision@k values computed at every rank position
    where a relevant document is retrieved, divided by the total number of
    relevant documents.

    Args:
        retrieved: Ordered list of retrieved document identifiers
            (most relevant first).
        relevant: Set of ground-truth relevant document identifiers.

    Returns:
        Average precision as a float in [0.0, 1.0]. Returns 0.0 if
        *retrieved* or *relevant* is empty, or no relevant docs appear
        in *retrieved*.

    Examples:
        >>> average_precision(["a", "b", "c", "d"], {"a", "c"})
        0.8333333333333334
    """
    if not retrieved or not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / len(relevant)


def mean_reciprocal_rank(
    queries_results: list[tuple[list[str], set[str]]],
) -> float:
    """Mean Reciprocal Rank (MRR) across multiple queries.

    For each query the Reciprocal Rank (RR) is 1 / (rank of the first
    relevant document), or 0.0 if no relevant document appears in the
    retrieved list.  MRR is the mean of all per-query RR values.

    Args:
        queries_results: List of (retrieved, relevant) pairs, one per query.
            *retrieved* is an ordered list of document IDs and *relevant* is
            the set of ground-truth relevant IDs for that query.

    Returns:
        MRR score as a float in [0.0, 1.0]. Returns 0.0 for an empty
        input list.

    Examples:
        >>> mean_reciprocal_rank([
        ...     (["b", "a", "c"], {"a"}),
        ...     (["a", "b", "c"], {"a"}),
        ... ])
        0.75
    """
    if not queries_results:
        return 0.0
    reciprocal_ranks: list[float] = []
    for retrieved, relevant in queries_results:
        rr = 0.0
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: gain is 1 for a document in *relevant*, 0
    otherwise.  The discount factor for position i (1-indexed) is
    ``1 / log2(i + 1)``.

    Args:
        retrieved: Ordered list of retrieved document identifiers
            (most relevant first).
        relevant: Set of ground-truth relevant document identifiers.
        k: Number of top results to consider. Must be >= 1.

    Returns:
        NDCG at k as a float in [0.0, 1.0]. Returns 0.0 if k <= 0 or
        *relevant* is empty.

    Examples:
        >>> ndcg_at_k(["a", "b", "c"], {"a"}, k=3)
        1.0
        >>> round(ndcg_at_k(["b", "a", "c"], {"a"}, k=3), 4)
        0.6309
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    # DCG: discount by log2(i + 2) where i is 0-indexed rank
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc in enumerate(top_k)
        if doc in relevant
    )
    # IDCG: ideal ordering puts all relevant docs at the top ranks
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retrieved: list[str],
    relevant: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute a standard suite of retrieval metrics for a single query.

    Convenience wrapper that returns Precision@K, Recall@K, and NDCG@K
    for every K in *k_values*, plus Average Precision (AP).

    Args:
        retrieved: Ordered list of retrieved document identifiers
            (most relevant first).
        relevant: Set of ground-truth relevant document identifiers.
        k_values: Cut-off ranks to evaluate. Defaults to [1, 3, 5, 10].

    Returns:
        Dictionary with the following keys (one group per k in *k_values*):

        - ``"p@{k}"``    – Precision at K
        - ``"r@{k}"``    – Recall at K
        - ``"ndcg@{k}"`` – NDCG at K
        - ``"ap"``       – Average Precision (over the full list)

    Examples:
        >>> metrics = evaluate_retrieval(["a", "b", "c"], {"a", "c"}, k_values=[1, 3])
        >>> metrics["p@1"]
        1.0
        >>> metrics["r@3"]
        1.0
        >>> "ap" in metrics
        True
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"p@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"r@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)

    metrics["ap"] = average_precision(retrieved, relevant)
    return metrics


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall.

    Args:
        precision: Precision value in [0.0, 1.0].
        recall: Recall value in [0.0, 1.0].

    Returns:
        F1 score as a float in [0.0, 1.0]. Returns 0.0 if both
        *precision* and *recall* are zero.
    """
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

"""Tests for the evaluation metrics module.

Covers all public functions in YehoshuaSimilarityComparisons/evaluation.py
with at least 15 test cases including edge cases, known mathematical
examples, and multi-query scenarios.
"""

from __future__ import annotations

import math

import pytest

from evaluation import (
    average_precision,
    evaluate_retrieval,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_retrieval(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_partial_retrieval(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        # top-2: ["a", "x"] -> 1 hit out of 2
        assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)

    def test_no_relevant_docs_in_top_k(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"z"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(0.0)

    def test_k_larger_than_list(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # k=10 but only 2 docs retrieved -> 2 hits / 10
        assert precision_at_k(retrieved, relevant, 10) == pytest.approx(0.2)

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, 0) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self) -> None:
        assert precision_at_k([], {"a"}, 5) == pytest.approx(0.0)

    def test_empty_relevant_no_hits(self) -> None:
        assert precision_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)

    def test_k_equals_one_hit(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a"}, 1) == pytest.approx(1.0)

    def test_k_equals_one_miss(self) -> None:
        assert precision_at_k(["x", "a", "c"], {"a"}, 1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_full_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c"}
        # top-2: ["a", "x"] -> 1 out of 3 relevant
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)

    def test_zero_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"z"}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], {"a"}, 0) == pytest.approx(0.0)

    def test_k_larger_than_list_full_recall(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 100) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------

class TestAveragePrecision:
    def test_perfect_ranking(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        # P@1=1, P@2=1, P@3=1 -> AP = (1+1+1)/3 = 1.0
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_reverse_ranking(self) -> None:
        retrieved = ["x", "y", "z", "a", "b"]
        relevant = {"a", "b"}
        # P@4 = 1/4, P@5 = 2/5 -> AP = (0.25 + 0.4) / 2 = 0.325
        expected = (1 / 4 + 2 / 5) / 2
        assert average_precision(retrieved, relevant) == pytest.approx(expected)

    def test_single_relevant_doc_first(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        # P@1 = 1.0 -> AP = 1.0
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_single_relevant_doc_second(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"b"}
        # P@2 = 1/2 -> AP = 0.5
        assert average_precision(retrieved, relevant) == pytest.approx(0.5)

    def test_no_relevant_in_retrieved(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"z"}
        assert average_precision(retrieved, relevant) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert average_precision(["a", "b"], set()) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self) -> None:
        assert average_precision([], {"a"}) == pytest.approx(0.0)

    def test_interleaved_relevant_docs(self) -> None:
        # a at rank 1, c at rank 3; relevant = {a, c}
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        # P@1 = 1/1 = 1.0, P@3 = 2/3 -> AP = (1.0 + 2/3) / 2
        expected = (1.0 + 2 / 3) / 2
        assert average_precision(retrieved, relevant) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_first_result_relevant(self) -> None:
        queries = [
            (["a", "b", "c"], {"a"}),
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(1.0)

    def test_second_result_relevant(self) -> None:
        queries = [
            (["a", "b", "c"], {"b"}),
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(0.5)

    def test_none_relevant(self) -> None:
        queries = [
            (["a", "b", "c"], {"z"}),
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(0.0)

    def test_multiple_queries_mixed(self) -> None:
        queries = [
            (["a", "b", "c"], {"a"}),   # RR = 1.0
            (["a", "b", "c"], {"b"}),   # RR = 0.5
            (["a", "b", "c"], {"z"}),   # RR = 0.0
        ]
        expected = (1.0 + 0.5 + 0.0) / 3
        assert mean_reciprocal_rank(queries) == pytest.approx(expected)

    def test_empty_input_returns_zero(self) -> None:
        assert mean_reciprocal_rank([]) == pytest.approx(0.0)

    def test_takes_first_relevant_only(self) -> None:
        # Both a and b are relevant; first hit at rank 1
        queries = [
            (["a", "b", "c"], {"a", "b"}),
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(1.0)

    def test_two_queries_average(self) -> None:
        queries = [
            (["b", "a", "c"], {"a"}),   # RR = 1/2
            (["a", "b", "c"], {"a"}),   # RR = 1/1
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_order(self) -> None:
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_reversed_order_single_relevant(self) -> None:
        # a is at 0-indexed position 2 (rank 3)
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        k = 3
        # DCG:  1/log2(3+1) = 1/log2(4) = 0.5
        # IDCG: 1/log2(1+1) = 1/log2(2) = 1.0
        dcg = 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2)
        expected = dcg / idcg
        assert ndcg_at_k(retrieved, relevant, k) == pytest.approx(expected)

    def test_partial_relevance_two_docs(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        k = 4
        # a at rank 1: 1/log2(2); b at rank 3: 1/log2(4)
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        # Ideal: a,b at ranks 1,2
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(retrieved, relevant, k) == pytest.approx(expected)

    def test_no_relevant_in_retrieved(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"z"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a"}, 0) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)

    def test_k_larger_than_retrieved_single_doc(self) -> None:
        retrieved = ["a"]
        relevant = {"a"}
        # Only one doc, k=5 -> DCG = IDCG -> NDCG = 1.0
        assert ndcg_at_k(retrieved, relevant, 5) == pytest.approx(1.0)

    def test_single_relevant_at_rank_two(self) -> None:
        # Known value: DCG = 1/log2(3) ≈ 0.6309; IDCG = 1/log2(2) = 1.0
        retrieved = ["x", "a"]
        relevant = {"a"}
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# evaluate_retrieval (convenience wrapper)
# ---------------------------------------------------------------------------

class TestEvaluateRetrieval:
    def test_default_k_values_keys_present(self) -> None:
        retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        relevant = {"a", "c"}
        metrics = evaluate_retrieval(retrieved, relevant)
        for k in [1, 3, 5, 10]:
            assert f"p@{k}" in metrics
            assert f"r@{k}" in metrics
            assert f"ndcg@{k}" in metrics
        assert "ap" in metrics

    def test_custom_k_values(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        metrics = evaluate_retrieval(retrieved, relevant, k_values=[1, 3])
        assert set(metrics.keys()) == {"p@1", "r@1", "ndcg@1", "p@3", "r@3", "ndcg@3", "ap"}

    def test_perfect_retrieval_all_ones(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        metrics = evaluate_retrieval(retrieved, relevant, k_values=[3])
        assert metrics["p@3"] == pytest.approx(1.0)
        assert metrics["r@3"] == pytest.approx(1.0)
        assert metrics["ndcg@3"] == pytest.approx(1.0)
        assert metrics["ap"] == pytest.approx(1.0)

    def test_no_relevant_docs_all_zeros(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"z"}
        metrics = evaluate_retrieval(retrieved, relevant, k_values=[1, 3])
        for v in metrics.values():
            assert v == pytest.approx(0.0)

    def test_values_match_individual_functions(self) -> None:
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        k = 3
        metrics = evaluate_retrieval(retrieved, relevant, k_values=[k])
        assert metrics[f"p@{k}"] == pytest.approx(precision_at_k(retrieved, relevant, k))
        assert metrics[f"r@{k}"] == pytest.approx(recall_at_k(retrieved, relevant, k))
        assert metrics[f"ndcg@{k}"] == pytest.approx(ndcg_at_k(retrieved, relevant, k))
        assert metrics["ap"] == pytest.approx(average_precision(retrieved, relevant))

    def test_empty_retrieved_returns_zeros(self) -> None:
        metrics = evaluate_retrieval([], {"a", "b"}, k_values=[1, 5])
        for v in metrics.values():
            assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------

class TestF1Score:
    def test_perfect_precision_and_recall(self) -> None:
        assert f1_score(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        assert f1_score(0.0, 0.8) == pytest.approx(0.0)

    def test_zero_recall(self) -> None:
        assert f1_score(0.8, 0.0) == pytest.approx(0.0)

    def test_both_zero(self) -> None:
        assert f1_score(0.0, 0.0) == pytest.approx(0.0)

    def test_balanced_precision_recall(self) -> None:
        assert f1_score(0.5, 0.5) == pytest.approx(0.5)

    def test_asymmetric_values(self) -> None:
        expected = 2 * 0.4 * 0.8 / (0.4 + 0.8)
        assert f1_score(0.4, 0.8) == pytest.approx(expected)

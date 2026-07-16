"""Tests for the hebrew_benchmark module.

All tests here run without downloading any ML models by mocking the
model loading layer.  Tests that require real model weights are marked
with @pytest.mark.slow and are excluded from the default test run.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from zehutai.hebrew.hebrew_benchmark import (
    HEBREW_BENCHMARK_PAIRS,
    MODEL_CONFIGS,
    _cosine_similarity,
    evaluate_benchmark,
    format_benchmark_report,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(scores: list[float] | None = None) -> MagicMock:
    """Return a mock model whose encode() produces embeddings yielding known scores.

    When *scores* is None the model returns unit vectors that are identical for
    both sentences in a pair (cosine similarity = 1.0).  This is a valid
    fallback for structural tests that do not care about the actual score.

    Args:
        scores: Optional list of target cosine similarity values, one per
            ``encode`` call.  Each call encodes exactly one sentence pair so
            each score controls one benchmark result.

    Returns:
        MagicMock with an ``encode`` method configured as described.
    """
    model = MagicMock()
    call_index = [0]

    if scores is None:

        def _encode(sentences: list[str], **kwargs: Any) -> np.ndarray:
            n = len(sentences)
            emb = np.ones((n, 4), dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / norms

    else:

        def _encode(sentences: list[str], **kwargs: Any) -> np.ndarray:  # type: ignore[misc]
            idx = call_index[0]
            call_index[0] += 1
            target = scores[idx % len(scores)]
            # Construct two unit vectors with cosine = target.
            # a = [1, 0], b = [target, sqrt(1 - target^2)] gives cos(a, b) = target.
            target_clamped = float(np.clip(target, -1.0, 1.0))
            sin_val = math.sqrt(max(0.0, 1.0 - target_clamped**2))
            emb = np.array(
                [[1.0, 0.0], [target_clamped, sin_val]],
                dtype=np.float32,
            )
            return emb

    model.encode = _encode
    return model


def _make_results(
    labels: list[str],
    scores: list[float],
    categories: list[str] | None = None,
) -> list[dict]:
    """Build synthetic results list for evaluate_benchmark tests.

    Args:
        labels: Parallel list of label strings ("high", "medium", "low").
        scores: Parallel list of cosine similarity float values.
        categories: Optional parallel list of category strings.

    Returns:
        List of result dicts with ``label``, ``score``, ``s1``, ``s2``,
        ``category``, and ``model`` keys.
    """
    if categories is None:
        categories = ["test"] * len(labels)
    return [
        {
            "s1": f"sentence_a_{i}",
            "s2": f"sentence_b_{i}",
            "label": label,
            "category": cat,
            "score": score,
            "model": "mock-model",
        }
        for i, (label, score, cat) in enumerate(zip(labels, scores, categories, strict=False))
    ]


# ---------------------------------------------------------------------------
# Tests: HEBREW_BENCHMARK_PAIRS data integrity
# ---------------------------------------------------------------------------


class TestBenchmarkPairsStructure:
    """Validate that HEBREW_BENCHMARK_PAIRS has the required schema."""

    def test_nonempty(self) -> None:
        """Benchmark pairs list must not be empty."""
        assert len(HEBREW_BENCHMARK_PAIRS) > 0

    def test_all_pairs_have_required_keys(self) -> None:
        """Every pair must contain s1, s2, label, and category."""
        required = {"s1", "s2", "label", "category"}
        for i, pair in enumerate(HEBREW_BENCHMARK_PAIRS):
            missing = required - pair.keys()
            assert not missing, f"Pair {i} is missing keys: {missing}"

    def test_s1_and_s2_are_nonempty_strings(self) -> None:
        """s1 and s2 must be non-empty strings."""
        for i, pair in enumerate(HEBREW_BENCHMARK_PAIRS):
            assert isinstance(pair["s1"], str) and pair["s1"], f"Pair {i}: s1 is empty"
            assert isinstance(pair["s2"], str) and pair["s2"], f"Pair {i}: s2 is empty"

    def test_labels_are_valid(self) -> None:
        """All label values must be 'high', 'medium', or 'low'."""
        valid_labels = {"high", "medium", "low"}
        for i, pair in enumerate(HEBREW_BENCHMARK_PAIRS):
            assert pair["label"] in valid_labels, f"Pair {i}: unexpected label '{pair['label']}'"

    def test_label_distribution_minimum_three_each(self) -> None:
        """There must be at least 3 pairs for each of high, medium, and low."""
        from collections import Counter

        counts = Counter(p["label"] for p in HEBREW_BENCHMARK_PAIRS)
        for label in ("high", "medium", "low"):
            assert counts[label] >= 3, f"Label '{label}' has only {counts[label]} pairs (need >= 3)"

    def test_s1_and_s2_are_different(self) -> None:
        """s1 and s2 should not be identical within a pair."""
        for i, pair in enumerate(HEBREW_BENCHMARK_PAIRS):
            assert pair["s1"] != pair["s2"], f"Pair {i}: s1 == s2"

    def test_categories_are_strings(self) -> None:
        """category values must be non-empty strings."""
        for i, pair in enumerate(HEBREW_BENCHMARK_PAIRS):
            assert isinstance(pair["category"], str) and pair["category"], (
                f"Pair {i}: category is not a non-empty string"
            )

    def test_contains_hebrew_text(self) -> None:
        """At least one sentence in every pair must contain Hebrew characters."""

        # Hebrew Unicode block: U+0590-U+05FF
        def _has_hebrew(text: str) -> bool:
            return any("\u0590" <= ch <= "\u05ff" for ch in text)

        for i, pair in enumerate(HEBREW_BENCHMARK_PAIRS):
            assert _has_hebrew(pair["s1"]) or _has_hebrew(pair["s2"]), (
                f"Pair {i} contains no Hebrew characters"
            )

    def test_high_similarity_category_paraphrase_exists(self) -> None:
        """There should be at least one pair with label=high and category=paraphrase."""
        found = any(
            p["label"] == "high" and p["category"] == "paraphrase" for p in HEBREW_BENCHMARK_PAIRS
        )
        assert found, "No high-similarity paraphrase pairs found"

    def test_register_category_exists(self) -> None:
        """There should be pairs testing register variation."""
        found = any(p["category"] == "register" for p in HEBREW_BENCHMARK_PAIRS)
        assert found, "No register-category pairs found"

    def test_construct_state_category_exists(self) -> None:
        """There should be pairs testing construct state (smichut)."""
        found = any(p["category"] == "construct_state" for p in HEBREW_BENCHMARK_PAIRS)
        assert found, "No construct_state-category pairs found"


# ---------------------------------------------------------------------------
# Tests: MODEL_CONFIGS
# ---------------------------------------------------------------------------


class TestModelConfigs:
    """Validate the MODEL_CONFIGS registry."""

    def test_nonempty(self) -> None:
        """MODEL_CONFIGS must define at least one model."""
        assert len(MODEL_CONFIGS) > 0

    def test_all_configs_have_name_and_type(self) -> None:
        """Every config entry must have 'name' and 'type' keys."""
        for key, cfg in MODEL_CONFIGS.items():
            assert "name" in cfg, f"Config '{key}' missing 'name'"
            assert "type" in cfg, f"Config '{key}' missing 'type'"

    def test_multilingual_mpnet_present(self) -> None:
        """multilingual-mpnet entry must be present."""
        assert "multilingual-mpnet" in MODEL_CONFIGS

    def test_multilingual_minilm_present(self) -> None:
        """multilingual-minilm entry must be present."""
        assert "multilingual-minilm" in MODEL_CONFIGS

    def test_dictabert_present(self) -> None:
        """dictabert entry must be present (even if not runnable)."""
        assert "dictabert" in MODEL_CONFIGS

    def test_alephbert_present(self) -> None:
        """alephbert entry must be present (even if not runnable)."""
        assert "alephbert" in MODEL_CONFIGS

    def test_names_are_strings(self) -> None:
        """All model name values must be non-empty strings."""
        for key, cfg in MODEL_CONFIGS.items():
            assert isinstance(cfg["name"], str) and cfg["name"], (
                f"Config '{key}': name must be a non-empty string"
            )


# ---------------------------------------------------------------------------
# Tests: _cosine_similarity helper
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Unit tests for the internal _cosine_similarity function."""

    def test_identical_vectors_return_one(self) -> None:
        """Cosine similarity of a vector with itself must be 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        result = _cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Cosine similarity of orthogonal vectors must be 0.0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = _cosine_similarity(a, b)
        assert abs(result) < 1e-6

    def test_opposite_vectors_return_minus_one(self) -> None:
        """Cosine similarity of a vector and its negation must be -1.0."""
        v = np.array([1.0, 2.0])
        result = _cosine_similarity(v, -v)
        assert abs(result - (-1.0)) < 1e-6

    def test_result_range(self) -> None:
        """Result must always be in [-1, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = rng.standard_normal(16)
            b = rng.standard_normal(16)
            result = _cosine_similarity(a, b)
            assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9

    def test_zero_vector_raises(self) -> None:
        """Zero-norm vector must raise ValueError."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Zero-norm"):
            _cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# Tests: evaluate_benchmark
# ---------------------------------------------------------------------------


class TestEvaluateBenchmark:
    """Tests for evaluate_benchmark() with synthetic scores."""

    def test_returns_dict(self) -> None:
        """evaluate_benchmark must return a dict."""
        results = _make_results(["high", "low"], [0.9, 0.1])
        assert isinstance(evaluate_benchmark(results), dict)

    def test_separation_score_high_minus_low(self) -> None:
        """separation_score should equal high_avg - low_avg."""
        results = _make_results(
            ["high", "high", "low", "low"],
            [0.8, 0.9, 0.1, 0.2],
        )
        metrics = evaluate_benchmark(results)
        expected = metrics["high_avg"] - metrics["low_avg"]
        assert abs(metrics["separation_score"] - expected) < 1e-9

    def test_high_low_gap_alias(self) -> None:
        """high_low_gap should be identical to separation_score."""
        results = _make_results(["high", "low"], [0.8, 0.2])
        metrics = evaluate_benchmark(results)
        assert metrics["high_low_gap"] == metrics["separation_score"]

    def test_averages_computed_correctly(self) -> None:
        """Group averages must match manually computed means."""
        results = _make_results(
            ["high", "high", "medium", "low", "low"],
            [0.9, 0.7, 0.5, 0.2, 0.1],
        )
        metrics = evaluate_benchmark(results)
        assert abs(metrics["high_avg"] - 0.8) < 1e-6
        assert abs(metrics["medium_avg"] - 0.5) < 1e-6
        assert abs(metrics["low_avg"] - 0.15) < 1e-6

    def test_rank_accuracy_perfect_ordering(self) -> None:
        """rank_accuracy should be 1.0 when high > medium > low."""
        results = _make_results(
            ["high", "medium", "low"],
            [0.9, 0.5, 0.1],
        )
        metrics = evaluate_benchmark(results)
        assert abs(metrics["rank_accuracy"] - 1.0) < 1e-9

    def test_rank_accuracy_reversed_ordering(self) -> None:
        """rank_accuracy should be 0.0 when high < medium < low."""
        results = _make_results(
            ["high", "medium", "low"],
            [0.1, 0.5, 0.9],
        )
        metrics = evaluate_benchmark(results)
        assert abs(metrics["rank_accuracy"] - 0.0) < 1e-9

    def test_count_metrics(self) -> None:
        """n_high, n_medium, n_low must reflect label counts."""
        results = _make_results(
            ["high", "high", "medium", "low", "low", "low"],
            [0.9, 0.8, 0.5, 0.2, 0.1, 0.15],
        )
        metrics = evaluate_benchmark(results)
        assert metrics["n_high"] == 2.0
        assert metrics["n_medium"] == 1.0
        assert metrics["n_low"] == 3.0

    def test_missing_label_group_nan(self) -> None:
        """If a label group is absent, its average should be NaN."""
        results = _make_results(["high", "high"], [0.8, 0.9])
        metrics = evaluate_benchmark(results)
        assert math.isnan(metrics["low_avg"])
        assert math.isnan(metrics["medium_avg"])

    def test_empty_results_raises(self) -> None:
        """Empty results list must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            evaluate_benchmark([])

    def test_avg_elapsed_ms_present_when_scores_have_elapsed(self) -> None:
        """avg_elapsed_ms is computed when elapsed_ms is present in results."""
        results = _make_results(["high", "low"], [0.8, 0.2])
        for r in results:
            r["elapsed_ms"] = 10.0
        metrics = evaluate_benchmark(results)
        assert abs(metrics["avg_elapsed_ms"] - 10.0) < 1e-9

    def test_avg_elapsed_ms_nan_when_absent(self) -> None:
        """avg_elapsed_ms should be NaN when elapsed_ms not in results."""
        results = _make_results(["high", "low"], [0.8, 0.2])
        metrics = evaluate_benchmark(results)
        assert math.isnan(metrics["avg_elapsed_ms"])

    def test_positive_separation_for_correct_ordering(self) -> None:
        """separation_score should be positive when high sims > low sims."""
        results = _make_results(
            ["high"] * 5 + ["medium"] * 3 + ["low"] * 4,
            [0.85, 0.90, 0.88, 0.82, 0.79, 0.55, 0.50, 0.52, 0.20, 0.15, 0.18, 0.10],
        )
        metrics = evaluate_benchmark(results)
        assert metrics["separation_score"] > 0.0


# ---------------------------------------------------------------------------
# Tests: format_benchmark_report
# ---------------------------------------------------------------------------


class TestFormatBenchmarkReport:
    """Tests for format_benchmark_report() output structure."""

    def _sample_all_results(self) -> dict[str, dict[str, float]]:
        return {
            "model-a": {
                "high_avg": 0.85,
                "medium_avg": 0.55,
                "low_avg": 0.20,
                "separation_score": 0.65,
                "high_low_gap": 0.65,
                "high_medium_gap": 0.30,
                "medium_low_gap": 0.35,
                "rank_accuracy": 1.0,
                "n_high": 5.0,
                "n_medium": 3.0,
                "n_low": 4.0,
                "avg_elapsed_ms": 12.5,
            },
            "model-b": {
                "high_avg": 0.70,
                "medium_avg": 0.50,
                "low_avg": 0.30,
                "separation_score": 0.40,
                "high_low_gap": 0.40,
                "high_medium_gap": 0.20,
                "medium_low_gap": 0.20,
                "rank_accuracy": 1.0,
                "n_high": 5.0,
                "n_medium": 3.0,
                "n_low": 4.0,
                "avg_elapsed_ms": 8.0,
            },
        }

    def test_returns_string(self) -> None:
        """format_benchmark_report must return a str."""
        report = format_benchmark_report(self._sample_all_results())
        assert isinstance(report, str)

    def test_report_nonempty(self) -> None:
        """Output must be non-empty."""
        report = format_benchmark_report(self._sample_all_results())
        assert len(report.strip()) > 0

    def test_model_names_appear_in_output(self) -> None:
        """Each model name must appear in the report."""
        all_results = self._sample_all_results()
        report = format_benchmark_report(all_results)
        for model_name in all_results:
            assert model_name in report, f"Model '{model_name}' not found in report"

    def test_report_contains_header(self) -> None:
        """Report must include a recognisable benchmark title."""
        report = format_benchmark_report(self._sample_all_results())
        assert "Benchmark" in report or "benchmark" in report

    def test_report_contains_ranking_section(self) -> None:
        """Report must include a ranking section."""
        report = format_benchmark_report(self._sample_all_results())
        assert "Ranking" in report or "ranking" in report

    def test_report_ranking_order(self) -> None:
        """Better separation score should appear first in the ranking."""
        all_results = self._sample_all_results()
        report = format_benchmark_report(all_results)
        pos_a = report.index("model-a")
        pos_b = report.index("model-b")
        # model-a has higher separation score (0.65 > 0.40) -> appears first
        assert pos_a < pos_b, "model-a (higher sep) should appear before model-b"

    def test_single_model_report(self) -> None:
        """Report with a single model must still be well-formed."""
        single = {"only-model": self._sample_all_results()["model-a"]}
        report = format_benchmark_report(single)
        assert "only-model" in report

    def test_empty_all_results_raises(self) -> None:
        """Empty all_results dict must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            format_benchmark_report({})

    def test_multiline_output(self) -> None:
        """Report must span multiple lines."""
        report = format_benchmark_report(self._sample_all_results())
        assert report.count("\n") >= 5

    def test_separation_scores_appear_in_report(self) -> None:
        """Numeric separation score values should be visible in the report."""
        report = format_benchmark_report(self._sample_all_results())
        # model-a has separation 0.65
        assert "0.65" in report or "0.6500" in report


# ---------------------------------------------------------------------------
# Tests: run_benchmark with mocked model
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Tests for run_benchmark() using a mock model instance."""

    def test_returns_list(self) -> None:
        """run_benchmark must return a list."""
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", model_instance=mock_model)
        assert isinstance(results, list)

    def test_result_count_matches_pairs(self) -> None:
        """Number of results must equal number of input pairs."""
        mock_model = _make_mock_model()
        pairs = HEBREW_BENCHMARK_PAIRS[:5]
        results = run_benchmark("mock-model", pairs=pairs, model_instance=mock_model)
        assert len(results) == 5

    def test_default_pairs_used_when_none(self) -> None:
        """When pairs=None, all HEBREW_BENCHMARK_PAIRS should be evaluated."""
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", model_instance=mock_model)
        assert len(results) == len(HEBREW_BENCHMARK_PAIRS)

    def test_result_has_score_key(self) -> None:
        """Every result dict must contain a 'score' key."""
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", model_instance=mock_model)
        for r in results:
            assert "score" in r, f"Missing 'score' key in result: {r}"

    def test_score_is_float_in_range(self) -> None:
        """Every score must be a float in [-1, 1]."""
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", model_instance=mock_model)
        for r in results:
            score = r["score"]
            assert isinstance(score, float), f"score is not float: {type(score)}"
            assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6, f"score out of range: {score}"

    def test_result_preserves_original_fields(self) -> None:
        """Original pair fields (s1, s2, label, category) must be in results."""
        mock_model = _make_mock_model()
        pairs = HEBREW_BENCHMARK_PAIRS[:3]
        results = run_benchmark("mock-model", pairs=pairs, model_instance=mock_model)
        for result, orig in zip(results, pairs, strict=False):
            for key in ("s1", "s2", "label", "category"):
                assert key in result, f"Key '{key}' missing from result"
                assert result[key] == orig[key], (
                    f"Key '{key}' mismatch: {result[key]} != {orig[key]}"
                )

    def test_result_contains_model_name(self) -> None:
        """Result dicts must contain the 'model' field with the given name."""
        mock_model = _make_mock_model()
        results = run_benchmark("test-model-name", model_instance=mock_model)
        for r in results:
            assert r.get("model") == "test-model-name"

    def test_result_contains_elapsed_ms(self) -> None:
        """Result dicts must contain an 'elapsed_ms' key with a non-negative float."""
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", model_instance=mock_model)
        for r in results:
            assert "elapsed_ms" in r, "elapsed_ms missing from result"
            assert r["elapsed_ms"] >= 0.0, "elapsed_ms must be non-negative"

    def test_custom_pairs_accepted(self) -> None:
        """run_benchmark should accept arbitrary custom pairs."""
        custom_pairs = [
            {"s1": "שלום עולם", "s2": "היי עולם", "label": "high", "category": "test"},
            {"s1": "כלב", "s2": "מחשב", "label": "low", "category": "test"},
        ]
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", pairs=custom_pairs, model_instance=mock_model)
        assert len(results) == 2

    def test_model_config_key_accepted(self) -> None:
        """run_benchmark should accept a MODEL_CONFIGS key as model_name."""
        mock_model = _make_mock_model()
        # Provide the mock directly so no real model is loaded
        results = run_benchmark(
            "multilingual-mpnet",
            pairs=HEBREW_BENCHMARK_PAIRS[:2],
            model_instance=mock_model,
        )
        assert len(results) == 2

    def test_integration_with_evaluate(self) -> None:
        """run_benchmark output should be directly usable by evaluate_benchmark."""
        mock_model = _make_mock_model()
        results = run_benchmark("mock-model", model_instance=mock_model)
        metrics = evaluate_benchmark(results)
        assert isinstance(metrics, dict)
        assert "separation_score" in metrics

    def test_integration_pipeline(self) -> None:
        """End-to-end: run -> evaluate -> format should produce a non-empty report."""
        # Use a mock model that returns plausible scores
        high_scores = [0.85] * 7  # high-label pairs
        med_scores = [0.55] * 3  # medium-label pairs
        low_scores = [0.20] * 4  # low-label pairs
        all_scores = high_scores + med_scores + low_scores
        mock_model = _make_mock_model(scores=all_scores)

        results = run_benchmark("pipeline-test", model_instance=mock_model)
        metrics = evaluate_benchmark(results)
        report = format_benchmark_report({"pipeline-test": metrics})

        assert isinstance(report, str)
        assert len(report.strip()) > 0
        assert "pipeline-test" in report


# ---------------------------------------------------------------------------
# Slow tests (require real model downloads)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestWithRealModel:
    """Integration tests that load actual models from Hugging Face.

    These tests are excluded from the default run.
    Run with: pytest -m slow --override-ini='addopts='
    """

    def test_multilingual_mpnet_produces_valid_scores(self) -> None:
        """Real multilingual-mpnet model should score high pairs above low pairs."""
        results = run_benchmark("multilingual-mpnet")
        metrics = evaluate_benchmark(results)
        assert metrics["separation_score"] > 0.1, (
            f"Expected separation > 0.1, got {metrics['separation_score']}"
        )

    def test_multilingual_minilm_produces_valid_scores(self) -> None:
        """Real multilingual-MiniLM model should score high pairs above low pairs."""
        results = run_benchmark("multilingual-minilm")
        metrics = evaluate_benchmark(results)
        assert metrics["separation_score"] > 0.1

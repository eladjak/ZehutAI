"""Tests for HEBREW_SENTENCE_PAIRS corpus and the summary-dict benchmark API.

All tests run without downloading any ML models by mocking
``embeddings_comparison.compare_sentences`` in the test suite.

Test coverage:
- HEBREW_SENTENCE_PAIRS has exactly 20 entries
- Category distribution is correct (7 high, 7 medium, 6 low)
- All pairs have the required keys
- format_benchmark_report returns a non-empty string
- run_benchmark returns the correct dict structure (mocked model)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

# Ensure project root and sub-package are importable
_root = str(Path(__file__).resolve().parent.parent)
_sub = str(Path(__file__).resolve().parent.parent / "YehoshuaSimilarityComparisons")
for _p in (_root, _sub):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hebrew_benchmark import (
    HEBREW_SENTENCE_PAIRS,
    format_benchmark_report,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_PAIR_KEYS = {"id", "sent1", "sent2", "category", "description"}
_VALID_CATEGORIES = {"high", "medium", "low"}


def _mock_compare_sentences(sentences: list[str]) -> float:
    """Deterministic fake for compare_sentences, seeded by content.

    Returns a float in (0, 1) derived from the hash of the inputs so that
    different pairs yield different values without loading any model.
    """
    seed = hash(tuple(sentences)) % (2**31)
    # Map to (0, 1) range deterministically
    return (seed % 10000) / 10000.0


def _make_run_benchmark_results(score: float = 0.5) -> dict[str, Any]:
    """Return a minimal run_benchmark() result dict with a fixed score."""
    pairs = [
        {
            "id": pair["id"],
            "sent1": pair["sent1"],
            "sent2": pair["sent2"],
            "category": pair["category"],
            "description": pair["description"],
            "score": score,
        }
        for pair in HEBREW_SENTENCE_PAIRS
    ]
    high_scores = [p["score"] for p in pairs if p["category"] == "high"]
    medium_scores = [p["score"] for p in pairs if p["category"] == "medium"]
    low_scores = [p["score"] for p in pairs if p["category"] == "low"]

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    high_avg = _avg(high_scores)
    medium_avg = _avg(medium_scores)
    low_avg = _avg(low_scores)

    return {
        "model": "test-model",
        "pairs": pairs,
        "stats": {
            "high_avg": high_avg,
            "medium_avg": medium_avg,
            "low_avg": low_avg,
            "separation_score": high_avg - low_avg,
        },
    }


# ---------------------------------------------------------------------------
# Tests: HEBREW_SENTENCE_PAIRS corpus integrity
# ---------------------------------------------------------------------------


class TestHebrewSentencePairsCorpus:
    """Validate the structure and content of HEBREW_SENTENCE_PAIRS."""

    def test_corpus_has_exactly_20_pairs(self) -> None:
        """HEBREW_SENTENCE_PAIRS must contain exactly 20 sentence pairs."""
        assert len(HEBREW_SENTENCE_PAIRS) == 20, (
            f"Expected 20 pairs, got {len(HEBREW_SENTENCE_PAIRS)}"
        )

    def test_high_category_count_is_7(self) -> None:
        """There must be exactly 7 pairs with category='high'."""
        counts = Counter(p["category"] for p in HEBREW_SENTENCE_PAIRS)
        assert counts["high"] == 7, (
            f"Expected 7 high pairs, got {counts['high']}"
        )

    def test_medium_category_count_is_7(self) -> None:
        """There must be exactly 7 pairs with category='medium'."""
        counts = Counter(p["category"] for p in HEBREW_SENTENCE_PAIRS)
        assert counts["medium"] == 7, (
            f"Expected 7 medium pairs, got {counts['medium']}"
        )

    def test_low_category_count_is_6(self) -> None:
        """There must be exactly 6 pairs with category='low'."""
        counts = Counter(p["category"] for p in HEBREW_SENTENCE_PAIRS)
        assert counts["low"] == 6, (
            f"Expected 6 low pairs, got {counts['low']}"
        )

    def test_all_pairs_have_required_keys(self) -> None:
        """Every pair must have id, sent1, sent2, category, and description."""
        for i, pair in enumerate(HEBREW_SENTENCE_PAIRS):
            missing = _REQUIRED_PAIR_KEYS - pair.keys()
            assert not missing, f"Pair {i} is missing keys: {missing}"

    def test_all_pair_values_are_nonempty_strings(self) -> None:
        """All five required fields must be non-empty strings."""
        for i, pair in enumerate(HEBREW_SENTENCE_PAIRS):
            for key in _REQUIRED_PAIR_KEYS:
                val = pair.get(key, "")
                assert isinstance(val, str) and val, (
                    f"Pair {i}: field '{key}' is not a non-empty string (got {val!r})"
                )

    def test_all_categories_are_valid(self) -> None:
        """All category values must be 'high', 'medium', or 'low'."""
        for i, pair in enumerate(HEBREW_SENTENCE_PAIRS):
            assert pair["category"] in _VALID_CATEGORIES, (
                f"Pair {i}: unexpected category '{pair['category']}'"
            )

    def test_all_ids_are_unique(self) -> None:
        """Each pair id must be unique within the corpus."""
        ids = [p["id"] for p in HEBREW_SENTENCE_PAIRS]
        assert len(ids) == len(set(ids)), "Duplicate ids found in HEBREW_SENTENCE_PAIRS"

    def test_sent1_and_sent2_are_different_in_each_pair(self) -> None:
        """sent1 and sent2 should not be identical within a pair."""
        for i, pair in enumerate(HEBREW_SENTENCE_PAIRS):
            assert pair["sent1"] != pair["sent2"], (
                f"Pair {i} ({pair['id']}): sent1 == sent2"
            )

    def test_pairs_contain_hebrew_characters(self) -> None:
        """Every pair must contain at least one Hebrew Unicode character."""
        def _has_hebrew(text: str) -> bool:
            return any("\u0590" <= ch <= "\u05ff" for ch in text)

        for i, pair in enumerate(HEBREW_SENTENCE_PAIRS):
            assert _has_hebrew(pair["sent1"]) or _has_hebrew(pair["sent2"]), (
                f"Pair {i} ({pair['id']}): no Hebrew characters found"
            )

    def test_high_ids_start_with_high_prefix(self) -> None:
        """Pairs with category='high' should have ids starting with 'high_'."""
        high_pairs = [p for p in HEBREW_SENTENCE_PAIRS if p["category"] == "high"]
        for pair in high_pairs:
            assert pair["id"].startswith("high_"), (
                f"High pair id '{pair['id']}' does not start with 'high_'"
            )

    def test_medium_ids_start_with_medium_prefix(self) -> None:
        """Pairs with category='medium' should have ids starting with 'medium_'."""
        medium_pairs = [p for p in HEBREW_SENTENCE_PAIRS if p["category"] == "medium"]
        for pair in medium_pairs:
            assert pair["id"].startswith("medium_"), (
                f"Medium pair id '{pair['id']}' does not start with 'medium_'"
            )

    def test_low_ids_start_with_low_prefix(self) -> None:
        """Pairs with category='low' should have ids starting with 'low_'."""
        low_pairs = [p for p in HEBREW_SENTENCE_PAIRS if p["category"] == "low"]
        for pair in low_pairs:
            assert pair["id"].startswith("low_"), (
                f"Low pair id '{pair['id']}' does not start with 'low_'"
            )


# ---------------------------------------------------------------------------
# Tests: format_benchmark_report
# ---------------------------------------------------------------------------


class TestFormatBenchmarkReport:
    """Tests for format_benchmark_report with a single-model summary dict."""

    def test_returns_string(self) -> None:
        """format_benchmark_report must return a str."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        assert isinstance(report, str)

    def test_report_is_nonempty(self) -> None:
        """Returned string must have non-zero length after stripping."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        assert len(report.strip()) > 0

    def test_report_is_multiline(self) -> None:
        """Report must span more than one line."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        assert report.count("\n") >= 3

    def test_report_contains_model_name(self) -> None:
        """The model name must appear in the report."""
        results = _make_run_benchmark_results()
        results["model"] = "my-test-model-xyz"
        report = format_benchmark_report(results)
        assert "my-test-model-xyz" in report

    def test_report_contains_pair_ids(self) -> None:
        """At least some pair ids must appear in the formatted table."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        # Check a sample of ids
        assert "high_01" in report
        assert "low_06" in report

    def test_report_contains_statistics_section(self) -> None:
        """Report must include a Statistics section."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        assert "Statistics" in report or "stats" in report.lower()

    def test_report_contains_separation_score(self) -> None:
        """Report must mention the separation metric."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        assert "Separation" in report or "separation" in report.lower()

    def test_report_for_distinct_categories(self) -> None:
        """Report must show all three category labels (high/medium/low)."""
        results = _make_run_benchmark_results()
        report = format_benchmark_report(results)
        assert "high" in report.lower()
        assert "medium" in report.lower()
        assert "low" in report.lower()


# ---------------------------------------------------------------------------
# Tests: run_benchmark (mocked compare_sentences)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Tests for run_benchmark() using a mocked compare_sentences."""

    def test_returns_dict(self) -> None:
        """run_benchmark must return a dict."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        assert isinstance(results, dict)

    def test_returns_correct_top_level_keys(self) -> None:
        """run_benchmark result must contain 'model', 'pairs', and 'stats'."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        assert "model" in results
        assert "pairs" in results
        assert "stats" in results

    def test_model_key_is_string(self) -> None:
        """results['model'] must be a string."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        assert isinstance(results["model"], str)

    def test_default_model_name(self) -> None:
        """When model_name is None, model should be the default mpnet name."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        assert "paraphrase-multilingual-mpnet-base-v2" in results["model"]

    def test_custom_model_name_preserved(self) -> None:
        """Custom model_name should appear in the results dict."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark("my-custom-model")
        finally:
            ec.compare_sentences = orig
        assert results["model"] == "my-custom-model"

    def test_pairs_list_has_20_entries(self) -> None:
        """results['pairs'] must contain 20 scored entries."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        assert len(results["pairs"]) == 20

    def test_each_pair_has_score_key(self) -> None:
        """Every entry in results['pairs'] must contain a 'score' key."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        for pair in results["pairs"]:
            assert "score" in pair, f"Missing 'score' in pair: {pair}"

    def test_each_pair_score_is_float(self) -> None:
        """Every score in results['pairs'] must be a float."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        for pair in results["pairs"]:
            assert isinstance(pair["score"], float), (
                f"score is not float: {type(pair['score'])}"
            )

    def test_stats_has_required_keys(self) -> None:
        """results['stats'] must include high_avg, medium_avg, low_avg, separation_score."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        required_stat_keys = {"high_avg", "medium_avg", "low_avg", "separation_score"}
        actual_keys = set(results["stats"].keys())
        missing = required_stat_keys - actual_keys
        assert not missing, f"Stats missing keys: {missing}"

    def test_separation_score_equals_high_minus_low(self) -> None:
        """separation_score must equal high_avg - low_avg."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        stats = results["stats"]
        expected = stats["high_avg"] - stats["low_avg"]
        assert abs(stats["separation_score"] - expected) < 1e-9, (
            f"separation_score {stats['separation_score']:.6f} != "
            f"high_avg - low_avg {expected:.6f}"
        )

    def test_pair_entries_preserve_original_fields(self) -> None:
        """Each pair in results['pairs'] must carry id, sent1, sent2, category, description."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        expected_fields = {"id", "sent1", "sent2", "category", "description"}
        for i, pair in enumerate(results["pairs"]):
            missing = expected_fields - pair.keys()
            assert not missing, f"Pair {i} missing fields: {missing}"

    def test_run_benchmark_then_format_produces_report(self) -> None:
        """Pipeline: run_benchmark -> format_benchmark_report produces a non-empty string."""
        import embeddings_comparison as ec
        orig = ec.compare_sentences
        ec.compare_sentences = _mock_compare_sentences  # type: ignore[assignment]
        try:
            results = run_benchmark()
        finally:
            ec.compare_sentences = orig
        report = format_benchmark_report(results)
        assert isinstance(report, str)
        assert len(report.strip()) > 0

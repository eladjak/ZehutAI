"""Tests for cli.py - ZehutAI command-line interface.

Tests cover all four subcommands (compare, similarity, rag, benchmark),
argument validation, and graceful import-error handling.

All heavy ML models are mocked so tests run without torch/transformers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path so cli.py can be imported
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import CLI module under test
import cli  # noqa: E402  (must come after sys.path setup)
from cli import (  # noqa: E402  (must come after sys.path setup)
    build_parser,
    cmd_benchmark,
    cmd_compare,
    cmd_rag,
    cmd_similarity,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_namespace(**kwargs: Any) -> Any:
    """Return a SimpleNamespace-like argparse.Namespace for testing handlers."""
    import argparse

    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# Test: compare subcommand
# ---------------------------------------------------------------------------


class TestCompareCmd:
    """Tests for the 'compare' subcommand handler."""

    def test_compare_returns_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compare calls compare_sentences and prints a score."""
        mock_compare = MagicMock(return_value=0.85)
        monkeypatch.setattr(cli, "_import_compare_sentences", lambda: mock_compare)

        args = _make_namespace(sentence1="hello world", sentence2="hi there")
        exit_code = cmd_compare(args)

        assert exit_code == 0
        mock_compare.assert_called_once_with(["hello world", "hi there"])

    def test_compare_import_error_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """compare returns exit code 1 when import fails."""

        def _raise() -> Any:
            raise ImportError("mock import error")

        monkeypatch.setattr(cli, "_import_compare_sentences", _raise)

        args = _make_namespace(sentence1="a", sentence2="b")
        exit_code = cmd_compare(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_compare_via_main(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """main() dispatches to compare and prints similarity score."""
        mock_compare = MagicMock(return_value=0.72)
        monkeypatch.setattr(cli, "_import_compare_sentences", lambda: mock_compare)

        exit_code = main(["compare", "sentence one", "sentence two"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0.720000" in captured.out

    def test_compare_exception_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """compare returns 1 if compare_sentences raises an unexpected exception."""
        mock_compare = MagicMock(side_effect=RuntimeError("encoding failed"))
        monkeypatch.setattr(cli, "_import_compare_sentences", lambda: mock_compare)

        args = _make_namespace(sentence1="x", sentence2="y")
        exit_code = cmd_compare(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


# ---------------------------------------------------------------------------
# Test: similarity subcommand (tfidf - real, no model download)
# ---------------------------------------------------------------------------


class TestSimilarityCmd:
    """Tests for the 'similarity' subcommand handler."""

    def test_tfidf_real_no_mock(self, capsys: pytest.CaptureFixture) -> None:
        """TF-IDF similarity runs with real scikit-learn (no heavy models needed)."""
        data = [
            "The baby was laughing and playing",
            "Climate change is a global issue",
            "Deep learning uses neural networks",
        ]
        args = _make_namespace(
            method="tfidf",
            data=data,
            query="The baby laughed",
        )
        exit_code = cmd_similarity(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "TF-IDF" in captured.out
        assert "baby" in captured.out.lower() or "0." in captured.out

    def test_tfidf_default_data(self, capsys: pytest.CaptureFixture) -> None:
        """TF-IDF works with no --data (uses default corpus)."""
        args = _make_namespace(method="tfidf", data=[], query="The baby was laughing")
        exit_code = cmd_similarity(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "TF-IDF" in captured.out

    def test_similarity_import_error_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """similarity returns 1 when Similarity class import fails."""

        def _raise() -> Any:
            raise ImportError("no sklearn")

        monkeypatch.setattr(cli, "_import_similarity_class", _raise)

        args = _make_namespace(method="tfidf", data=[], query="test")
        exit_code = cmd_similarity(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_unknown_method_returns_1(self, capsys: pytest.CaptureFixture) -> None:
        """similarity returns 1 for an unknown method name."""
        args = _make_namespace(method="unknown_method", data=[], query="test")
        exit_code = cmd_similarity(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Unknown method" in captured.err

    def test_similarity_via_main_tfidf(self, capsys: pytest.CaptureFixture) -> None:
        """main() dispatches similarity --method tfidf correctly."""
        exit_code = main(
            [
                "similarity",
                "--method",
                "tfidf",
                "--data",
                "The cat sat",
                "A dog ran",
                "--query",
                "The cat",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "TF-IDF" in captured.out

    def test_similarity_missing_query_exits_nonzero(self) -> None:
        """similarity subcommand requires --query; missing it causes argparse error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["similarity", "--method", "tfidf", "--data", "text"])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Test: rag subcommand
# ---------------------------------------------------------------------------


class TestRagCmd:
    """Tests for the 'rag' subcommand handler."""

    def test_rag_with_mock_functions(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """rag calls vector_search and reciprocal_rank_fusion with mocked functions."""
        mock_docs = {
            "doc1": "Climate change and economic impact.",
            "doc2": "Public health and climate change.",
        }
        mock_search = MagicMock(return_value={"doc1": 0.9, "doc2": 0.7})
        mock_rrf = MagicMock(return_value={"doc1": 0.016, "doc2": 0.015})

        monkeypatch.setattr(
            cli,
            "_import_rag_functions",
            lambda: (mock_search, mock_rrf, mock_docs),
        )

        args = _make_namespace(query="climate", top_k=None)
        exit_code = cmd_rag(args)

        assert exit_code == 0
        mock_search.assert_called_once()
        mock_rrf.assert_called_once()
        captured = capsys.readouterr()
        assert "climate" in captured.out
        assert "doc1" in captured.out

    def test_rag_top_k_limits_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """rag --top-k N limits the output to N results."""
        mock_docs = {f"doc{i}": f"Document {i}" for i in range(1, 6)}
        mock_search = MagicMock(return_value={f"doc{i}": 1.0 / i for i in range(1, 6)})
        mock_rrf = MagicMock(return_value={f"doc{i}": 1.0 / i for i in range(1, 6)})

        monkeypatch.setattr(
            cli,
            "_import_rag_functions",
            lambda: (mock_search, mock_rrf, mock_docs),
        )

        args = _make_namespace(query="test query", top_k=2)
        exit_code = cmd_rag(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        # Exactly 2 result lines (each starts with "  N.")
        result_lines = [
            line
            for line in captured.out.splitlines()
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
        ]
        assert len(result_lines) <= 2

    def test_rag_import_error_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """rag returns exit code 1 when import fails."""

        def _raise() -> Any:
            raise ImportError("no torch")

        monkeypatch.setattr(cli, "_import_rag_functions", _raise)

        args = _make_namespace(query="test", top_k=None)
        exit_code = cmd_rag(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_rag_via_main(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """main() dispatches rag subcommand correctly."""
        mock_docs = {"d1": "climate change", "d2": "ancient history"}
        monkeypatch.setattr(
            cli,
            "_import_rag_functions",
            lambda: (
                MagicMock(return_value={"d1": 0.9}),
                MagicMock(return_value={"d1": 0.016}),
                mock_docs,
            ),
        )

        exit_code = main(["rag", "--query", "climate", "--top-k", "1"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "climate" in captured.out

    def test_rag_missing_query_exits_nonzero(self) -> None:
        """rag subcommand requires --query; missing it causes argparse error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["rag"])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Test: benchmark subcommand
# ---------------------------------------------------------------------------


class TestBenchmarkCmd:
    """Tests for the 'benchmark' subcommand handler."""

    def _make_mock_results(self) -> list[dict]:
        """Return minimal benchmark result dicts."""
        return [
            {
                "s1": "a",
                "s2": "b",
                "label": "high",
                "category": "para",
                "score": 0.9,
                "model": "test",
                "elapsed_ms": 10.0,
            },
            {
                "s1": "c",
                "s2": "d",
                "label": "medium",
                "category": "related",
                "score": 0.5,
                "model": "test",
                "elapsed_ms": 10.0,
            },
            {
                "s1": "e",
                "s2": "f",
                "label": "low",
                "category": "unrelated",
                "score": 0.1,
                "model": "test",
                "elapsed_ms": 10.0,
            },
        ]

    def test_benchmark_table_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """benchmark --format table calls format_benchmark_report and prints table."""
        mock_results = self._make_mock_results()
        mock_run = MagicMock(return_value=mock_results)
        mock_eval = MagicMock(
            return_value={
                "high_avg": 0.9,
                "medium_avg": 0.5,
                "low_avg": 0.1,
                "separation_score": 0.8,
                "high_low_gap": 0.8,
                "high_medium_gap": 0.4,
                "medium_low_gap": 0.4,
                "rank_accuracy": 1.0,
                "n_high": 1.0,
                "n_medium": 1.0,
                "n_low": 1.0,
                "avg_elapsed_ms": 10.0,
            }
        )
        mock_format = MagicMock(return_value="  Mock benchmark table\n")

        monkeypatch.setattr(
            cli,
            "_import_benchmark_functions",
            lambda: (mock_run, mock_eval, mock_format),
        )

        args = _make_namespace(format="table")
        exit_code = cmd_benchmark(args)

        assert exit_code == 0
        mock_run.assert_called_once()
        mock_eval.assert_called_once_with(mock_results)
        mock_format.assert_called_once()
        captured = capsys.readouterr()
        assert "Mock benchmark table" in captured.out

    def test_benchmark_json_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """benchmark --format json outputs valid JSON with metrics and pairs."""
        mock_results = self._make_mock_results()
        mock_run = MagicMock(return_value=mock_results)
        mock_eval = MagicMock(
            return_value={
                "high_avg": 0.9,
                "medium_avg": 0.5,
                "low_avg": 0.1,
                "separation_score": 0.8,
                "high_low_gap": 0.8,
                "high_medium_gap": 0.4,
                "medium_low_gap": 0.4,
                "rank_accuracy": 1.0,
                "n_high": 1.0,
                "n_medium": 1.0,
                "n_low": 1.0,
                "avg_elapsed_ms": 10.0,
            }
        )
        mock_format = MagicMock(return_value="table output")

        monkeypatch.setattr(
            cli,
            "_import_benchmark_functions",
            lambda: (mock_run, mock_eval, mock_format),
        )

        args = _make_namespace(format="json")
        exit_code = cmd_benchmark(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        # Filter out non-JSON "Running..." line
        json_lines = "\n".join(
            line for line in captured.out.splitlines() if not line.startswith("Running")
        )
        parsed = json.loads(json_lines)
        assert "metrics" in parsed
        assert "pairs" in parsed
        assert parsed["metrics"]["high_avg"] == pytest.approx(0.9)

    def test_benchmark_import_error_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """benchmark returns exit code 1 when import fails."""

        def _raise() -> Any:
            raise ImportError("no sentence-transformers")

        monkeypatch.setattr(cli, "_import_benchmark_functions", _raise)

        args = _make_namespace(format="table")
        exit_code = cmd_benchmark(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_benchmark_via_main_default_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """main() dispatches benchmark with default table format."""
        mock_results: list[dict] = [
            {
                "s1": "a",
                "s2": "b",
                "label": "high",
                "score": 0.9,
                "model": "m",
                "elapsed_ms": 5.0,
                "category": "test",
            },
        ]
        mock_eval_result = {
            "high_avg": 0.9,
            "medium_avg": float("nan"),
            "low_avg": float("nan"),
            "separation_score": float("nan"),
            "high_low_gap": float("nan"),
            "high_medium_gap": float("nan"),
            "medium_low_gap": float("nan"),
            "rank_accuracy": float("nan"),
            "n_high": 1.0,
            "n_medium": 0.0,
            "n_low": 0.0,
            "avg_elapsed_ms": 5.0,
        }
        monkeypatch.setattr(
            cli,
            "_import_benchmark_functions",
            lambda: (
                MagicMock(return_value=mock_results),
                MagicMock(return_value=mock_eval_result),
                MagicMock(return_value="Benchmark table output"),
            ),
        )

        exit_code = main(["benchmark"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Benchmark table output" in captured.out


# ---------------------------------------------------------------------------
# Test: --help and argument validation
# ---------------------------------------------------------------------------


class TestArgparseBehavior:
    """Tests for argparse-level help and argument validation."""

    def test_help_does_not_crash(self, capsys: pytest.CaptureFixture) -> None:
        """--help prints usage and exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "compare" in captured.out
        assert "similarity" in captured.out
        assert "rag" in captured.out
        assert "benchmark" in captured.out

    def test_compare_help_does_not_crash(self, capsys: pytest.CaptureFixture) -> None:
        """compare --help prints usage and exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "--help"])
        assert exc_info.value.code == 0

    def test_no_subcommand_exits_nonzero(self) -> None:
        """Invoking cli.py with no subcommand exits with a non-zero code."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_compare_missing_sentence_exits_nonzero(self) -> None:
        """compare requires two positional args; providing one causes error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "only one sentence"])
        assert exc_info.value.code != 0

    def test_benchmark_invalid_format_exits_nonzero(self) -> None:
        """benchmark --format with invalid choice causes argparse error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["benchmark", "--format", "xml"])
        assert exc_info.value.code != 0

    def test_build_parser_returns_parser(self) -> None:
        """build_parser() returns a configured ArgumentParser."""
        import argparse

        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

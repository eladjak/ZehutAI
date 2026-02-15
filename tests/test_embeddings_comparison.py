"""Tests for the embeddings_comparison module.

These tests mock the SentenceTransformer model so they run fast
without downloading model weights.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import embeddings_comparison as ec


# ---------------------------------------------------------------------------
# Unit tests (mocked model)
# ---------------------------------------------------------------------------

class TestCompareSentences:
    """Tests for compare_sentences() with mocked model."""

    def test_returns_float(
        self, mock_sentence_transformer: MagicMock, sample_sentences: list[str]
    ) -> None:
        """compare_sentences should return a float."""
        result = ec.compare_sentences(sample_sentences)
        assert isinstance(result, float)

    def test_similarity_in_range(
        self, mock_sentence_transformer: MagicMock, sample_sentences: list[str]
    ) -> None:
        """Cosine similarity should be between -1 and 1."""
        result = ec.compare_sentences(sample_sentences)
        assert -1.0 <= result <= 1.0

    def test_different_sentence_pairs_give_different_scores(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Different sentence pairs should produce different similarity scores."""
        result_a = ec.compare_sentences(["cat on mat", "dog on rug"])
        result_b = ec.compare_sentences(["quantum physics", "baking recipes"])
        # With seeded random embeddings, different inputs produce different scores
        assert result_a != result_b

    def test_rejects_single_sentence(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Should raise ValueError when given only 1 sentence."""
        with pytest.raises(ValueError, match="Expected exactly 2 sentences"):
            ec.compare_sentences(["only one"])

    def test_rejects_three_sentences(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Should raise ValueError when given 3 sentences."""
        with pytest.raises(ValueError, match="Expected exactly 2 sentences"):
            ec.compare_sentences(["a", "b", "c"])

    def test_rejects_empty_list(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="Expected exactly 2 sentences"):
            ec.compare_sentences([])


class TestModelCaching:
    """Tests for the lazy model loading / caching mechanism."""

    def test_get_model_returns_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_get_model should return the same object on second call."""
        mock = MagicMock()
        monkeypatch.setattr(ec, "_model", mock)
        assert ec._get_model() is mock

    def test_get_model_creates_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_get_model should create model when _model is None."""
        monkeypatch.setattr(ec, "_model", None)
        mock_cls = MagicMock()
        monkeypatch.setattr(ec, "SentenceTransformer", mock_cls)
        result = ec._get_model()
        mock_cls.assert_called_once_with(ec.MODEL_NAME)
        assert result is mock_cls.return_value


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_model_name_is_multilingual_mpnet(self) -> None:
        """MODEL_NAME should reference the multilingual mpnet model."""
        assert "multilingual" in ec.MODEL_NAME
        assert "mpnet" in ec.MODEL_NAME

    def test_model_attr_exists(self) -> None:
        """_model attribute should exist on the module."""
        assert hasattr(ec, "_model")

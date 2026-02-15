"""Tests for the Similarity class in sim.py.

Tests that don't require model downloads use mocking or pure-logic methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import DEFAULT_DATA, DEFAULT_QUERY, Similarity


# ---------------------------------------------------------------------------
# Similarity class instantiation
# ---------------------------------------------------------------------------

class TestSimilarityInit:
    """Tests for Similarity class construction."""

    def test_init_empty_models(self) -> None:
        """New Similarity should have no registered NN models."""
        sim = Similarity()
        assert sim.nnModels == []

    def test_init_empty_results(self) -> None:
        """New Similarity should have empty results dict."""
        sim = Similarity()
        assert sim.results == {}

    def test_init_empty_texts(self) -> None:
        """New Similarity should have empty texts list."""
        sim = Similarity()
        assert sim.texts == []

    def test_init_empty_methods(self) -> None:
        """New Similarity should have empty methods list."""
        sim = Similarity()
        assert sim.methods == []


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

class TestModelRegistration:
    """Tests for addModel / removeModel."""

    def test_add_model(self) -> None:
        """addModel should append to nnModels list."""
        sim = Similarity()
        sim.addModel("tok", "mod", "weights-v1")
        assert len(sim.nnModels) == 1
        assert sim.nnModels[0] == ("tok", "mod", "weights-v1")

    def test_add_multiple_models(self) -> None:
        """addModel called twice should produce 2 entries."""
        sim = Similarity()
        sim.addModel("tok1", "mod1", "w1")
        sim.addModel("tok2", "mod2", "w2")
        assert len(sim.nnModels) == 2

    def test_remove_model_by_weights(self) -> None:
        """removeModel should remove matching model."""
        sim = Similarity()
        sim.addModel("tok1", "mod1", "w1")
        sim.addModel("tok2", "mod2", "w2")
        sim.removeModel("w1")
        assert len(sim.nnModels) == 1
        assert sim.nnModels[0][2] == "w2"

    def test_remove_nonexistent_model(self) -> None:
        """removeModel should be a no-op for unknown weights."""
        sim = Similarity()
        sim.addModel("tok", "mod", "w1")
        sim.removeModel("does-not-exist")
        assert len(sim.nnModels) == 1

    def test_remove_first_match_only(self) -> None:
        """removeModel should only remove the first matching entry."""
        sim = Similarity()
        sim.addModel("tok1", "mod1", "w1")
        sim.addModel("tok2", "mod2", "w1")  # duplicate weights
        sim.removeModel("w1")
        assert len(sim.nnModels) == 1
        # Second entry (tok2) should remain
        assert sim.nnModels[0][0] == "tok2"


# ---------------------------------------------------------------------------
# add_texts method
# ---------------------------------------------------------------------------

class TestAddTexts:
    """Tests for the add_texts method."""

    def test_add_texts_sets_list(self) -> None:
        """add_texts with a list should replace self.texts."""
        sim = Similarity()
        sim.add_texts(["hello", "world"])
        assert sim.texts == ["hello", "world"]

    def test_add_texts_none_is_noop(self) -> None:
        """add_texts(None) should not change self.texts."""
        sim = Similarity()
        sim.texts = ["existing"]
        sim.add_texts(None)
        assert sim.texts == ["existing"]


# ---------------------------------------------------------------------------
# TF-IDF method (fast, no model download needed)
# ---------------------------------------------------------------------------

class TestMethodScikitlearn:
    """Tests for the TF-IDF based similarity method."""

    def test_returns_list(self, sample_data: list[str], sample_query: str) -> None:
        """methodScikitlearn should return a list."""
        sim = Similarity()
        result = sim.methodScikitlearn(data=sample_data, query=sample_query)
        assert isinstance(result, list)

    def test_result_length_matches_data(
        self, sample_data: list[str], sample_query: str
    ) -> None:
        """Should return one result per document in data."""
        sim = Similarity()
        result = sim.methodScikitlearn(data=sample_data, query=sample_query)
        assert len(result) == len(sample_data)

    def test_result_tuples_format(
        self, sample_data: list[str], sample_query: str
    ) -> None:
        """Each result should be (text, similarity_score)."""
        sim = Similarity()
        result = sim.methodScikitlearn(data=sample_data, query=sample_query)
        for text, score in result:
            assert isinstance(text, str)
            assert isinstance(score, float)

    def test_similarity_scores_in_range(
        self, sample_data: list[str], sample_query: str
    ) -> None:
        """TF-IDF cosine similarity should be between 0 and 1."""
        sim = Similarity()
        result = sim.methodScikitlearn(data=sample_data, query=sample_query)
        for _text, score in result:
            assert 0.0 <= score <= 1.0

    def test_identical_text_high_similarity(self) -> None:
        """Identical text and query should produce high similarity."""
        sim = Similarity()
        text = "The quick brown fox jumps over the lazy dog"
        result = sim.methodScikitlearn(data=[text], query=text)
        assert result[0][1] > 0.99

    def test_unrelated_text_low_similarity(self) -> None:
        """Completely unrelated text should have low similarity."""
        sim = Similarity()
        result = sim.methodScikitlearn(
            data=["quantum physics experiments"],
            query="chocolate cake recipe baking",
        )
        assert result[0][1] < 0.3

    def test_uses_default_data_when_none(self) -> None:
        """Should use DEFAULT_DATA when data=None."""
        sim = Similarity()
        result = sim.methodScikitlearn(data=None)
        assert len(result) == len(DEFAULT_DATA)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Tests for static/helper methods on Similarity."""

    def test_squared_sum_known_vector(self) -> None:
        """squared_sum([3, 4]) should be 5.0 (L2 norm)."""
        result = Similarity.squared_sum([3.0, 4.0])
        assert result == 5.0

    def test_squared_sum_unit_vector(self) -> None:
        """squared_sum of a unit vector should be ~1.0."""
        result = Similarity.squared_sum([1.0, 0.0, 0.0])
        assert result == 1.0

    def test_squared_sum_zeros(self) -> None:
        """squared_sum of zero vector should be 0.0."""
        result = Similarity.squared_sum([0.0, 0.0])
        assert result == 0.0

    def test_cos_similarity_identical(self) -> None:
        """Cosine similarity of identical vectors should be 1.0."""
        sim = Similarity()
        vec = [1.0, 2.0, 3.0]
        result = sim.cos_similarity(vec, vec)
        assert result == 1.0

    def test_cos_similarity_orthogonal(self) -> None:
        """Cosine similarity of orthogonal vectors should be 0.0."""
        sim = Similarity()
        result = sim.cos_similarity([1.0, 0.0], [0.0, 1.0])
        assert result == 0.0

    def test_cos_similarity_opposite(self) -> None:
        """Cosine similarity of opposite vectors should be -1.0."""
        sim = Similarity()
        result = sim.cos_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == -1.0

    def test_cos_similarity_numpy_arrays(self) -> None:
        """cos_similarity should work with numpy arrays."""
        sim = Similarity()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = sim.cos_similarity(a, b)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

class TestDefaults:
    """Tests for module-level default constants."""

    def test_default_data_non_empty(self) -> None:
        """DEFAULT_DATA should contain at least 2 documents."""
        assert len(DEFAULT_DATA) >= 2

    def test_default_query_non_empty(self) -> None:
        """DEFAULT_QUERY should be a non-empty string."""
        assert len(DEFAULT_QUERY) > 0

    def test_default_data_all_strings(self) -> None:
        """All entries in DEFAULT_DATA should be strings."""
        for item in DEFAULT_DATA:
            assert isinstance(item, str)

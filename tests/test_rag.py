"""Tests for the RAG pipeline in rag.py.

Tests for vector_search and reciprocal_rank_fusion do NOT require
model downloads. Tests for generate_queries would need DictaLM 2.0
and are not included here (would need @pytest.mark.slow + integration env).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag import (
    ALL_DOCUMENTS,
    chunk_documents,
    generate_output,
    preprocess_text,
    reciprocal_rank_fusion,
    vector_search,
)


# ---------------------------------------------------------------------------
# preprocess_text tests
# ---------------------------------------------------------------------------

class TestPreprocessText:
    """Tests for text normalization."""

    def test_strips_whitespace(self) -> None:
        """Should strip leading and trailing whitespace."""
        assert preprocess_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        """Should collapse multiple spaces to one."""
        assert preprocess_text("hello    world") == "hello world"

    def test_normalizes_tabs_and_newlines(self) -> None:
        """Should normalize tabs and newlines to single spaces."""
        assert preprocess_text("hello\t\tworld\n\nfoo") == "hello world foo"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert preprocess_text("") == ""

    def test_single_word(self) -> None:
        """Single word with no extra whitespace should remain unchanged."""
        assert preprocess_text("word") == "word"

    def test_already_clean(self) -> None:
        """Already clean text should pass through unchanged."""
        assert preprocess_text("hello world") == "hello world"


# ---------------------------------------------------------------------------
# chunk_documents tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    """Tests for document chunking with overlap."""

    def test_short_doc_single_chunk(self) -> None:
        """Document shorter than chunk_size should produce one chunk."""
        docs = {"d1": "short text here"}
        chunks = chunk_documents(docs, chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0]["source_id"] == "d1"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["chunk_id"] == "d1"

    def test_long_doc_multiple_chunks(self) -> None:
        """Document longer than chunk_size should produce multiple chunks."""
        docs = {"d1": " ".join(f"word{i}" for i in range(250))}
        chunks = chunk_documents(docs, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        assert all(c["source_id"] == "d1" for c in chunks)

    def test_chunk_ids_sequential(self) -> None:
        """Chunk indices should be sequential starting from 0."""
        docs = {"d1": " ".join(f"w{i}" for i in range(300))}
        chunks = chunk_documents(docs, chunk_size=100, overlap=20)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_id_format(self) -> None:
        """Multi-chunk docs should use 'source_chunkN' format."""
        docs = {"doc1": " ".join(f"w{i}" for i in range(250))}
        chunks = chunk_documents(docs, chunk_size=100, overlap=20)
        assert chunks[0]["chunk_id"] == "doc1_chunk0"
        assert chunks[1]["chunk_id"] == "doc1_chunk1"

    def test_overlap_content(self) -> None:
        """Chunks should overlap by the specified number of words."""
        words = [f"w{i}" for i in range(20)]
        docs = {"d1": " ".join(words)}
        chunks = chunk_documents(docs, chunk_size=10, overlap=3)
        # First chunk: w0..w9, Second chunk should start at w7
        chunk1_words = chunks[0]["text"].split()
        chunk2_words = chunks[1]["text"].split()
        overlap_words = set(chunk1_words[-3:]) & set(chunk2_words[:3])
        assert len(overlap_words) == 3

    def test_multiple_documents(self) -> None:
        """Should process multiple documents independently."""
        docs = {
            "d1": "short doc",
            "d2": " ".join(f"w{i}" for i in range(200)),
        }
        chunks = chunk_documents(docs, chunk_size=100, overlap=20)
        d1_chunks = [c for c in chunks if c["source_id"] == "d1"]
        d2_chunks = [c for c in chunks if c["source_id"] == "d2"]
        assert len(d1_chunks) == 1
        assert len(d2_chunks) >= 2

    def test_overlap_must_be_less_than_chunk_size(self) -> None:
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            chunk_documents({"d1": "text"}, chunk_size=10, overlap=10)

    def test_empty_documents(self) -> None:
        """Empty input should return empty list."""
        assert chunk_documents({}) == []

    def test_preprocesses_text(self) -> None:
        """Chunk text should be preprocessed (normalized whitespace)."""
        docs = {"d1": "  hello   world  "}
        chunks = chunk_documents(docs, chunk_size=100, overlap=10)
        assert chunks[0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# vector_search with top_k tests
# ---------------------------------------------------------------------------

class TestVectorSearchTopK:
    """Tests for vector_search top_k parameter."""

    @patch("rag.compare_sentences")
    def test_top_k_limits_results(self, mock_compare: MagicMock) -> None:
        """top_k should limit the number of returned results."""
        scores = iter([0.9, 0.7, 0.5, 0.3, 0.1])
        mock_compare.side_effect = lambda _: next(scores)
        docs = {f"d{i}": f"text{i}" for i in range(5)}
        result = vector_search("query", docs, top_k=3)
        assert len(result) == 3

    @patch("rag.compare_sentences")
    def test_top_k_returns_highest(self, mock_compare: MagicMock) -> None:
        """top_k should return the highest scoring documents."""
        mock_compare.side_effect = lambda args: {"d1": 0.1, "d2": 0.9, "d3": 0.5}[args[0]]
        docs = {"d1": "a", "d2": "b", "d3": "c"}
        result = vector_search("query", docs, top_k=2)
        assert "d2" in result
        assert len(result) == 2

    @patch("rag.compare_sentences")
    def test_top_k_none_returns_all(self, mock_compare: MagicMock) -> None:
        """top_k=None (default) should return all results."""
        mock_compare.return_value = 0.5
        docs = {"d1": "a", "d2": "b", "d3": "c"}
        result = vector_search("query", docs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests (pure logic, no models)
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Tests for the Reciprocal Rank Fusion algorithm."""

    def test_single_query_single_doc(self) -> None:
        """Single query with single doc should produce a single fused score."""
        search_results = {"q1": {"doc1": 0.9}}
        result = reciprocal_rank_fusion(search_results)
        assert "doc1" in result
        assert len(result) == 1

    def test_single_query_multiple_docs(self) -> None:
        """All documents should appear in fused results."""
        search_results = {
            "q1": {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5}
        }
        result = reciprocal_rank_fusion(search_results)
        assert len(result) == 3
        assert set(result.keys()) == {"doc1", "doc2", "doc3"}

    def test_fused_scores_are_positive(self) -> None:
        """All RRF scores should be positive."""
        search_results = {
            "q1": {"doc1": 0.9, "doc2": 0.3},
            "q2": {"doc1": 0.4, "doc2": 0.8},
        }
        result = reciprocal_rank_fusion(search_results)
        for score in result.values():
            assert score > 0

    def test_results_sorted_descending(self) -> None:
        """Output should be sorted by fused score descending."""
        search_results = {
            "q1": {"doc1": 0.9, "doc2": 0.3, "doc3": 0.5},
            "q2": {"doc1": 0.2, "doc2": 0.8, "doc3": 0.6},
        }
        result = reciprocal_rank_fusion(search_results)
        scores = list(result.values())
        assert scores == sorted(scores, reverse=True)

    def test_doc_appearing_in_multiple_queries_gets_higher_score(self) -> None:
        """A doc ranked #1 in both queries should beat a doc ranked #1 in one."""
        search_results = {
            "q1": {"doc_a": 0.95, "doc_b": 0.1},
            "q2": {"doc_a": 0.90, "doc_b": 0.2},
        }
        result = reciprocal_rank_fusion(search_results)
        assert result["doc_a"] > result["doc_b"]

    def test_k_parameter_changes_scores(self) -> None:
        """Different k values should produce different absolute scores."""
        search_results = {"q1": {"doc1": 0.9, "doc2": 0.5}}
        result_k60 = reciprocal_rank_fusion(search_results, k=60)
        result_k10 = reciprocal_rank_fusion(search_results, k=10)
        # k=10 gives higher absolute scores (1/(rank+10) > 1/(rank+60))
        assert result_k10["doc1"] > result_k60["doc1"]

    def test_empty_search_results(self) -> None:
        """Empty input should return empty dict."""
        result = reciprocal_rank_fusion({})
        assert result == {}

    def test_query_with_no_docs(self) -> None:
        """Query with empty doc_scores should not crash."""
        result = reciprocal_rank_fusion({"q1": {}})
        assert result == {}

    def test_rrf_score_formula(self) -> None:
        """Verify the exact RRF score formula: sum of 1/(rank + k)."""
        search_results = {"q1": {"doc1": 0.9}}  # rank=0 for doc1
        k = 60
        result = reciprocal_rank_fusion(search_results, k=k)
        expected = 1.0 / (0 + k)  # rank 0 + k
        assert abs(result["doc1"] - expected) < 1e-10

    def test_multi_query_rrf_accumulation(self) -> None:
        """RRF should accumulate scores across queries."""
        # doc1 is rank 0 in both queries
        search_results = {
            "q1": {"doc1": 0.9},
            "q2": {"doc1": 0.8},
        }
        k = 60
        result = reciprocal_rank_fusion(search_results, k=k)
        expected = 2.0 / (0 + k)  # rank 0 in both queries
        assert abs(result["doc1"] - expected) < 1e-10


# ---------------------------------------------------------------------------
# vector_search tests (mocked compare_sentences)
# ---------------------------------------------------------------------------

class TestVectorSearch:
    """Tests for vector_search with mocked similarity."""

    @patch("rag.compare_sentences")
    def test_returns_dict(
        self, mock_compare: MagicMock, sample_documents: dict[str, str]
    ) -> None:
        """vector_search should return a dict."""
        mock_compare.return_value = 0.5
        result = vector_search("test query", sample_documents)
        assert isinstance(result, dict)

    @patch("rag.compare_sentences")
    def test_all_docs_scored(
        self, mock_compare: MagicMock, sample_documents: dict[str, str]
    ) -> None:
        """Every document should have a score."""
        mock_compare.return_value = 0.42
        result = vector_search("test query", sample_documents)
        assert len(result) == len(sample_documents)

    @patch("rag.compare_sentences")
    def test_results_sorted_descending(
        self, mock_compare: MagicMock
    ) -> None:
        """Results should be sorted by score descending."""
        # Return different scores for different docs
        scores = iter([0.3, 0.9, 0.6])
        mock_compare.side_effect = lambda _: next(scores)

        docs = {"doc1": "a", "doc2": "b", "doc3": "c"}
        result = vector_search("query", docs)
        result_scores = list(result.values())
        assert result_scores == sorted(result_scores, reverse=True)

    @patch("rag.compare_sentences")
    def test_calls_compare_for_each_doc(
        self, mock_compare: MagicMock
    ) -> None:
        """compare_sentences should be called once per document."""
        mock_compare.return_value = 0.5
        docs = {"d1": "a", "d2": "b", "d3": "c"}
        vector_search("query", docs)
        assert mock_compare.call_count == 3

    @patch("rag.compare_sentences")
    def test_empty_documents(self, mock_compare: MagicMock) -> None:
        """Empty documents dict should return empty result."""
        result = vector_search("query", {})
        assert result == {}
        mock_compare.assert_not_called()


# ---------------------------------------------------------------------------
# generate_output tests (pure string formatting)
# ---------------------------------------------------------------------------

class TestGenerateOutput:
    """Tests for generate_output."""

    def test_returns_string(self) -> None:
        """generate_output should return a string."""
        result = generate_output({"doc1": 0.5}, ["q1"])
        assert isinstance(result, str)

    def test_includes_query_info(self) -> None:
        """Output should mention the queries."""
        result = generate_output({"doc1": 0.5}, ["climate change"])
        assert "climate change" in result

    def test_includes_doc_keys(self) -> None:
        """Output should mention the document keys."""
        result = generate_output({"my_doc": 0.5}, ["q"])
        assert "my_doc" in result

    def test_empty_results(self) -> None:
        """Should handle empty results without error."""
        result = generate_output({}, [])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ALL_DOCUMENTS constant
# ---------------------------------------------------------------------------

class TestAllDocuments:
    """Tests for the predefined documents constant."""

    def test_has_10_documents(self) -> None:
        """ALL_DOCUMENTS should contain 10 entries."""
        assert len(ALL_DOCUMENTS) == 10

    def test_keys_are_doc_ids(self) -> None:
        """Keys should follow 'docN' naming convention."""
        for key in ALL_DOCUMENTS:
            assert key.startswith("doc")

    def test_values_are_strings(self) -> None:
        """All values should be non-empty strings."""
        for value in ALL_DOCUMENTS.values():
            assert isinstance(value, str)
            assert len(value) > 0

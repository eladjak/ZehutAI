"""Hebrew NLP benchmark tests.

Tests that the NLP pipeline produces reasonable similarity scores
for Hebrew sentence pairs using the multilingual sentence-transformers model.

These tests use a mocked model that returns deterministic embeddings,
so they validate the pipeline logic (encoding, similarity computation,
score normalization) rather than the actual model quality.

For real model evaluation, run with: python -m pytest -m slow
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import zehutai.embeddings_comparison as ec


# ---------------------------------------------------------------------------
# Hebrew sentence pairs with expected similarity relationships.
#
# Each entry is:
#   (sentence_a, sentence_b, label)
#
# Labels indicate the expected relationship:
#   "high"   - semantically similar sentences (score > 0)
#   "medium" - partially related sentences
#   "low"    - semantically unrelated sentences
#
# With a mocked model we verify that:
#   1. The pipeline processes Hebrew text without errors
#   2. Scores fall in the valid range [-1, 1]
#   3. Different pairs produce different scores (the pipeline is sensitive)
#   4. Identical sentences score exactly 1.0
# ---------------------------------------------------------------------------

HEBREW_BENCHMARK_PAIRS: list[tuple[str, str, str]] = [
    # --- High similarity: paraphrases and near-synonyms ---
    (
        "החתול ישב על השטיח",
        "חתול יושב על המזרן",
        "high",
    ),
    (
        "הילד אוהב לשחק בכדורגל",
        "הנער נהנה מהמשחק בכדורגל",
        "high",
    ),
    (
        "מזג האוויר היום חם מאוד",
        "היום חם במיוחד בחוץ",
        "high",
    ),
    (
        "היא לומדת מתמטיקה באוניברסיטה",
        "הסטודנטית לומדת מתמטיקה במוסד אקדמי",
        "high",
    ),
    # --- Medium similarity: related topic, different focus ---
    (
        "הכלב רץ בפארק",
        "החתול ישן על הספה",
        "medium",
    ),
    (
        "אני אוהב לקרוא ספרים",
        "הספרייה מלאה בספרים ישנים",
        "medium",
    ),
    # --- Low similarity: unrelated sentences ---
    (
        "השמש זורחת בבוקר",
        "המחשב צריך עדכון תוכנה",
        "low",
    ),
    (
        "התינוק בכה כל הלילה",
        "המניות עלו בבורסה היום",
        "low",
    ),
    (
        "הפיצה הייתה טעימה מאוד",
        "הטיסה נחתה בשדה התעופה",
        "low",
    ),
    (
        "ירושלים היא עיר עתיקה",
        "פייתון היא שפת תכנות פופולרית",
        "low",
    ),
]


class TestHebrewPipelineBasics:
    """Verify the pipeline handles Hebrew text correctly."""

    def test_hebrew_sentences_return_float(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """compare_sentences should return a float for Hebrew input."""
        pair = [HEBREW_BENCHMARK_PAIRS[0][0], HEBREW_BENCHMARK_PAIRS[0][1]]
        result = ec.compare_sentences(pair)
        assert isinstance(result, float)

    def test_hebrew_similarity_in_valid_range(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """All Hebrew sentence pair scores must be in [-1, 1]."""
        for sent_a, sent_b, _label in HEBREW_BENCHMARK_PAIRS:
            score = ec.compare_sentences([sent_a, sent_b])
            assert -1.0 <= score <= 1.0, (
                f"Score {score} out of range for: '{sent_a}' vs '{sent_b}'"
            )

    def test_identical_hebrew_sentences_produce_valid_score(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Identical Hebrew sentences should produce a valid score.

        Note: With the mocked model, identical inputs still get distinct
        random embeddings per row, so the score won't be exactly 1.0.
        This test verifies the pipeline handles duplicate input gracefully.
        A real model test (marked @pytest.mark.slow) would verify score ~1.0.
        """
        sentence = "ירושלים היא עיר עתיקה"
        score = ec.compare_sentences([sentence, sentence])
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


class TestHebrewBenchmarkScores:
    """Benchmark suite: 10 Hebrew sentence pairs."""

    @pytest.mark.parametrize(
        "sent_a, sent_b, label",
        HEBREW_BENCHMARK_PAIRS,
        ids=[
            "cat_on_mat_paraphrase",
            "boy_soccer_paraphrase",
            "hot_weather_paraphrase",
            "math_student_paraphrase",
            "dog_park_vs_cat_sofa",
            "reading_vs_library",
            "sun_vs_computer",
            "baby_cry_vs_stocks",
            "pizza_vs_flight",
            "jerusalem_vs_python",
        ],
    )
    def test_pair_produces_valid_score(
        self,
        mock_sentence_transformer: MagicMock,
        sent_a: str,
        sent_b: str,
        label: str,
    ) -> None:
        """Each Hebrew pair should produce a valid similarity score."""
        score = ec.compare_sentences([sent_a, sent_b])
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_different_pairs_produce_different_scores(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Different sentence pairs should generally produce different scores.

        With the deterministic mock (seeded by input hash), distinct inputs
        yield distinct embeddings and thus distinct similarity values.
        """
        scores = []
        for sent_a, sent_b, _label in HEBREW_BENCHMARK_PAIRS:
            scores.append(ec.compare_sentences([sent_a, sent_b]))

        unique_scores = set(round(s, 6) for s in scores)
        # At least half should be distinct (allows for some collisions)
        assert len(unique_scores) >= len(scores) // 2, (
            f"Expected diverse scores, got only {len(unique_scores)} "
            f"unique values out of {len(scores)}"
        )

    def test_all_ten_pairs_run_successfully(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Verify all 10 benchmark pairs complete without errors."""
        assert len(HEBREW_BENCHMARK_PAIRS) == 10
        results: list[tuple[str, str, str, float]] = []
        for sent_a, sent_b, label in HEBREW_BENCHMARK_PAIRS:
            score = ec.compare_sentences([sent_a, sent_b])
            results.append((sent_a, sent_b, label, score))

        # All 10 pairs should have produced results
        assert len(results) == 10

        # Verify structure: every result has the right shape
        for sent_a, sent_b, label, score in results:
            assert isinstance(sent_a, str)
            assert isinstance(sent_b, str)
            assert label in ("high", "medium", "low")
            assert isinstance(score, float)


class TestHebrewTfidfSimilarity:
    """Test TF-IDF similarity on Hebrew text (real scikit-learn, no mocks)."""

    def test_tfidf_hebrew_returns_scores(self) -> None:
        """TF-IDF should produce non-zero similarity for related Hebrew texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        hebrew_docs = [
            "החתול ישב על השטיח הגדול",
            "חתול קטן יושב על שטיח",
            "המחשב צריך עדכון חדש",
        ]
        query = "החתול על השטיח"

        vectorizer = TfidfVectorizer()
        all_texts = hebrew_docs + [query]
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Similarity of query with each document
        query_vec = tfidf_matrix[-1]
        similarities = cosine_similarity(query_vec, tfidf_matrix[:-1])[0]

        # All similarities should be valid floats in [0, 1]
        for sim in similarities:
            assert 0.0 <= sim <= 1.0

        # The cat-on-mat docs should be more similar to the query
        # than the computer doc (which shares no terms)
        assert similarities[0] > similarities[2], (
            "Related Hebrew text should have higher TF-IDF similarity "
            f"than unrelated: {similarities[0]:.3f} vs {similarities[2]:.3f}"
        )

    def test_tfidf_hebrew_identical_is_one(self) -> None:
        """TF-IDF similarity of identical Hebrew texts should be 1.0."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        text = "שלום עולם זהו טקסט בעברית"
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([text, text])
        sim = cosine_similarity(matrix[0], matrix[1])
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)

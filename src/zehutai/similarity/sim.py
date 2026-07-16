"""Similarity comparison class supporting multiple NLP methods.

Compares text similarity using NLTK/Doc2Vec, TF-IDF (scikit-learn),
BERT, and RoBERTa embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import transformers
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, RobertaTokenizer

# Module-level model cache: avoids re-downloading on every method call
_model_cache: dict[str, tuple[Any, Any]] = {}


def _get_or_load_model(
    tokenizer_cls: Any,
    model_cls: Any,
    model_weights: str,
) -> tuple[Any, Any]:
    """Load and cache a tokenizer/model pair by weights identifier.

    Args:
        tokenizer_cls: Tokenizer class (e.g., BertTokenizer).
        model_cls: Model class (e.g., transformers.BertModel).
        model_weights: Pretrained weights identifier.

    Returns:
        Tuple of (tokenizer_instance, model_instance).
    """
    if model_weights not in _model_cache:
        tokenizer = tokenizer_cls.from_pretrained(model_weights, clean_up_tokenization_spaces=True)
        model = model_cls.from_pretrained(model_weights)
        _model_cache[model_weights] = (tokenizer, model)
    return _model_cache[model_weights]


# Default sample data used across methods
DEFAULT_DATA: list[str] = [
    "The movie is awesome. It was a good thriller",
    "We are learning NLP through GeeksforGeeks",
    "The baby learned to walk in the 5th month itself",
]

DEFAULT_QUERY: str = "The baby was laughing and playing"


class Similarity:
    """Multi-method text similarity comparison engine.

    Supports Doc2Vec, TF-IDF, BERT, and RoBERTa similarity methods.
    Neural network models can be registered and run in batch.
    """

    def __init__(self) -> None:
        self.nnModels: list[tuple[Any, Any, str]] = []
        self.texts: list[str] = []
        self.methods: list[Any] = []
        self.results: dict[str, list[tuple[float, str, str]]] = {}

    def addModel(self, tokenizer: Any, model: Any, model_weights: str) -> None:
        """Register a neural network model for batch comparison.

        Args:
            tokenizer: Tokenizer class (e.g., BertTokenizer).
            model: Model class (e.g., BertModel).
            model_weights: Pretrained weights identifier (e.g., 'bert-base-uncased').
        """
        self.nnModels.append((tokenizer, model, model_weights))

    def removeModel(self, target_weights: str) -> None:
        """Remove a registered model by its weights identifier.

        Args:
            target_weights: The pretrained weights identifier to remove.
        """
        for i in range(len(self.nnModels)):
            _tokenizer, _model, model_weights = self.nnModels[i]
            if model_weights == target_weights:
                self.nnModels.pop(i)
                break

    def add_texts(self, texts: list[str] | None = None) -> None:
        """Set custom texts for comparison.

        Args:
            texts: List of texts to use in comparisons.
        """
        if texts is not None:
            self.texts = texts

    def compareMethods(self) -> None:
        """Run all registered comparison methods and print results."""
        for method in self.methods:
            result = method()
            print(result)

    def methodNLTK(
        self,
        data: list[str] | None = None,
        query: str = DEFAULT_QUERY,
    ) -> list[tuple[str, float]]:
        """Compare texts using Doc2Vec trained on the provided data.

        Args:
            data: Source documents to train Doc2Vec on. Defaults to sample data.
            query: Query text to find similar documents for.

        Returns:
            List of (document_index, similarity_score) tuples.
        """
        if data is None:
            data = DEFAULT_DATA

        tokenized_data = [word_tokenize(document.lower()) for document in data]
        tagged_data = [
            TaggedDocument(words=words, tags=[str(idx)]) for idx, words in enumerate(tokenized_data)
        ]

        model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=1000)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        inferred_vector = model.infer_vector(word_tokenize(query.lower()))
        similar_documents = model.dv.most_similar([inferred_vector], topn=len(model.dv))

        return similar_documents

    def methodScikitlearn(
        self,
        data: list[str] | None = None,
        query: str = DEFAULT_QUERY,
    ) -> list[tuple[str, float]]:
        """Compare texts using TF-IDF cosine similarity.

        Args:
            data: Source documents to compare against. Defaults to sample data.
            query: Query text to compare.

        Returns:
            List of (text, similarity_score) tuples.
        """
        if data is None:
            data = DEFAULT_DATA

        vectorizer = TfidfVectorizer()
        results: list[tuple[str, float]] = []
        for text in data:
            vectors = vectorizer.fit_transform([text, query])
            similarity = cosine_similarity(vectors)
            results.append((text, float(similarity[1, 0])))
        return results

    def methodNNEmbeddings(
        self,
        tokenizer: Any,
        model: Any,
        model_weights: str,
        data: list[str] | None = None,
        query: str = DEFAULT_QUERY,
    ) -> list[tuple[float, str, str]]:
        """Find distances between query and source texts using NN embeddings.

        Args:
            tokenizer: The tokenizer class (e.g., BertTokenizer).
            model: The model class (e.g., BertModel).
            model_weights: Pretrained weights identifier (e.g., 'bert-base-uncased').
            data: Source texts to compare against. Defaults to sample data.
            query: Query text to compare.

        Returns:
            List of (similarity, source_text, query_text) tuples.
        """
        if data is None:
            data = DEFAULT_DATA

        tokenizer, model = _get_or_load_model(tokenizer, model, model_weights)

        tokenized_query = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )

        texts_list: list[tuple[float, str, str]] = []
        for target_text in data:
            tokenized_target = tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
            )

            embedding1 = (
                model(
                    tokenized_target["input_ids"],
                    attention_mask=tokenized_target["attention_mask"],
                )[0]
                .detach()
                .numpy()[0, :, 0]
            )
            embedding2 = (
                model(
                    tokenized_query["input_ids"],
                    attention_mask=tokenized_query["attention_mask"],
                )[0]
                .detach()
                .numpy()[0, :, 0]
            )

            similarity = float(
                np.dot(embedding1, embedding2)
                / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )
            texts_list.append((similarity, target_text, query))
        return texts_list

    def runNNModels(
        self,
        data: list[str] | None = None,
        query: str = DEFAULT_QUERY,
    ) -> None:
        """Run all registered NN models and store results.

        Args:
            data: Source texts to compare against. Defaults to sample data.
            query: Query text to compare.
        """
        if data is None:
            data = DEFAULT_DATA

        for model_objects in self.nnModels:
            tokenizer, model, model_weights = model_objects

            tokenized_query = tokenizer(
                query,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
            )
            text_results: list[tuple[float, str, str]] = []
            for text in data:
                tokenized_text = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    return_attention_mask=True,
                )

                embedding1 = (
                    model(
                        tokenized_text["input_ids"],
                        attention_mask=tokenized_text["attention_mask"],
                    )[0]
                    .detach()
                    .numpy()[0, :, 0]
                )
                embedding2 = (
                    model(
                        tokenized_query["input_ids"],
                        attention_mask=tokenized_query["attention_mask"],
                    )[0]
                    .detach()
                    .numpy()[0, :, 0]
                )

                embedding1_sq = np.square(embedding1.sum())
                embedding2_sq = np.square(embedding2.sum())
                similarity = cosine_similarity(
                    embedding1_sq.reshape(1, -1), embedding2_sq.reshape(1, -1)
                ).round(3)
                text_results.append((float(similarity[0, 0]), text, query))
            self.results[model_weights] = text_results

    @staticmethod
    def squared_sum(x: list[float] | np.ndarray) -> float:
        """Return the square root of the sum of squares, rounded to 3 decimal places.

        Args:
            x: Input vector.

        Returns:
            L2 norm of the vector, rounded.
        """
        return round(float(np.sqrt(sum(a * a for a in x))), 3)

    def cos_similarity(self, x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
        """Return cosine similarity between two vectors.

        Args:
            x: First vector.
            y: Second vector.

        Returns:
            Cosine similarity score.
        """
        numerator = sum(a * b for a, b in zip(x, y, strict=False))
        denominator = self.squared_sum(x) * self.squared_sum(y)
        return round(numerator / float(denominator), 3)

    def methodBert(
        self,
        data: list[str] | None = None,
        query: str = DEFAULT_QUERY,
    ) -> list[tuple[float, str, str]]:
        """Compare texts using BERT embeddings.

        Args:
            data: Source texts to compare against. Defaults to sample data.
            query: Query text to compare.

        Returns:
            List of (similarity, source_text, query_text) tuples.
        """
        if data is None:
            data = DEFAULT_DATA

        tokenizer, model = _get_or_load_model(
            BertTokenizer, transformers.BertModel, "bert-base-uncased"
        )

        tokenized_query = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )

        results: list[tuple[float, str, str]] = []
        for text in data:
            tokenized_text = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
            )

            embedding1 = (
                model(
                    tokenized_text["input_ids"],
                    attention_mask=tokenized_text["attention_mask"],
                )[0]
                .detach()
                .numpy()[0, :, 0]
            )
            embedding2 = (
                model(
                    tokenized_query["input_ids"],
                    attention_mask=tokenized_query["attention_mask"],
                )[0]
                .detach()
                .numpy()[0, :, 0]
            )

            similarity = float(
                np.dot(embedding1, embedding2)
                / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )
            results.append((similarity, text, query))
        return results

    def methodRoBERTa(
        self,
        data: list[str] | None = None,
        query: str = DEFAULT_QUERY,
    ) -> list[tuple[float, str, str]]:
        """Compare texts using RoBERTa embeddings.

        Args:
            data: Source texts to compare against. Defaults to sample data.
            query: Query text to compare.

        Returns:
            List of (similarity, source_text, query_text) tuples.
        """
        if data is None:
            data = DEFAULT_DATA

        tokenizer, model = _get_or_load_model(
            RobertaTokenizer, transformers.RobertaModel, "roberta-base"
        )

        tokenized_query = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )

        results: list[tuple[float, str, str]] = []
        for text in data:
            tokenized_text = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
            )

            embedding1 = (
                model(
                    tokenized_text["input_ids"],
                    attention_mask=tokenized_text["attention_mask"],
                )[0]
                .detach()
                .numpy()[0, :, 0]
            )
            embedding2 = (
                model(
                    tokenized_query["input_ids"],
                    attention_mask=tokenized_query["attention_mask"],
                )[0]
                .detach()
                .numpy()[0, :, 0]
            )

            similarity = float(
                np.dot(embedding1, embedding2)
                / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )
            results.append((similarity, text, query))
        return results

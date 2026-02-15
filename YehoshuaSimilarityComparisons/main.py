"""Entry point for running similarity comparison methods."""

from __future__ import annotations

from sim import Similarity


def main() -> None:
    """Run similarity comparison demo."""
    similarity = Similarity()
    # Uncomment methods to run:
    # print("NLTK/Doc2Vec:", similarity.methodNLTK())
    # print("TF-IDF:", similarity.methodScikitlearn())
    # print("BERT:", similarity.methodBert())
    # print("RoBERTa:", similarity.methodRoBERTa())


if __name__ == "__main__":
    main()

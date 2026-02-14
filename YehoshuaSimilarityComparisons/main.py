"""Entry point for running similarity comparison methods."""

from __future__ import annotations

from plotting import runModels
from sim import Similarity

if __name__ == "__main__":
    similarity = Similarity()
    # Uncomment methods to run:
    # similarity.methodNLTK()
    # similarity.methodScikitlearn()
    # similarity.methodBert()
    # similarity.methodRoBERTa()

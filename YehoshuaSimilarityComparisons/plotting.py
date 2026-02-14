"""Model runner - loads BERT and runs similarity comparison."""

from __future__ import annotations

import transformers
from sim import Similarity
from transformers import BertTokenizer


def runModels() -> None:
    """Initialize Similarity with BERT model and run NN comparison."""
    similarity = Similarity()
    similarity.addModel(BertTokenizer, transformers.BertModel, "bert-base-uncased")
    similarity.runNNModels()

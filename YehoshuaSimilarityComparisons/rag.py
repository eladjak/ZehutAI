"""RAG (Retrieval-Augmented Generation) fusion pipeline.

Implements query expansion using DictaLM 2.0, vector search with
sentence-transformers, and Reciprocal Rank Fusion (RRF) for reranking.
"""

from __future__ import annotations

import torch
from embeddings_comparison import compare_sentences
from transformers import AutoModelForCausalLM, AutoTokenizer

# Detect device once at module level
device: str = "cuda" if torch.cuda.is_available() else "cpu"


def generate_queries(original_query: str) -> list[str]:
    """Expand a query into multiple related queries using DictaLM 2.0.

    Args:
        original_query: The original search query to expand.

    Returns:
        List of generated query variations.
    """
    model = AutoModelForCausalLM.from_pretrained(
        "dicta-il/dictalm2.0-instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

    messages = [
        {"role": "user", "content": original_query},
    ]

    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Fixed: previously passed undefined `input_ids` as second argument
    generated_ids = model.generate(encoded, max_new_tokens=50, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    generated_queries = decoded[0].strip().split("\n")
    return generated_queries


def vector_search(query: str, all_documents: dict[str, str]) -> dict[str, float]:
    """Return cosine similarity scores between query and all documents.

    Args:
        query: The search query.
        all_documents: Dict mapping document ID to document text.

    Returns:
        Dict of document IDs to similarity scores, sorted descending.
    """
    available_docs = list(all_documents.keys())
    scores = {doc: compare_sentences([doc, query]) for doc in available_docs}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def reciprocal_rank_fusion(
    search_results_dict: dict[str, dict[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        search_results_dict: Dict mapping query to its search results
            (each a dict of doc_id -> score).
        k: RRF constant (default 60). Higher values reduce the influence
            of high rankings from individual queries.

    Returns:
        Dict of document IDs to fused scores, sorted descending.
    """
    fused_scores: dict[str, float] = {}

    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(
            sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + k)

    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )
    return reranked_results


def generate_output(
    reranked_results: dict[str, float], queries: list[str]
) -> str:
    """Generate final output based on reranked results and queries.

    Args:
        reranked_results: Fused document scores from RRF.
        queries: The expanded queries used for retrieval.

    Returns:
        Summary string of results.
    """
    return (
        f"Final output based on {queries} "
        f"and reranked documents: {list(reranked_results.keys())}"
    )


# Predefined set of documents (usually these would be from a search database)
ALL_DOCUMENTS: dict[str, str] = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism.",
}


if __name__ == "__main__":
    original_query = "impact of climate change"
    generated_queries = generate_queries(original_query)

    all_results: dict[str, dict[str, float]] = {}
    for query in generated_queries:
        search_results = vector_search(query, ALL_DOCUMENTS)
        all_results[query] = search_results

    reranked_results = reciprocal_rank_fusion(all_results)

    final_output = generate_output(reranked_results, generated_queries)
    print(final_output)

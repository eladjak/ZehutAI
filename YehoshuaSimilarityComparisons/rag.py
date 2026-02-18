"""RAG (Retrieval-Augmented Generation) fusion pipeline.

Implements query expansion using DictaLM 2.0, vector search with
sentence-transformers, and Reciprocal Rank Fusion (RRF) for reranking.
"""

from __future__ import annotations

import re

import torch
from embeddings_comparison import compare_sentences
from transformers import AutoModelForCausalLM, AutoTokenizer

# Detect device once at module level
device: str = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_text(text: str) -> str:
    """Normalize a text string for consistent comparison.

    Strips leading/trailing whitespace and collapses internal runs of
    whitespace (spaces, tabs, newlines) to a single space.

    Args:
        text: Raw input text to normalize.

    Returns:
        Normalized text with uniform single-space separation and no
        leading or trailing whitespace.

    Examples:
        >>> preprocess_text("  hello   world  ")
        'hello world'
        >>> preprocess_text("line1\\n\\nline2\\t end")
        'line1 line2 end'
    """
    return re.sub(r"\s+", " ", text).strip()


def chunk_documents(
    documents: dict[str, str],
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[dict[str, str | int]]:
    """Split documents into overlapping chunks for finer-grained retrieval.

    Long documents are tokenized by whitespace and split into windows of
    ``chunk_size`` words with ``overlap`` words of context carried over
    from the previous chunk.  Short documents that fit within a single
    chunk are returned as-is.

    Args:
        documents: Mapping of document ID to document text.
        chunk_size: Maximum number of words per chunk (default 200).
        overlap: Number of words from the end of the previous chunk to
            include at the start of the next chunk (default 50).
            Must be strictly less than ``chunk_size``.

    Returns:
        List of dicts, each with keys:
            - ``"chunk_id"`` (str): Unique identifier in the form
              ``"<source_id>_chunk<N>"`` (or just ``"<source_id>"`` when
              the document fits in a single chunk).
            - ``"source_id"`` (str): The original document ID.
            - ``"chunk_index"`` (int): Zero-based index of this chunk
              within its source document.
            - ``"text"`` (str): The chunk text.

    Raises:
        ValueError: If ``overlap`` >= ``chunk_size``.

    Examples:
        >>> docs = {"d1": "word " * 250}
        >>> chunks = chunk_documents(docs, chunk_size=100, overlap=20)
        >>> chunks[0]["source_id"]
        'd1'
        >>> chunks[0]["chunk_index"]
        0
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    result: list[dict[str, str | int]] = []

    for source_id, text in documents.items():
        cleaned = preprocess_text(text)
        words = cleaned.split()

        if len(words) <= chunk_size:
            result.append(
                {
                    "chunk_id": source_id,
                    "source_id": source_id,
                    "chunk_index": 0,
                    "text": cleaned,
                }
            )
            continue

        step = chunk_size - overlap
        chunk_index = 0
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            result.append(
                {
                    "chunk_id": f"{source_id}_chunk{chunk_index}",
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "text": " ".join(chunk_words),
                }
            )
            if end >= len(words):
                break
            start += step
            chunk_index += 1

    return result


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


def vector_search(
    query: str,
    all_documents: dict[str, str],
    top_k: int | None = None,
) -> dict[str, float]:
    """Return cosine similarity scores between query and all documents.

    The query and each document text are preprocessed with
    :func:`preprocess_text` before comparison to ensure consistent
    whitespace handling.

    Args:
        query: The search query.
        all_documents: Dict mapping document ID to document text.
        top_k: If given, return only the *top_k* highest-scoring
            documents.  When ``None`` (default) all documents are
            returned, preserving backward-compatible behaviour.

    Returns:
        Dict of document IDs to similarity scores, sorted descending.
        Contains at most ``top_k`` entries when ``top_k`` is specified.
    """
    cleaned_query = preprocess_text(query)
    available_docs = list(all_documents.keys())
    scores = {
        doc: compare_sentences([preprocess_text(doc), cleaned_query])
        for doc in available_docs
    }
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    if top_k is not None:
        sorted_scores = dict(list(sorted_scores.items())[:top_k])
    return sorted_scores


def reciprocal_rank_fusion(
    search_results_dict: dict[str, dict[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each document's fused score is the sum of ``1 / (rank + k)`` across
    all query result lists in which it appears, where ``rank`` is the
    zero-based position in each list after sorting by descending score.

    Note on ``k=60``:
        The value 60 is the standard constant introduced in the original
        RRF paper (Cormack, Clarke & Buettcher, SIGIR 2009).  It was
        empirically shown to be robust across a wide range of retrieval
        tasks: it softens the score penalty for documents ranked slightly
        lower, preventing any single highly-ranked result from dominating
        the fused list, while still rewarding consistent top rankings
        across multiple queries.  Lowering ``k`` gives more weight to
        rank-1 documents; raising it produces a more uniform distribution.

    Args:
        search_results_dict: Dict mapping query to its search results
            (each a dict of doc_id -> score).
        k: RRF smoothing constant (default 60, the standard value from
            the original paper).  Higher values reduce the influence of
            high rankings from individual queries.

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

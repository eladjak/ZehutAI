"""RAG (Retrieval-Augmented Generation) fusion pipeline.

Implements query expansion using DictaLM 2.0, vector search with
sentence-transformers, and Reciprocal Rank Fusion (RRF) for reranking.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zehutai.embeddings_comparison import compare_sentences

# Detect device once at module level
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Module-level model cache for RAG generation models
# ---------------------------------------------------------------------------

_rag_model_cache: dict[str, tuple[Any, Any]] = {}


def _get_or_load_rag_model(model_name: str) -> tuple[Any, Any]:
    """Cache and return (model, tokenizer) for RAG generation models.

    Loads the model and tokenizer from HuggingFace on first call, then
    returns the cached objects on subsequent calls to avoid redundant
    downloads and GPU memory allocations.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``"dicta-il/dictalm2.0-instruct"``).

    Returns:
        A ``(model, tokenizer)`` tuple.  The model is loaded with
        ``torch_dtype=torch.bfloat16`` and ``device_map`` set to the
        module-level :data:`device`.
    """
    if model_name not in _rag_model_cache:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _rag_model_cache[model_name] = (model, tokenizer)
    return _rag_model_cache[model_name]


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


def preprocess_hebrew(text: str, *, normalize_finals: bool = False) -> str:
    """Normalize a Hebrew text string for consistent comparison.

    Applies general whitespace normalization via :func:`preprocess_text`,
    then strips Hebrew niqqud (vowel-pointing diacritics) and optionally
    maps final letter forms to their standard equivalents so that, e.g.,
    ``"מלך"`` and ``"מלכ"`` compare as identical tokens.

    Args:
        text: Raw Hebrew (or mixed) input text.
        normalize_finals: When ``True``, replace Hebrew final letters
            (ך, ם, ן, ף, ץ) with their medial counterparts
            (כ, מ, נ, פ, צ).  Defaults to ``False``.

    Returns:
        Cleaned text with niqqud removed and, if requested, final letters
        normalized.

    Examples:
        >>> preprocess_hebrew("שָׁלוֹם")
        'שלום'
        >>> preprocess_hebrew("מלך", normalize_finals=True)
        'מלכ'
    """
    # Step 1: general whitespace normalization
    result = preprocess_text(text)

    # Step 2: remove Hebrew niqqud / cantillation marks
    # Unicode ranges: U+0591-U+05BD (cantillation + most niqqud),
    # U+05BF-U+05C7 (rafe, shin/sin dot, holam, qamats, etc.)
    result = re.sub(r"[\u0591-\u05BD\u05BF-\u05C7]", "", result)

    # Step 3: optionally normalize final letter forms
    if normalize_finals:
        _FINALS_MAP = str.maketrans("ךםןףץ", "כמנפצ")
        result = result.translate(_FINALS_MAP)

    return result


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

    The model and tokenizer are loaded once and cached at module level via
    :func:`_get_or_load_rag_model` to avoid repeated downloads and GPU
    memory allocations across calls.

    Args:
        original_query: The original search query to expand.

    Returns:
        List of generated query variations.
    """
    model, tokenizer = _get_or_load_rag_model("dicta-il/dictalm2.0-instruct")

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


def rag_pipeline(
    query: str,
    documents: dict[str, str],
    use_query_expansion: bool = False,
    top_k: int | None = None,
    rrf_k: int = 60,
    chunk_size: int | None = None,
    chunk_overlap: int = 50,
) -> dict[str, float]:
    """Run an end-to-end RAG retrieval pipeline.

    Optionally chunks documents, optionally expands the query, runs vector
    search for each query variant, then fuses the ranked lists with
    Reciprocal Rank Fusion.

    Pipeline stages:

    1. **Chunking** (optional): if ``chunk_size`` is set, documents are
       split into overlapping chunks via :func:`chunk_documents`.  The
       resulting flat text chunks (keyed by ``chunk_id``) are used as the
       retrieval corpus.
    2. **Query expansion** (optional): if ``use_query_expansion`` is
       ``True``, :func:`generate_queries` is called to produce additional
       query variants.  Otherwise only the original query is used.
    3. **Vector search**: :func:`vector_search` is called once per query
       variant against the (possibly chunked) document corpus.
    4. **Fusion**: all per-query ranked lists are merged with
       :func:`reciprocal_rank_fusion`.
    5. **Top-K trimming**: if ``top_k`` is set, the fused results are
       trimmed to the highest-scoring ``top_k`` entries.

    Args:
        query: The user's search query.
        documents: Mapping of document ID to document text.
        use_query_expansion: If ``True``, use DictaLM 2.0 to generate
            additional query variants before retrieval.
        top_k: Maximum number of results to return.  ``None`` returns all.
        rrf_k: Smoothing constant for Reciprocal Rank Fusion (default 60).
        chunk_size: If set, documents are chunked to at most this many
            words before retrieval.  ``None`` disables chunking.
        chunk_overlap: Word overlap between consecutive chunks when
            ``chunk_size`` is set (default 50).

    Returns:
        Dict of document (or chunk) IDs to fused RRF scores, sorted
        descending.  At most ``top_k`` entries when ``top_k`` is given.

    Raises:
        ValueError: Propagated from :func:`chunk_documents` when
            ``chunk_overlap >= chunk_size``.

    Examples:
        >>> docs = {"d1": "climate change", "d2": "ancient history"}
        >>> result = rag_pipeline("climate", docs)
        >>> list(result.keys())[0]  # highest scorer
        'd1'
    """
    # Stage 1: optional document chunking
    if chunk_size is not None:
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
        retrieval_corpus: dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}  # type: ignore[misc]
    else:
        retrieval_corpus = dict(documents)

    # Stage 2: optional query expansion
    if use_query_expansion:
        queries = generate_queries(query)
    else:
        queries = [query]

    # Stage 3: vector search per query
    all_results: dict[str, dict[str, float]] = {}
    for q in queries:
        all_results[q] = vector_search(q, retrieval_corpus)

    # Stage 4: reciprocal rank fusion
    fused = reciprocal_rank_fusion(all_results, k=rrf_k)

    # Stage 5: optional top-k trimming
    if top_k is not None:
        fused = dict(list(fused.items())[:top_k])

    return fused


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

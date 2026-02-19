"""Hebrew embedding model benchmark suite.

Compares different embedding models on Hebrew text similarity using
a curated set of human-annotated sentence pairs that cover paraphrases,
related concepts, unrelated texts, and Hebrew-specific linguistic phenomena
(register variation, construct state / smichut).

Also exposes a standalone benchmark corpus (HEBREW_SENTENCE_PAIRS) of 20
sentence pairs with explicit ``sent1``/``sent2``/``category``/``id``/``description``
fields, along with :func:`run_benchmark` (summary-dict API) and
:func:`format_benchmark_report` (single-model table API) for quick evaluation
via :func:`zehutai.embeddings_comparison.compare_sentences`.

The benchmark data is model-agnostic: it can be imported and inspected
independently of any model loading.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

HEBREW_BENCHMARK_PAIRS: list[dict] = [
    # High similarity pairs (paraphrases)
    {
        "s1": "החתול ישב על השטיח",
        "s2": "חתלתול נח על המזרן",
        "label": "high",
        "category": "paraphrase",
    },
    {
        "s1": "הילד רץ בגינה",
        "s2": "הילד משחק בחצר",
        "label": "high",
        "category": "paraphrase",
    },
    {
        "s1": "מחר יהיה חם מאוד",
        "s2": "מזג האוויר מחר יהיה חמסיני",
        "label": "high",
        "category": "paraphrase",
    },
    {
        "s1": "הממשלה אישרה את התקציב",
        "s2": "תקציב המדינה אושר על ידי הממשלה",
        "label": "high",
        "category": "paraphrase",
    },
    {
        "s1": "הרופא רשם תרופות למטופל",
        "s2": "המטופל קיבל מרשם מהרופא",
        "label": "high",
        "category": "paraphrase",
    },
    # Medium similarity (related but different)
    {
        "s1": "הכלב רץ בפארק",
        "s2": "החתול ישן על הספה",
        "label": "medium",
        "category": "related_animals",
    },
    {
        "s1": "התלמידים לומדים מתמטיקה",
        "s2": "המורה מלמדת היסטוריה",
        "label": "medium",
        "category": "education",
    },
    {
        "s1": "המכונית נוסעת מהר",
        "s2": "האוטובוס עצר בתחנה",
        "label": "medium",
        "category": "transport",
    },
    # Low similarity (unrelated)
    {
        "s1": "השמש זורחת בבוקר",
        "s2": "המחשב קרס אתמול",
        "label": "low",
        "category": "unrelated",
    },
    {
        "s1": "התינוק בכה בלילה",
        "s2": "שוק המניות עלה היום",
        "label": "low",
        "category": "unrelated",
    },
    {
        "s1": "הפיצה הייתה טעימה",
        "s2": "הטיסה התעכבה בשעתיים",
        "label": "low",
        "category": "unrelated",
    },
    {
        "s1": "ירושלים היא עיר עתיקה",
        "s2": "פייתון היא שפת תכנות",
        "label": "low",
        "category": "unrelated",
    },
    # Hebrew-specific: same meaning, different register (formal vs colloquial)
    {
        "s1": "אני רוצה לאכול",
        "s2": "ברצוני לסעוד",
        "label": "high",
        "category": "register",
    },
    {
        "s1": "הוא הלך הביתה",
        "s2": 'הנ"ל שב למעונו',
        "label": "high",
        "category": "register",
    },
    # Hebrew-specific: construct state (smichut)
    {
        "s1": "בית הספר נסגר",
        "s2": "בית ספר סגר את שעריו",
        "label": "high",
        "category": "construct_state",
    },
]

# ---------------------------------------------------------------------------
# Extended corpus: 20 pairs with sent1/sent2/category/id/description schema
# ---------------------------------------------------------------------------

HEBREW_SENTENCE_PAIRS: list[dict[str, str]] = [
    # ------------------------------------------------------------------
    # HIGH similarity pairs (7) – paraphrases and near-duplicates
    # ------------------------------------------------------------------
    {
        "id": "high_01",
        "sent1": "הממשלה החליטה להעלות את מס ההכנסה",
        "sent2": "ממשלת ישראל אישרה העלאה במס ההכנסה",
        "category": "high",
        "description": "Politics: government income tax increase (paraphrase)",
    },
    {
        "id": "high_02",
        "sent1": "החברה השיקה מוצר טכנולוגי חדש",
        "sent2": "הפירמה הציגה מוצר חדשני בתחום הטכנולוגיה",
        "category": "high",
        "description": "Technology: company product launch (paraphrase)",
    },
    {
        "id": "high_03",
        "sent1": "הפיצה עם הגבינה הייתה טעימה מאוד",
        "sent2": "הפיצה הגבינתית הייתה ממש טובה",
        "category": "high",
        "description": "Food: tasty cheese pizza (near-duplicate)",
    },
    {
        "id": "high_04",
        "sent1": "קבוצת הכדורגל זכתה באליפות הארצית",
        "sent2": "הנבחרת הייתה לאלופת הארץ בכדורגל",
        "category": "high",
        "description": "Sports: football team wins national championship (paraphrase)",
    },
    {
        "id": "high_05",
        "sent1": "היום ירד גשם כבד בכל רחבי הארץ",
        "sent2": "גשמים עזים ירדו ברחבי ישראל היום",
        "category": "high",
        "description": "Weather: heavy rain across the country (paraphrase)",
    },
    {
        "id": "high_06",
        "sent1": "הסטודנטים למדו לקראת הבחינות הסופיות",
        "sent2": "תלמידי האוניברסיטה השקיעו בלימוד לקראת המבחנים",
        "category": "high",
        "description": "Education: students studying for final exams (paraphrase)",
    },
    {
        "id": "high_07",
        "sent1": "הרופא המליץ לחולה לנוח ולשתות הרבה מים",
        "sent2": "הרופאה יעצה למטופל לישון ולהרבות בשתייה",
        "category": "high",
        "description": "Health: doctor advises rest and hydration (paraphrase)",
    },
    # ------------------------------------------------------------------
    # MEDIUM similarity pairs (7) – related topic, different wording/focus
    # ------------------------------------------------------------------
    {
        "id": "medium_01",
        "sent1": "ראש הממשלה נאם בכנסת על המדיניות הכלכלית",
        "sent2": "שר האוצר הציג את תקציב המדינה לשנה הבאה",
        "category": "medium",
        "description": "Politics: political economy speeches (same domain, different content)",
    },
    {
        "id": "medium_02",
        "sent1": "הסמארטפון החדש מגיע עם מצלמה מתקדמת",
        "sent2": "יצרנית האלקטרוניקה השיקה טבלט עם מסך גדול",
        "category": "medium",
        "description": "Technology: different consumer electronics launches",
    },
    {
        "id": "medium_03",
        "sent1": "השף הכין מנה מיוחדת לחג",
        "sent2": "המסעדה זכתה בפרס יוקרתי על בישול ים תיכוני",
        "category": "medium",
        "description": "Food: cooking and restaurant (same domain, different focus)",
    },
    {
        "id": "medium_04",
        "sent1": "השחקן כבש שלושה שערים במשחק",
        "sent2": "הקהל במגרש בא לראות את נבחרת הכדורגל",
        "category": "medium",
        "description": "Sports: football player vs. fans at stadium",
    },
    {
        "id": "medium_05",
        "sent1": "הטמפרטורות ירדו משמעותית בחודשי החורף",
        "sent2": "הקיץ השנה היה חם מן הרגיל",
        "category": "medium",
        "description": "Weather: seasonal temperature changes (related, different seasons)",
    },
    {
        "id": "medium_06",
        "sent1": "בית הספר פתח שנת לימודים חדשה",
        "sent2": "הילדים קיבלו ציונים גבוהים בבחינות",
        "category": "medium",
        "description": "Education: school year and grades (same domain, different events)",
    },
    {
        "id": "medium_07",
        "sent1": "התרופה החדשה הוכחה כיעילה נגד הנגיף",
        "sent2": "בתי החולים נערכים לעונת השפעת",
        "category": "medium",
        "description": "Health: medicine and hospital preparedness (related domain)",
    },
    # ------------------------------------------------------------------
    # LOW similarity pairs (6) – unrelated topics
    # ------------------------------------------------------------------
    {
        "id": "low_01",
        "sent1": "הבחירות לכנסת יתקיימו בחודש מרץ",
        "sent2": "המחשב הנייד שלי צריך טעינה",
        "category": "low",
        "description": "Politics vs. technology (unrelated)",
    },
    {
        "id": "low_02",
        "sent1": "הסושי במסעדה היה טרי ומשובח",
        "sent2": "סיום אליפות אירופה בכדורגל קרב",
        "category": "low",
        "description": "Food vs. sports (unrelated)",
    },
    {
        "id": "low_03",
        "sent1": "הסופה גרמה לנזקים כבדים בצפון הארץ",
        "sent2": "שיעור הביולוגיה עסק בתאי גזע",
        "category": "low",
        "description": "Weather vs. education (unrelated)",
    },
    {
        "id": "low_04",
        "sent1": "התינוק עשה את צעדיו הראשונים",
        "sent2": "הבורסה בתל אביב נסגרה בעלייה חדה",
        "category": "low",
        "description": "Family/health vs. finance (unrelated)",
    },
    {
        "id": "low_05",
        "sent1": "הסרט הישראלי זכה בפרס בפסטיבל קאן",
        "sent2": "הנהג שכח להביא את המפתחות",
        "category": "low",
        "description": "Culture vs. daily life (unrelated)",
    },
    {
        "id": "low_06",
        "sent1": "האמן צייר ציור ענק על קיר הגלריה",
        "sent2": "הגשם פגע בגידולים החקלאיים בנגב",
        "category": "low",
        "description": "Culture/art vs. agriculture/weather (unrelated)",
    },
]

# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "multilingual-mpnet": {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "type": "sentence-transformers",
    },
    "multilingual-minilm": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "type": "sentence-transformers",
    },
    # These would need real models - keep as config even if not runnable:
    "dictabert": {
        "name": "dicta-il/dictabert",
        "type": "bert",
    },
    "alephbert": {
        "name": "onlplab/alephbert-base",
        "type": "bert",
    },
}

# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].

    Raises:
        ValueError: If either vector has zero norm.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Zero-norm vector encountered; cannot compute cosine similarity.")
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_sentence_transformer(model_name: str) -> Any:
    """Load a SentenceTransformer model by name.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        A loaded SentenceTransformer instance.

    Raises:
        ImportError: If sentence_transformers is not installed.
        OSError: If the model cannot be found or downloaded.
    """
    st = importlib.import_module("sentence_transformers")
    return st.SentenceTransformer(model_name)


def _encode_pair(model: Any, s1: str, s2: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode a sentence pair using a SentenceTransformer model.

    Args:
        model: A loaded SentenceTransformer (or compatible) model with an
            ``encode`` method that returns numpy arrays.
        s1: First sentence.
        s2: Second sentence.

    Returns:
        Tuple of (embedding_for_s1, embedding_for_s2) as 1-D numpy arrays.
    """
    embeddings = model.encode([s1, s2])
    emb1 = np.array(embeddings[0], dtype=np.float32)
    emb2 = np.array(embeddings[1], dtype=np.float32)
    return emb1, emb2


def _run_benchmark_pairs(
    model_name: str,
    pairs: list[dict] | None = None,
    *,
    model_instance: Any = None,
) -> list[dict]:
    """Internal: run a SentenceTransformer model on pairs with s1/s2/label schema.

    Args:
        model_name: Key in :data:`MODEL_CONFIGS` or a raw Hugging Face identifier.
        pairs: List of pair dicts with ``s1`` and ``s2`` keys.
            Defaults to :data:`HEBREW_BENCHMARK_PAIRS`.
        model_instance: Optional pre-loaded model.  When supplied, model
            loading is skipped entirely.

    Returns:
        List of result dicts preserving all original fields plus ``score``,
        ``model``, and ``elapsed_ms``.
    """
    if pairs is None:
        pairs = HEBREW_BENCHMARK_PAIRS

    model_identifier: str = MODEL_CONFIGS.get(model_name, {}).get("name", model_name)
    if model_instance is None:
        model_instance = _load_sentence_transformer(model_identifier)

    results: list[dict] = []
    for pair in pairs:
        s1: str = pair["s1"]
        s2: str = pair["s2"]

        t_start = time.perf_counter()
        emb1, emb2 = _encode_pair(model_instance, s1, s2)
        score = _cosine_similarity(emb1, emb2)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        result = dict(pair)
        result["score"] = round(score, 6)
        result["model"] = model_name
        result["elapsed_ms"] = round(elapsed_ms, 2)
        results.append(result)

    return results


def _run_benchmark_summary(model_name: str | None = None) -> dict[str, Any]:
    """Internal: run compare_sentences on HEBREW_SENTENCE_PAIRS and return summary.

    Args:
        model_name: Optional display name for the model.  Defaults to
            ``"paraphrase-multilingual-mpnet-base-v2"``.

    Returns:
        Summary dict with keys ``model``, ``pairs``, and ``stats``.
    """
    from zehutai.embeddings_comparison import compare_sentences  # local import for testability

    effective_model = model_name or "paraphrase-multilingual-mpnet-base-v2"
    pairs_with_scores: list[dict[str, Any]] = []
    category_scores: dict[str, list[float]] = {"high": [], "medium": [], "low": []}

    for pair in HEBREW_SENTENCE_PAIRS:
        score = compare_sentences([pair["sent1"], pair["sent2"]])
        entry: dict[str, Any] = {
            "id": pair["id"],
            "sent1": pair["sent1"],
            "sent2": pair["sent2"],
            "category": pair["category"],
            "description": pair["description"],
            "score": score,
        }
        pairs_with_scores.append(entry)
        category_scores[pair["category"]].append(score)

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    high_avg = _avg(category_scores["high"])
    medium_avg = _avg(category_scores["medium"])
    low_avg = _avg(category_scores["low"])

    return {
        "model": effective_model,
        "pairs": pairs_with_scores,
        "stats": {
            "high_avg": high_avg,
            "medium_avg": medium_avg,
            "low_avg": low_avg,
            "separation_score": high_avg - low_avg,
        },
    }


def run_benchmark(
    model_name: str | None = None,
    pairs: list[dict] | None = None,
    *,
    model_instance: Any = None,
) -> dict[str, Any] | list[dict]:
    """Run the Hebrew similarity benchmark.

    Supports two call styles:

    **New (summary-dict) style** – for evaluating against
    :data:`HEBREW_SENTENCE_PAIRS` via :func:`zehutai.embeddings_comparison.compare_sentences`:

        results = run_benchmark()
        results = run_benchmark("paraphrase-multilingual-mpnet-base-v2")

    Returns a summary dict::

        {
            "model": str,
            "pairs": [{"id", "sent1", "sent2", "category", "description", "score"}, ...],
            "stats": {"high_avg", "medium_avg", "low_avg", "separation_score"},
        }

    **Legacy (list) style** – for evaluating a pre-loaded SentenceTransformer
    model against :data:`HEBREW_BENCHMARK_PAIRS` (or custom pairs with ``s1``/``s2``
    keys), compatible with :func:`evaluate_benchmark`:

        results = run_benchmark("model-key", model_instance=loaded_model)
        results = run_benchmark("model-key", pairs=custom_pairs, model_instance=m)

    Returns a list of result dicts (original pair fields + ``score``, ``model``,
    ``elapsed_ms``).

    Args:
        model_name: Model display name / :data:`MODEL_CONFIGS` key.  When
            ``None`` in new-style mode, defaults to
            ``"paraphrase-multilingual-mpnet-base-v2"``.
        pairs: (Legacy style only) List of pair dicts with ``s1``/``s2`` keys.
            Defaults to :data:`HEBREW_BENCHMARK_PAIRS`.
        model_instance: (Legacy style only) Pre-loaded model object.

    Returns:
        Summary dict (new style) or list of scored dicts (legacy style).
    """
    if pairs is not None or model_instance is not None:
        # Legacy call: needs a concrete model_name string
        name = model_name if model_name is not None else "paraphrase-multilingual-mpnet-base-v2"
        return _run_benchmark_pairs(name, pairs, model_instance=model_instance)
    return _run_benchmark_summary(model_name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_benchmark(results: list[dict]) -> dict[str, float]:
    """Compute summary metrics from benchmark results.

    Metrics computed:

    - ``high_avg``: mean similarity for pairs labelled "high".
    - ``medium_avg``: mean similarity for pairs labelled "medium".
    - ``low_avg``: mean similarity for pairs labelled "low".
    - ``separation_score``: ``high_avg - low_avg`` (higher is better; ideal > 0).
    - ``high_low_gap``: alias for ``separation_score``.
    - ``high_medium_gap``: ``high_avg - medium_avg``.
    - ``medium_low_gap``: ``medium_avg - low_avg``.
    - ``rank_accuracy``: fraction of pairs where the ordering
      ``high > medium > low`` is preserved when comparing group averages.
      Returns 1.0 when ``high_avg > medium_avg > low_avg``,
      0.5 when only one ordering holds, and 0.0 when neither holds.
    - ``n_high``, ``n_medium``, ``n_low``: pair counts per label.
    - ``avg_elapsed_ms``: mean encoding time per pair (if present).

    Args:
        results: List of result dicts as returned by :func:`run_benchmark`.
            Each dict must contain ``label`` (str) and ``score`` (float).

    Returns:
        Dict of metric names to float values.  If a label group is empty,
        its average is ``float('nan')``.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("results list must not be empty")

    by_label: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    elapsed_values: list[float] = []

    for r in results:
        label = r.get("label", "")
        score = float(r["score"])
        if label in by_label:
            by_label[label].append(score)
        if "elapsed_ms" in r:
            elapsed_values.append(float(r["elapsed_ms"]))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    high_avg = _mean(by_label["high"])
    medium_avg = _mean(by_label["medium"])
    low_avg = _mean(by_label["low"])

    separation_score = high_avg - low_avg

    # rank_accuracy: how well the three groups are ordered
    ordering_checks = [
        not (np.isnan(high_avg) or np.isnan(medium_avg)) and high_avg > medium_avg,
        not (np.isnan(medium_avg) or np.isnan(low_avg)) and medium_avg > low_avg,
    ]
    n_checks = sum(1 for _ in ordering_checks)
    rank_accuracy = sum(1.0 for ok in ordering_checks if ok) / n_checks if n_checks else float("nan")

    metrics: dict[str, float] = {
        "high_avg": high_avg,
        "medium_avg": medium_avg,
        "low_avg": low_avg,
        "separation_score": separation_score,
        "high_low_gap": separation_score,
        "high_medium_gap": high_avg - medium_avg,
        "medium_low_gap": medium_avg - low_avg,
        "rank_accuracy": rank_accuracy,
        "n_high": float(len(by_label["high"])),
        "n_medium": float(len(by_label["medium"])),
        "n_low": float(len(by_label["low"])),
        "avg_elapsed_ms": _mean(elapsed_values),
    }
    return metrics


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_METRIC_LABELS: dict[str, str] = {
    "high_avg": "High-sim avg",
    "medium_avg": "Medium-sim avg",
    "low_avg": "Low-sim avg",
    "separation_score": "Separation (H-L)",
    "high_medium_gap": "Gap (H-M)",
    "medium_low_gap": "Gap (M-L)",
    "rank_accuracy": "Rank accuracy",
    "avg_elapsed_ms": "Avg ms/pair",
}


def _format_summary_report(results: dict[str, Any]) -> str:
    """Format a single-model run_benchmark() summary dict as a table string.

    Args:
        results: Dict returned by the new-style :func:`run_benchmark` (contains
            ``model``, ``pairs``, and ``stats`` keys).

    Returns:
        Multi-line formatted string with a per-pair table and statistics.
    """
    lines: list[str] = []
    model = results.get("model", "unknown")
    pairs: list[dict[str, Any]] = results.get("pairs", [])
    stats: dict[str, float] = results.get("stats", {})

    header = f"Hebrew NLP Benchmark  |  Model: {model}"
    separator = "=" * max(len(header), 80)

    lines.append(separator)
    lines.append(header)
    lines.append(separator)
    lines.append(f"{'ID':<12} {'Category':<10} {'Score':>7}  Description")
    lines.append("-" * 80)

    for pair in pairs:
        pair_id = pair.get("id", "")
        category = pair.get("category", "")
        score = pair.get("score", 0.0)
        description = pair.get("description", "")
        lines.append(f"{pair_id:<12} {category:<10} {score:>7.4f}  {description}")

    lines.append("=" * 80)
    lines.append("Statistics")
    lines.append("-" * 40)
    lines.append(f"  High   avg  : {stats.get('high_avg', 0.0):.4f}")
    lines.append(f"  Medium avg  : {stats.get('medium_avg', 0.0):.4f}")
    lines.append(f"  Low    avg  : {stats.get('low_avg', 0.0):.4f}")
    lines.append(
        f"  Separation  : {stats.get('separation_score', 0.0):.4f}  "
        "(high_avg - low_avg; higher = better model)"
    )
    lines.append("=" * 80)

    return "\n".join(lines)


def format_benchmark_report(all_results: dict[str, Any]) -> str:
    """Format benchmark results as a human-readable string.

    Supports two input formats:

    **New-style (single-model summary):** pass the dict returned by
    ``run_benchmark()`` (contains ``"model"``, ``"pairs"``, ``"stats"`` keys).
    Returns a per-pair table followed by per-category statistics.

    **Legacy-style (multi-model comparison):** pass a mapping from model
    name to the metrics dict returned by :func:`evaluate_benchmark`.
    Produces an ASCII table with one row per model and one column per
    metric, followed by a ranking section ordered by ``separation_score``.

    Args:
        all_results: Either a single-model summary dict (new style) or a
            ``dict[str, dict[str, float]]`` mapping model names to metrics
            (legacy style).

    Returns:
        Multi-line string suitable for printing to a terminal.

    Raises:
        ValueError: If ``all_results`` is empty (legacy style only).

    Note:
        Detection is automatic: if the dict contains a ``"pairs"`` key it
        is treated as a new-style summary; otherwise it is treated as a
        legacy multi-model mapping.
    """
    # Detect new-style single-model summary dict
    if "pairs" in all_results:
        return _format_summary_report(all_results)

    # Legacy multi-model format
    if not all_results:
        raise ValueError("all_results must not be empty")

    return _format_multi_model_report(all_results)


def _format_multi_model_report(all_results: dict[str, dict[str, float]]) -> str:
    """Format a human-readable comparison table of model benchmark results.

    Produces an ASCII table with one row per model and one column per metric,
    followed by a ranking section that orders models by ``separation_score``.

    Args:
        all_results: Mapping from model name to the metrics dict returned by
            :func:`evaluate_benchmark`.

    Returns:
        Multi-line string suitable for printing to a terminal.

    Raises:
        ValueError: If all_results is empty.
    """
    if not all_results:
        raise ValueError("all_results must not be empty")

    metric_keys = [
        "high_avg",
        "medium_avg",
        "low_avg",
        "separation_score",
        "rank_accuracy",
        "avg_elapsed_ms",
    ]

    col_headers = [_METRIC_LABELS.get(k, k) for k in metric_keys]

    # Determine column widths
    model_col_width = max(len("Model"), max(len(m) for m in all_results))
    metric_col_widths = [max(len(h), 10) for h in col_headers]

    def _fmt(val: float) -> str:
        if np.isnan(val):
            return "  N/A    "
        return f"{val:8.4f}"

    def _separator() -> str:
        parts = ["-" * model_col_width]
        parts += ["-" * w for w in metric_col_widths]
        return "+-" + "-+-".join(parts) + "-+"

    def _header_row() -> str:
        parts = [f"{'Model':<{model_col_width}}"]
        for h, w in zip(col_headers, metric_col_widths):
            parts.append(f"{h:^{w}}")
        return "| " + " | ".join(parts) + " |"

    def _data_row(model_name: str, metrics: dict[str, float]) -> str:
        parts = [f"{model_name:<{model_col_width}}"]
        for k, w in zip(metric_keys, metric_col_widths):
            val = metrics.get(k, float("nan"))
            formatted = _fmt(val).strip()
            parts.append(f"{formatted:^{w}}")
        return "| " + " | ".join(parts) + " |"

    sep = _separator()
    lines: list[str] = []
    lines.append("")
    lines.append("  Hebrew Embedding Model Benchmark")
    lines.append("  " + "=" * (len(sep) - 2))
    lines.append(sep)
    lines.append(_header_row())
    lines.append(sep)

    for model_name, metrics in all_results.items():
        lines.append(_data_row(model_name, metrics))

    lines.append(sep)

    # Ranking section: sort by separation_score descending
    ranked = sorted(
        all_results.items(),
        key=lambda kv: kv[1].get("separation_score", float("-inf")),
        reverse=True,
    )
    lines.append("")
    lines.append("  Ranking by separation score (high_avg - low_avg):")
    for rank, (model_name, metrics) in enumerate(ranked, start=1):
        sep_score = metrics.get("separation_score", float("nan"))
        rank_acc = metrics.get("rank_accuracy", float("nan"))
        sep_str = f"{sep_score:.4f}" if not np.isnan(sep_score) else "N/A"
        acc_str = f"{rank_acc:.2%}" if not np.isnan(rank_acc) else "N/A"
        lines.append(f"    {rank}. {model_name:<{model_col_width}}  sep={sep_str}  rank_acc={acc_str}")

    lines.append("")
    return "\n".join(lines)

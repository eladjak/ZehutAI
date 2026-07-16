"""ZehutAI command-line interface.

ממשק שורת פקודה לניתוח טקסט עברי ורב-לשוני / CLI for Hebrew & multilingual NLP.

Usage:
    python cli.py compare "sentence one" "sentence two"
    python cli.py similarity --method tfidf --data "text1" "text2" --query "query"
    python cli.py rag --query "query text" --top-k 5
    python cli.py benchmark [--format table|json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Graceful import helpers
# ---------------------------------------------------------------------------


def _import_compare_sentences() -> Any:
    """Import compare_sentences, raising ImportError with helpful message if unavailable."""
    try:
        from zehutai.embeddings_comparison import compare_sentences

        return compare_sentences
    except ImportError as exc:
        raise ImportError(
            "Could not import embeddings_comparison. "
            "Make sure sentence-transformers is installed: "
            "pip install sentence-transformers"
        ) from exc


def _import_similarity_class() -> Any:
    """Import the Similarity class from zehutai.similarity."""
    try:
        from zehutai.similarity.sim import Similarity

        return Similarity
    except ImportError as exc:
        raise ImportError(
            "Could not import Similarity class. "
            "Make sure scikit-learn, gensim, and nltk are installed: "
            "pip install scikit-learn gensim nltk"
        ) from exc


def _import_rag_functions() -> tuple[Any, Any, Any]:
    """Import vector_search, reciprocal_rank_fusion, and ALL_DOCUMENTS from zehutai.rag."""
    try:
        from zehutai.rag.rag import ALL_DOCUMENTS, reciprocal_rank_fusion, vector_search

        return vector_search, reciprocal_rank_fusion, ALL_DOCUMENTS
    except ImportError as exc:
        raise ImportError(
            "Could not import RAG functions. "
            "Make sure torch and transformers are installed: "
            "pip install torch transformers"
        ) from exc


def _import_benchmark_functions() -> tuple[Any, Any, Any]:
    """Import run_benchmark, evaluate_benchmark, and format_benchmark_report."""
    try:
        from zehutai.hebrew.hebrew_benchmark import (
            evaluate_benchmark,
            format_benchmark_report,
            run_benchmark,
        )

        return run_benchmark, evaluate_benchmark, format_benchmark_report
    except ImportError as exc:
        raise ImportError(
            "Could not import benchmark functions. "
            "Make sure numpy and sentence-transformers are installed: "
            "pip install numpy sentence-transformers"
        ) from exc


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    """Handle the 'compare' subcommand.

    Compares two sentences using the multilingual sentence-transformers model
    and prints their cosine similarity score.

    Args:
        args: Parsed CLI arguments with sentence1 and sentence2.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        compare_sentences = _import_compare_sentences()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        score = compare_sentences([args.sentence1, args.sentence2])
        print(f"Similarity score / ציון דמיון: {score:.6f}")
        return 0
    except Exception as exc:
        print(f"Error computing similarity: {exc}", file=sys.stderr)
        return 1


def cmd_similarity(args: argparse.Namespace) -> int:
    """Handle the 'similarity' subcommand.

    Runs a similarity method from the Similarity class on provided texts.

    Args:
        args: Parsed CLI arguments with method, data, and query.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        Similarity = _import_similarity_class()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    method = args.method
    data: list[str] = args.data if args.data else []
    query: str = args.query

    sim = Similarity()

    try:
        if method == "tfidf":
            results = sim.methodScikitlearn(data=data or None, query=query)
            print(f"TF-IDF similarity results (query: '{query}'):")
            print("-" * 60)
            for idx, (text, score) in enumerate(results, 1):
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  {idx}. [{score:.4f}]  {preview}")

        elif method == "nltk":
            results = sim.methodNLTK(data=data or None, query=query)
            print(f"Doc2Vec/NLTK similarity results (query: '{query}'):")
            print("-" * 60)
            for doc_tag, score in results:
                # doc_tag is a string index like "0", "1", "2"
                actual_data = data if data else sim.texts
                try:
                    idx = int(doc_tag)
                    text = (
                        actual_data[idx]
                        if actual_data and idx < len(actual_data)
                        else f"Document {doc_tag}"
                    )
                except (ValueError, IndexError):
                    text = f"Document {doc_tag}"
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  [{score:.4f}]  {preview}")

        elif method == "bert":
            results = sim.methodBert(data=data or None, query=query)
            print(f"BERT similarity results (query: '{query}'):")
            print("-" * 60)
            for score, text, _q in sorted(results, key=lambda x: x[0], reverse=True):
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  [{score:.4f}]  {preview}")

        elif method == "roberta":
            results = sim.methodRoBERTa(data=data or None, query=query)
            print(f"RoBERTa similarity results (query: '{query}'):")
            print("-" * 60)
            for score, text, _q in sorted(results, key=lambda x: x[0], reverse=True):
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  [{score:.4f}]  {preview}")

        else:
            print(
                f"Unknown method: '{method}'. Choose from: tfidf, nltk, bert, roberta",
                file=sys.stderr,
            )
            return 1

        return 0

    except ImportError as exc:
        print(f"Import error for method '{method}': {exc}", file=sys.stderr)
        print("Tip: 'bert' and 'roberta' require torch and transformers.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error running similarity method '{method}': {exc}", file=sys.stderr)
        return 1


def cmd_rag(args: argparse.Namespace) -> int:
    """Handle the 'rag' subcommand.

    Runs vector search + Reciprocal Rank Fusion on the built-in document set.

    Args:
        args: Parsed CLI arguments with query and top_k.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        vector_search, reciprocal_rank_fusion, ALL_DOCUMENTS = _import_rag_functions()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    query: str = args.query
    top_k: int | None = args.top_k

    try:
        # Run vector search for the query
        search_results = vector_search(query, ALL_DOCUMENTS, top_k=top_k)

        # Wrap in dict for RRF (single query)
        all_results: dict[str, dict[str, float]] = {query: search_results}
        fused = reciprocal_rank_fusion(all_results)

        # Trim to top_k after fusion
        if top_k is not None:
            fused = dict(list(fused.items())[:top_k])

        print(f"RAG results for query / תוצאות חיפוש: '{query}'")
        print("-" * 60)
        for rank, (doc_id, score) in enumerate(fused.items(), 1):
            doc_text = ALL_DOCUMENTS.get(doc_id, doc_id)
            preview = doc_text[:60] + "..." if len(doc_text) > 60 else doc_text
            print(f"  {rank}. [{score:.6f}]  {doc_id}: {preview}")

        return 0

    except Exception as exc:
        print(f"Error running RAG pipeline: {exc}", file=sys.stderr)
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the 'benchmark' subcommand.

    Runs the Hebrew NLP benchmark on the multilingual-mpnet model.

    Args:
        args: Parsed CLI arguments with format (table or json).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        run_benchmark, evaluate_benchmark, format_benchmark_report = _import_benchmark_functions()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_format: str = args.format

    try:
        print("Running Hebrew benchmark / מריץ מבחן עברי... (this may take a moment)")

        model_name = "multilingual-mpnet"
        results = run_benchmark(model_name)
        metrics = evaluate_benchmark(results)
        all_results = {model_name: metrics}

        if output_format == "json":
            # Include both metrics summary and per-pair scores
            output: dict[str, Any] = {
                "model": model_name,
                "metrics": {
                    k: (None if (isinstance(v, float) and v != v) else v)
                    for k, v in metrics.items()
                },
                "pairs": results,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))

        else:
            # Default: table format
            report = format_benchmark_report(all_results)
            print(report)

        return 0

    except Exception as exc:
        print(f"Error running benchmark: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "ZehutAI - Hebrew/Multilingual NLP CLI\n"
            "זהות AI - ממשק שורת פקודה לניתוח טקסט עברי ורב-לשוני"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples / דוגמאות:\n"
            "  python cli.py compare 'hello world' 'hi there'\n"
            "  python cli.py similarity --method tfidf --query 'NLP' --data 'text1' 'text2'\n"
            "  python cli.py rag --query 'climate change' --top-k 3\n"
            "  python cli.py benchmark --format table\n"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        help="Subcommand to run / פקודת משנה",
    )
    subparsers.required = True

    # -----------------------------------------------------------------------
    # compare subcommand
    # -----------------------------------------------------------------------
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two sentences / השוואת שתי משפטים",
        description=(
            "Compare two sentences using multilingual sentence-transformers.\n"
            "השוואת שני משפטים תוך שימוש במודל sentence-transformers רב-לשוני."
        ),
    )
    compare_parser.add_argument(
        "sentence1",
        help="First sentence / משפט ראשון",
    )
    compare_parser.add_argument(
        "sentence2",
        help="Second sentence / משפט שני",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # -----------------------------------------------------------------------
    # similarity subcommand
    # -----------------------------------------------------------------------
    sim_parser = subparsers.add_parser(
        "similarity",
        help="Run similarity method / הפעלת שיטת דמיון",
        description=(
            "Run a named similarity method on a collection of texts.\n"
            "הפעלת שיטת דמיון על קבוצת טקסטים."
        ),
    )
    sim_parser.add_argument(
        "--method",
        choices=["tfidf", "nltk", "bert", "roberta"],
        default="tfidf",
        help=(
            "Similarity method to use (default: tfidf) / שיטת דמיון (ברירת מחדל: tfidf)\n"
            "  tfidf   - TF-IDF cosine similarity (no model download required)\n"
            "  nltk    - Doc2Vec trained on the corpus\n"
            "  bert    - BERT embeddings (requires torch)\n"
            "  roberta - RoBERTa embeddings (requires torch)"
        ),
    )
    sim_parser.add_argument(
        "--data",
        nargs="+",
        metavar="TEXT",
        help="Corpus texts to compare against (space-separated) / טקסטים לפילוס",
    )
    sim_parser.add_argument(
        "--query",
        required=True,
        help="Query text / טקסט שאילתה",
    )
    sim_parser.set_defaults(func=cmd_similarity)

    # -----------------------------------------------------------------------
    # rag subcommand
    # -----------------------------------------------------------------------
    rag_parser = subparsers.add_parser(
        "rag",
        help="Run RAG vector search / הפעלת חיפוש RAG",
        description=(
            "Run vector search + Reciprocal Rank Fusion on the built-in documents.\n"
            "הפעלת חיפוש וקטורי + מיזוג דירוגים על מאגר המסמכים המובנה."
        ),
    )
    rag_parser.add_argument(
        "--query",
        required=True,
        help="Search query / שאילתת חיפוש",
    )
    rag_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        dest="top_k",
        help="Return only top K results / החזרת K התוצאות הטובות ביותר בלבד",
    )
    rag_parser.set_defaults(func=cmd_rag)

    # -----------------------------------------------------------------------
    # benchmark subcommand
    # -----------------------------------------------------------------------
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run Hebrew NLP benchmark / הרצת מבחן NLP עברי",
        description=(
            "Run the Hebrew sentence similarity benchmark on the multilingual-mpnet model.\n"
            "הרצת מבחן דמיון משפטים בעברית על מודל paraphrase-multilingual-mpnet-base-v2."
        ),
    )
    bench_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format / פורמט פלט (default: table)",
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the ZehutAI CLI.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Every subparser sets a 'func' default
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

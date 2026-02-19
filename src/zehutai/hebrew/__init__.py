"""Hebrew sub-package: embedding model benchmark suite."""

from zehutai.hebrew.hebrew_benchmark import (
    HEBREW_BENCHMARK_PAIRS,
    HEBREW_SENTENCE_PAIRS,
    MODEL_CONFIGS,
    evaluate_benchmark,
    format_benchmark_report,
    run_benchmark,
)

__all__ = [
    "HEBREW_BENCHMARK_PAIRS",
    "HEBREW_SENTENCE_PAIRS",
    "MODEL_CONFIGS",
    "evaluate_benchmark",
    "format_benchmark_report",
    "run_benchmark",
]

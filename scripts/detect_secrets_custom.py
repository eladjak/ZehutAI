"""Pre-commit hook: scan staged files for common secret patterns.

Cross-platform (pure Python) replacement for the old /bin/sh hook.
Referenced from .pre-commit-config.yaml (local repo hook, language: system).
Exits 1 (blocking the commit) if any potential secret is detected.
"""

from __future__ import annotations

import re
import sys

PATTERNS = [
    r"sk-[a-zA-Z0-9_-]{20,}",
    r"api[_-]?key\s*[:=]\s*[\"\x27][a-zA-Z0-9]",
    r"password\s*[:=]\s*[\"\x27][^\"\x27]{8,}",
    r"secret\s*[:=]\s*[\"\x27][a-zA-Z0-9]",
    r"token\s*[:=]\s*[\"\x27][a-zA-Z0-9]",
    r"OPENAI_API_KEY\s*=\s*[\"\x27]sk-",
    r"ghp_[a-zA-Z0-9]{36}",
    r"github_pat_[a-zA-Z0-9_]{82}",
    r"xox[bpors]-[a-zA-Z0-9-]+",
]


def main(paths: list[str]) -> int:
    """Scan the given file paths; return 1 if any secret pattern matches."""
    found = False
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        for pat in PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                print(f"ERROR: Secret pattern [{pat[:30]}...] in {path}")
                found = True
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# ZehutAI

AI-powered text analysis and semantic similarity system focused on Hebrew/multilingual content. The project explores various NLP approaches for comparing text similarity and implementing Retrieval-Augmented Generation (RAG) pipelines.

## Project Overview

ZehutAI ("Zehut" = Identity in Hebrew) investigates how different embedding models and NLP techniques can be used to:

- **Semantic similarity** - Compare text meaning across languages (Hebrew/English)
- **RAG (Retrieval-Augmented Generation)** - Query expansion + vector search + reciprocal rank fusion
- **Multiple model comparison** - BERT, RoBERTa, Doc2Vec, TF-IDF, sentence-transformers

### Key Model

The primary embedding model is [`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) - a multilingual sentence transformer supporting 50+ languages including Hebrew.

## Project Structure

```
ZehutAI/
  embeddings_comparison.py          # Standalone sentence similarity using sentence-transformers
  YehoshuaSimilarityComparisons/
    embeddings_comparison.py        # Extended version with warning suppression
    sim.py                          # Similarity class: NLTK/Doc2Vec, TF-IDF, BERT, RoBERTa methods
    plotting.py                     # Model runner (loads BERT and runs similarity comparison)
    main.py                         # Entry point for similarity comparisons
    rag.py                          # RAG fusion pipeline: query expansion + vector search + RRF
    ragtest.py                      # Test script for DictaLM 2.0 Hebrew language model
    simple_rag.py                   # Hebrew RAG with DictaLM/mt5-xl-heq models and political documents
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU recommended (CPU fallback available)

### Installation

```bash
# Clone the repository
git clone https://github.com/eladjak/ZehutAI.git
cd ZehutAI

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if using NLTK methods)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Running

```bash
# Basic sentence similarity comparison
python embeddings_comparison.py

# Full similarity method comparison (BERT, RoBERTa, Doc2Vec, TF-IDF)
cd YehoshuaSimilarityComparisons
python main.py

# RAG fusion pipeline
python rag.py
```

## Technologies

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Multilingual sentence embeddings |
| `transformers` | BERT, RoBERTa, DictaLM, mt5 models |
| `torch` | Deep learning framework |
| `gensim` | Doc2Vec embeddings |
| `nltk` | Tokenization |
| `scikit-learn` | TF-IDF, cosine similarity |
| `scipy` | Distance metrics |
| `numpy` | Numerical computing |
| `matplotlib` | Visualization |

## Development

```bash
# Type checking
python -m mypy .

# Run tests
python -m pytest

# Linting
ruff check .
```

## Web UI — ממשק אינטרנטי (עברית)

בתיקיית [`webapp/`](webapp/) יש ממשק web מלא בעברית (RTL, dark) מעל מנוע המחקר —
מסך השוואת משפטים (עם מד דמיון מונפש), דירוג דמיון, חיפוש RAG ובנצ'מרק עברי.
המנוע לא שוכתב: השרת (FastAPI) מייבא את הפונקציות הקיימות כפי שהן.

**הרצה מקומית:**

```bash
pip install -r requirements.txt fastapi "uvicorn[standard]"
export ZEHUTAI_TOKEN=dev-token   # Windows: set ZEHUTAI_TOKEN=dev-token
uvicorn webapp.server:app --port 3980
# ואז לגלוש אל: http://127.0.0.1:3980/zehutai?k=dev-token
```

**המופע החי** רץ על השרת של אלעד מאחורי קישור פרטי עם טוקן
(`https://hub.eladjak.com/zehutai?k=<token>`) — הטוקן נמסר אישית, לא נמצא בריפו.
עדכון המופע החי לאחר push:

```bash
cd /opt/zehutai && git pull && systemctl restart zehutai-web
```

פרטי ארכיטקטורה, endpoints והנחות v1 — ב-[`webapp/README.md`](webapp/README.md).

## Status

Early-stage research project. See [PROGRESS.md](PROGRESS.md) for current status and next steps.

---

⭐ If you find this useful, please star the repo!

*[README בעברית](README.he.md)*

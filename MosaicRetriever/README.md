# MosaicRetriever (Phase-0)

MosaicRetriever is a minimal, venv-friendly retrieval playground that wires:
- BM25 via Pyserini (prebuilt BEIR FEVER index)
- Dense retrieval via Sentence-BERT + FAISS (cosine/IP)

Phase-0 focuses on a clean baseline: a single API that can run both lexical and dense retrieval on the BEIR FEVER dataset. No UI yet.

## Setup

- Python 3.11 (works on 3.10 too)
- Use venv/virtualenv (no conda assumptions)

Commands:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Build + Sanity

- Build FAISS (downloads FEVER on first run):
```
python scripts/build_faiss.py
```

- Quick sanity (runs one FEVER test query, prints top-3 BM25 and Dense):
```
python scripts/quick_sanity.py
```

Notes:
- BM25 uses Pyserini prebuilt index: `beir-v1.0.0-fever-flat`.
- First run will download FEVER dataset into `data/cache/fever` and the Sentence-BERT model.
- If the Pyserini prebuilt index can’t be opened (e.g., missing Java), you’ll get a clear error suggesting to install Java or switch to a local Lucene build.

## Layout

```
MosaicRetriever/
  .gitignore
  README.md
  requirements.txt
  scripts/
    build_faiss.py
    quick_sanity.py
  src/
    __init__.py
    datasets.py
    bm25.py
    dense.py
    api.py
  data/
    indexes/
      .keep
      faiss/
    cache/
```

## Next Phases (placeholder)
- HyDE prompting
- RRF fusion
- Learning-to-Rank (LTR)

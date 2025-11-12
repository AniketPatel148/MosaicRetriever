"""MosaicRetriever package (Phase-0).

Provides:
- Dataset download utilities for BEIR FEVER.
- BM25 search via Pyserini prebuilt index.
- Dense indexing/search with Sentence-BERT + FAISS.
- A small API (UniSearchAPI) to query both.
"""

__all__ = [
    "datasets",
    "bm25",
    "dense",
    "api",
]


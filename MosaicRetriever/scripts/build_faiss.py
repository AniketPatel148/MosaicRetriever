#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import DEFAULT_FAISS_DIR
from src.datasets import ensure_beir_fever
from src.dense import DenseConfig, DenseIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save FAISS index for FEVER.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_FAISS_DIR, help="Output directory for FAISS index.")
    parser.add_argument("--model", type=str, default="multi-qa-mpnet-base-dot-v1", help="Sentence-BERT model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of docs to index (for quick tests).")
    args = parser.parse_args()

    corpus, _, _ = ensure_beir_fever()
    config = DenseConfig(model_name=args.model, batch_size=args.batch_size)
    indexer = DenseIndexer(index_dir=args.index_dir, config=config)

    print(f"Building FAISS at {args.index_dir} using model '{args.model}' ...")
    def gen_docs():
        for did, obj in corpus.items():
            yield did, obj.get("title", ""), obj.get("text", "")

    indexer.build_from_corpus(gen_docs(), limit=args.limit)
    indexer.save()
    print("Saved index.faiss, docids.txt, meta.json")


if __name__ == "__main__":
    main()

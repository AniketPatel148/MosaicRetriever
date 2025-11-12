#!/usr/bin/env python
from __future__ import annotations

from src.api import UniSearchAPI
from src.datasets import ensure_beir_fever, first_test_query


def _snippet(text: str, n: int = 120) -> str:
    t = " ".join(text.split())
    return t[:n] + ("..." if len(t) > n else "")


def main() -> None:
    corpus, queries, qrels = ensure_beir_fever()
    qid, query = first_test_query(queries)
    print("Query:")
    print(query)

    api = UniSearchAPI(corpus=corpus)

    bm25_results = api.lexical_search(query, k=3)
    dense_results = api.dense_search(query, k=3)

    print("BM25 top-3:")
    for docid, score in bm25_results:
        text = api.get_doc(docid)
        print(f"[{score:.4f}] {docid} :: {_snippet(text)}")

    print("Dense top-3:")
    for docid, score in dense_results:
        text = api.get_doc(docid)
        print(f"[{score:.4f}] {docid} :: {_snippet(text)}")


if __name__ == "__main__":
    main()

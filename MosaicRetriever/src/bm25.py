from __future__ import annotations

import json
from typing import Iterable, List, Optional, Tuple


PREBUILT_INDEX = "beir-v1.0.0-fever-flat"


class BM25Searcher:
    """BM25 search via Pyserini prebuilt BEIR FEVER index.

    Methods:
    - search(query, k) -> list[(docid, score)]
    - set_bm25(k1, b)
    - get_doc(docid) -> "title\ntext" (best-effort from stored raw)
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        self.k1 = k1
        self.b = b
        try:
            from pyserini.search.lucene import LuceneSearcher

            self._searcher = LuceneSearcher.from_prebuilt_index(PREBUILT_INDEX)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Failed to initialize Pyserini prebuilt index 'beir-v1.0.0-fever-flat'.\n"
                "Please ensure Java (JRE/JDK) is installed and available on PATH.\n"
                "Alternatively, switch to a local Lucene index build and point Pyserini to it.\n"
                f"Underlying error: {e}"
            ) from e
        self._searcher.set_bm25(self.k1, self.b)

    def set_bm25(self, k1: float, b: float) -> None:
        self.k1 = k1
        self.b = b
        self._searcher.set_bm25(k1, b)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        hits = self._searcher.search(query, k=k)
        return [(h.docid, float(h.score)) for h in hits]

    def get_doc(self, docid: str) -> str:
        """Return "title\ntext" for a document id using stored raw or contents."""
        doc = self._searcher.doc(docid)
        if doc is None:
            return ""
        raw = doc.raw()
        if raw:
            try:
                obj = json.loads(raw)
                title = obj.get("title", "")
                text = obj.get("text") or obj.get("contents") or ""
                return f"{title}\n{text}".strip()
            except Exception:  # noqa: BLE001
                pass
        contents = doc.contents().strip() if doc.contents() else ""
        return contents

    def iter_docids(self, limit: Optional[int] = None) -> Iterable[str]:
        """Yield docids from the underlying index (best-effort)."""
        try:
            from pyserini.index.lucene import IndexReader

            reader = IndexReader.from_prebuilt_index(PREBUILT_INDEX)
            count = 0
            for did in reader.docids():  # type: ignore[attr-defined]
                yield did
                count += 1
                if limit is not None and count >= limit:
                    break
        except Exception:
            return

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .bm25 import BM25Searcher
from .dense import DenseConfig, DenseIndexer, DenseSearcher


DEFAULT_FAISS_DIR = Path(__file__).resolve().parents[1] / "data" / "indexes" / "faiss"


class UniSearchAPI:
    """Unified API to run BM25 and Dense retrieval.

    __init__(corpus, faiss_dir):
    - Accepts in-memory corpus dict[docid] -> {title, text}.
    - Initializes BM25Searcher over Pyserini prebuilt FEVER index.
    - Builds or loads FAISS index under faiss_dir, then creates DenseSearcher.
    """

    def __init__(
        self,
        corpus: Dict[str, Dict[str, str]],
        faiss_dir: Path | str = DEFAULT_FAISS_DIR,
        dense_config: Optional[DenseConfig] = None,
        dense_limit: Optional[int] = None,
    ) -> None:
        self.corpus = corpus
        self.faiss_dir = Path(faiss_dir)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.dense_config = dense_config or DenseConfig()

        # BM25
        self.bm25 = BM25Searcher()

        # Dense: load or build FAISS
        if (self.faiss_dir / "index.faiss").exists():
            self.dense = DenseSearcher(index_dir=self.faiss_dir, config=self.dense_config)
        else:
            self._build_dense_index(limit=dense_limit)
            self.dense = DenseSearcher(index_dir=self.faiss_dir, config=self.dense_config)

    def _build_dense_index(self, limit: Optional[int] = None) -> None:
        indexer = DenseIndexer(index_dir=self.faiss_dir, config=self.dense_config)
        def gen_docs():
            for did, obj in self.corpus.items():
                yield did, obj.get("title", ""), obj.get("text", "")
        indexer.build_from_corpus(gen_docs(), limit=limit)
        indexer.save()

    def lexical_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        return self.bm25.search(query, k=k)

    def dense_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        return self.dense.search(query, k=k)

    def get_doc(self, docid: str) -> str:
        obj = self.corpus.get(docid)
        if not obj:
            return ""
        title = obj.get("title", "")
        text = obj.get("text", "")
        return (f"{title}\n{text}").strip()

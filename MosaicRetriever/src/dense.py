from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class DenseConfig:
    model_name: str = "multi-qa-mpnet-base-dot-v1"
    batch_size: int = 64
    # Force CPU by default to avoid occasional MPS/accelerator segfaults on macOS.
    device: Optional[str] = "cpu"


class DenseIndexer:
    """Builds and persists a FAISS (IP) index using Sentence-BERT embeddings.

    - Encodes (title + text) and L2-normalizes embeddings so that IP == cosine.
    - Saves: index.faiss, docids.txt, meta.json
    """

    def __init__(self, index_dir: Path, config: Optional[DenseConfig] = None) -> None:
        self.index_dir = Path(index_dir)
        self.config = config or DenseConfig()
        self.index: Optional[faiss.Index] = None
        self.docids: List[str] = []
        self.model: Optional[SentenceTransformer] = None

    def _ensure_dir(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self.model

    @staticmethod
    def _merge_text(title: str, text: str) -> str:
        title = title.strip()
        text = text.strip()
        if title and text:
            return f"{title}\n{text}"
        return title or text

    def build_from_corpus(
        self,
        docs: Iterable[Tuple[str, str, str]],
        limit: Optional[int] = None,
        chunk_size: int = 8192,
    ) -> None:
        """Build FAISS from an iterable of (docid, title, text) using streaming batches."""
        self._ensure_dir()
        model = self._load_model()

        index: Optional[faiss.IndexFlatIP] = None
        self.docids = []

        buf_texts: List[str] = []
        buf_ids: List[str] = []

        def flush() -> None:
            nonlocal index, buf_texts, buf_ids
            if not buf_texts:
                return
            embeddings = model.encode(
                buf_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.asarray(embeddings)
            embeddings = embeddings.astype(np.float32)
            if index is None:
                index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.docids.extend(buf_ids)
            buf_texts = []
            buf_ids = []

        for i, (docid, title, text) in enumerate(docs):
            buf_ids.append(docid)
            buf_texts.append(self._merge_text(title, text))
            if len(buf_texts) >= chunk_size:
                flush()
            if limit is not None and (i + 1) >= limit:
                break

        flush()
        if index is None:
            raise ValueError("No documents provided to build FAISS index.")
        self.index = index

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index not built; call build_from_corpus first.")
        self._ensure_dir()
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "docids.txt", "w", encoding="utf-8") as f:
            for did in self.docids:
                f.write(did + "\n")
        meta = {
            "model": self.config.model_name,
            "size": len(self.docids),
            "type": "IndexFlatIP",
        }
        with open(self.index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.docids = []
        with open(self.index_dir / "docids.txt", "r", encoding="utf-8") as f:
            for line in f:
                self.docids.append(line.strip())


class DenseSearcher:
    """Search interface over a saved DenseIndexer (FAISS + docids) and model."""

    def __init__(self, index_dir: Path, config: Optional[DenseConfig] = None) -> None:
        self.index_dir = Path(index_dir)
        self.config = config or DenseConfig()
        self.indexer = DenseIndexer(index_dir=self.index_dir, config=self.config)
        self.indexer.load()
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self.indexer.index is None:
            raise RuntimeError("Dense index not loaded.")
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, idx = self.indexer.index.search(q, k)
        scores = scores[0]
        idx = idx[0]
        results: List[Tuple[str, float]] = []
        for i, s in zip(idx, scores):
            if i < 0 or i >= len(self.indexer.docids):
                continue
            results.append((self.indexer.docids[i], float(s)))
        return results

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import zipfile


# Directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FEVER_DIR = CACHE_DIR / "fever"


Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _beir_files_exist(base: Path) -> bool:
    return (
        (base / "corpus.jsonl").exists()
        and (base / "queries.jsonl").exists()
        and (base / "qrels" / "test.tsv").exists()
    )


def _parse_corpus_jsonl(fp: io.TextIOBase) -> Corpus:
    corpus: Corpus = {}
    for line in fp:
        if not line.strip():
            continue
        obj = json.loads(line)
        did = str(obj.get("_id") or obj.get("id"))
        title = str(obj.get("title", ""))
        text = str(obj.get("text") or obj.get("contents") or "")
        corpus[did] = {"title": title, "text": text}
    return corpus


def _parse_queries_jsonl(fp: io.TextIOBase) -> Queries:
    queries: Queries = {}
    for line in fp:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = str(obj.get("_id") or obj.get("id"))
        text = str(obj.get("text") or obj.get("query") or "")
        queries[qid] = text
    return queries


def _parse_qrels_tsv(fp: io.TextIOBase) -> Qrels:
    qrels: Qrels = {}
    for line in fp:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
        qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def _load_local_files(cache_dir: Path) -> Tuple[Corpus, Queries, Qrels]:
    corpus_path = cache_dir / "corpus.jsonl"
    queries_path = cache_dir / "queries.jsonl"
    qrels_path = cache_dir / "qrels" / "test.tsv"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = _parse_corpus_jsonl(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        all_queries = _parse_queries_jsonl(f)
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = _parse_qrels_tsv(f)

    # Keep only test queries (those present in qrels/test.tsv)
    queries = {qid: all_queries.get(qid, "") for qid in qrels.keys()}
    return corpus, queries, qrels


def _persist_beir_files(cache_dir: Path, corpus: Corpus, all_queries: Queries, qrels: Qrels) -> None:
    _ensure_dir(cache_dir)
    qrels_dir = cache_dir / "qrels"
    _ensure_dir(qrels_dir)

    corpus_path = cache_dir / "corpus.jsonl"
    queries_path = cache_dir / "queries.jsonl"
    qrels_path = qrels_dir / "test.tsv"

    with open(corpus_path, "w", encoding="utf-8") as f:
        for did, obj in corpus.items():
            f.write(json.dumps({"_id": did, "title": obj["title"], "text": obj["text"]}) + "\n")
    with open(queries_path, "w", encoding="utf-8") as f:
        for qid, text in all_queries.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    with open(qrels_path, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for did, rel in docs.items():
                f.write(f"{qid} 0 {did} {rel}\n")


def _load_from_zip_bytes(raw: bytes, cache_dir: Path) -> Tuple[Corpus, Queries, Qrels]:
    """Parse BEIR-format FEVER zip bytes, persist to cache_dir, and return dicts."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = set(zf.namelist())

        def find(name: str) -> str:
            if name in names:
                return name
            alt = f"fever/{name}"
            if alt in names:
                return alt
            raise FileNotFoundError(f"{name} not found in FEVER zip")

        corpus_name = find("corpus.jsonl")
        queries_name = find("queries.jsonl")
        qrels_name = find("qrels/test.tsv")

        with zf.open(corpus_name, "r") as f:
            corpus = _parse_corpus_jsonl(io.TextIOWrapper(f, encoding="utf-8"))
        with zf.open(queries_name, "r") as f:
            all_queries = _parse_queries_jsonl(io.TextIOWrapper(f, encoding="utf-8"))
        with zf.open(qrels_name, "r") as f:
            qrels = _parse_qrels_tsv(io.TextIOWrapper(f, encoding="utf-8"))

    queries = {qid: all_queries.get(qid, "") for qid in qrels.keys()}
    _persist_beir_files(cache_dir, corpus, all_queries, qrels)
    return corpus, queries, qrels


def _load_from_ir_datasets(cache_dir: Path) -> Tuple[Corpus, Queries, Qrels] | None:
    """Load FEVER via ir_datasets (beir/fever/dev) if available."""
    try:
        import ir_datasets  # type: ignore
    except ImportError:
        return None

    try:
        ds = ir_datasets.load("beir/fever/dev")
    except Exception:
        return None

    corpus: Corpus = {}
    for doc in ds.docs_iter():
        did = str(getattr(doc, "doc_id", None) or getattr(doc, "id", None))
        title = str(getattr(doc, "title", "") or "")
        text = str(getattr(doc, "text", "") or getattr(doc, "contents", "") or "")
        corpus[did] = {"title": title, "text": text}

    all_queries: Queries = {}
    for q in ds.queries_iter():
        qid = str(getattr(q, "query_id", None) or getattr(q, "id", None))
        text = str(getattr(q, "text", "") or getattr(q, "query", "") or "")
        all_queries[qid] = text

    qrels: Qrels = {}
    for qr in ds.qrels_iter():
        qid = str(getattr(qr, "query_id", None) or getattr(qr, "id", None))
        did = str(getattr(qr, "doc_id", None) or getattr(qr, "corpus_id", None))
        rel = int(getattr(qr, "relevance", None) or getattr(qr, "score", None) or getattr(qr, "rel", 1) or 1)
        qrels.setdefault(qid, {})[did] = rel

    queries = {qid: all_queries.get(qid, "") for qid in qrels.keys()}
    _persist_beir_files(cache_dir, corpus, all_queries, qrels)
    return corpus, queries, qrels


def ensure_beir_fever(cache_dir: Path = FEVER_DIR) -> Tuple[Corpus, Queries, Qrels]:
    """Ensure BEIR FEVER exists under data/cache/fever and return in-memory dicts.

    Returns:
    - corpus: dict[doc_id] -> {"title": str, "text": str}
    - queries: dict[qid] -> str (test split only, i.e., filtered by qrels/test.tsv)
    - qrels: dict[qid] -> dict[doc_id] -> int

    Behavior:
    - If files exist locally, loads them.
    - Otherwise tries to load via ir_datasets (beir/fever/dev), persists to cache, and returns parsed dicts.
    """
    _ensure_dir(cache_dir)
    # First, prefer any already-extracted BEIR-format files.
    if _beir_files_exist(cache_dir):
        return _load_local_files(cache_dir)

    local_dir_env = os.environ.get("FEVER_DATA_DIR")
    local_dirs = [
        Path(local_dir_env) if local_dir_env else None,
        DATA_DIR / "fever",
    ]
    for candidate in local_dirs:
        if candidate and candidate != cache_dir and _beir_files_exist(candidate):
            return _load_local_files(candidate)

    # Next, try local zip files to avoid downloading when provided manually.
    zip_env = os.environ.get("FEVER_ZIP_PATH")
    zip_candidates = [
        Path(zip_env) if zip_env else None,
        DATA_DIR / "fever.zip",
        cache_dir / "fever.zip",
    ]
    for zp in zip_candidates:
        if zp and zp.is_file():
            raw = zp.read_bytes()
            return _load_from_zip_bytes(raw, cache_dir)

    # Try ir_datasets (preferred).
    ir_data = _load_from_ir_datasets(cache_dir)
    if ir_data:
        return ir_data

    raise RuntimeError(
        "FEVER dataset not found. Provide BEIR-format files under data/cache/fever "
        "or a local zip at data/fever.zip/FEVER_ZIP_PATH, or install ir-datasets to "
        "load automatically (beir/fever/dev)."
    )


def first_test_query(queries: Queries) -> Tuple[str, str]:
    """Return one (qid, text) pair from test queries dict."""
    for qid, text in queries.items():
        return qid, text
    raise RuntimeError("No test queries found in FEVER dataset.")

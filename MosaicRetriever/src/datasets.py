from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

from tqdm import tqdm
from urllib.request import urlopen
import zipfile


# Directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FEVER_DIR = CACHE_DIR / "fever"


# Source URL
FEVER_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip"


Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_zip_to_memory(url: str) -> bytes:
    """Download URL into memory and return raw bytes."""
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1024 * 64
        data = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading FEVER") as pbar:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                data.write(buf)
                pbar.update(len(buf))
        return data.getvalue()


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


def ensure_beir_fever(cache_dir: Path = FEVER_DIR) -> Tuple[Corpus, Queries, Qrels]:
    """Ensure BEIR FEVER exists under data/cache/fever and return in-memory dicts.

    Returns:
    - corpus: dict[doc_id] -> {"title": str, "text": str}
    - queries: dict[qid] -> str (test split only, i.e., filtered by qrels/test.tsv)
    - qrels: dict[qid] -> dict[doc_id] -> int

    Behavior:
    - If files exist locally, loads them.
    - Otherwise downloads the dataset zip to memory, extracts required files in memory,
      writes them to cache_dir for idempotence, and returns parsed dicts.
    """
    _ensure_dir(cache_dir)
    corpus_path = cache_dir / "corpus.jsonl"
    queries_path = cache_dir / "queries.jsonl"
    qrels_dir = cache_dir / "qrels"
    qrels_path = qrels_dir / "test.tsv"

    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        return _load_local_files(cache_dir)

    # Download to memory and parse/write out
    raw = _download_zip_to_memory(FEVER_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Paths can be with or without leading folder 'fever/'
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

    # Filter queries to test split
    queries = {qid: all_queries.get(qid, "") for qid in qrels.keys()}

    # Persist to disk for idempotence
    _ensure_dir(qrels_dir)
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

    return corpus, queries, qrels


def first_test_query(queries: Queries) -> Tuple[str, str]:
    """Return one (qid, text) pair from test queries dict."""
    for qid, text in queries.items():
        return qid, text
    raise RuntimeError("No test queries found in FEVER dataset.")

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.build_index import DEFAULT_SENTENCE_MODEL, _encode_with_transformer
from systems.system_b_utils import normalize_text

INDEX_ROOT = PROJECT_ROOT / "artifacts" / "retrieval" / "index"


@lru_cache(maxsize=4)
def load_index_config(split: str = "standard") -> dict[str, Any]:
    path = INDEX_ROOT / split / "index_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Retrieval index config not found at {path}. Run retrieval/build_index.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_jsonl(path_str: str) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / path_str
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=8)
def _load_numpy(path_str: str) -> np.ndarray:
    return np.load(PROJECT_ROOT / path_str)


@lru_cache(maxsize=8)
def _load_joblib(path_str: str):
    return joblib.load(PROJECT_ROOT / path_str)


@lru_cache(maxsize=4)
def _load_pool_bundle(split: str, pool_name: str) -> dict[str, Any]:
    config = load_index_config(split)
    pool_cfg = config["pools"][pool_name]
    bundle: dict[str, Any] = {
        "backend": config["backend"],
        "metadata": _load_jsonl(pool_cfg["metadata_path"]),
        "model_name": config.get("model_name", DEFAULT_SENTENCE_MODEL),
    }
    if config["backend"] == "transformer":
        bundle["embeddings"] = _load_numpy(pool_cfg["embeddings_path"])
    else:
        bundle["matrix"] = _load_joblib(pool_cfg["matrix_path"])
        bundle["vectorizer"] = _load_joblib(pool_cfg["vectorizer_path"])
    return bundle


def _pool_name_from_direction(direction: str) -> str:
    if direction == "n2s":
        return "sarcastic"
    if direction == "s2n":
        return "nonsarcastic"
    raise ValueError(f"Unknown direction: {direction}")


def retrieve_neighbors(
    query_text: str,
    direction: str,
    k: int = 5,
    split: str = "standard",
    exclude_ids: set[str] | None = None,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    if not query_text or not str(query_text).strip():
        return []

    pool_name = _pool_name_from_direction(direction)
    bundle = _load_pool_bundle(split, pool_name)
    metadata = bundle["metadata"]
    exclude_ids = exclude_ids or set()

    if bundle["backend"] == "transformer":
        query_vec = _encode_with_transformer([query_text], model_name=bundle["model_name"], batch_size=1)
        scores = cosine_similarity(query_vec, bundle["embeddings"])[0]
    else:
        query_vec = bundle["vectorizer"].transform([query_text])
        scores = cosine_similarity(query_vec, bundle["matrix"])[0]

    ranked_indices = np.argsort(-scores)
    seen_norm_texts: set[str] = set()
    results: list[dict[str, Any]] = []

    for idx in ranked_indices:
        meta = metadata[int(idx)]
        if meta["id"] in exclude_ids:
            continue
        score = float(scores[int(idx)])
        if min_similarity is not None and score < min_similarity:
            continue

        norm = normalize_text(meta["text"])
        if norm in seen_norm_texts:
            continue
        seen_norm_texts.add(norm)

        results.append(
            {
                "id": meta["id"],
                "text": meta["text"],
                "label": int(meta["label"]),
                "publisher": meta.get("publisher"),
                "anchors": meta.get("anchors", {}),
                "similarity": score,
                "split": meta.get("split", "train"),
                "pool_name": pool_name,
            }
        )
        if len(results) >= k:
            break

    return results

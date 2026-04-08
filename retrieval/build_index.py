from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_b_utils import load_anchored_split_frames

DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_ROOT = PROJECT_ROOT / "artifacts" / "retrieval" / "index"


@lru_cache(maxsize=2)
def _load_transformer_model(model_name: str = DEFAULT_SENTENCE_MODEL):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_with_transformer(texts: list[str], model_name: str, batch_size: int = 128) -> np.ndarray:
    import torch

    tokenizer, model = _load_transformer_model(model_name)
    chunks: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=96)
        with torch.no_grad():
            outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        chunks.append(normalized.cpu().numpy().astype(np.float32))

    if not chunks:
        return np.zeros((0, 384), dtype=np.float32)
    return np.vstack(chunks)


def _encode_with_tfidf(texts: list[str], max_features: int = 50000):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def _make_metadata_records(df) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": row["id"],
                "text": row["text"],
                "label": int(row["label"]),
                "publisher": row.get("publisher"),
                "anchors": row.get("anchors", {}),
                "split": "train",
            }
        )
    return records


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_style_pool(train_df, target_label: int):
    return train_df[train_df["label"].astype(int) == int(target_label)].reset_index(drop=True)


def build_retrieval_index(
    split: str = "standard",
    model_name: str = DEFAULT_SENTENCE_MODEL,
    backend: str = "auto",
    force: bool = False,
    batch_size: int = 128,
) -> dict[str, Any]:
    if backend not in {"auto", "transformer", "tfidf"}:
        raise ValueError("backend must be one of: auto, transformer, tfidf")

    train_df, _, _ = load_anchored_split_frames(split=split)
    output_dir = INDEX_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "index_config.json"
    if config_path.exists() and not force:
        return json.loads(config_path.read_text(encoding="utf-8"))

    sarcastic_df = _build_style_pool(train_df, target_label=1)
    nonsarcastic_df = _build_style_pool(train_df, target_label=0)

    backend_used = backend
    if backend == "auto":
        try:
            _ = _load_transformer_model(model_name)
            backend_used = "transformer"
        except Exception:
            backend_used = "tfidf"

    style_frames = {
        "sarcastic": sarcastic_df,
        "nonsarcastic": nonsarcastic_df,
    }

    config: dict[str, Any] = {
        "split_strategy": split,
        "backend": backend_used,
        "model_name": model_name,
        "batch_size": batch_size,
        "pools": {},
    }

    for pool_name, pool_df in style_frames.items():
        texts = pool_df["text"].astype(str).tolist()
        metadata = _make_metadata_records(pool_df)
        meta_path = output_dir / f"train_{pool_name}_metadata.jsonl"
        _write_jsonl(metadata, meta_path)

        if backend_used == "transformer":
            embeddings = _encode_with_transformer(texts, model_name=model_name, batch_size=batch_size)
            emb_path = output_dir / f"train_{pool_name}_embeddings.npy"
            np.save(emb_path, embeddings)
            pool_config = {
                "pool_name": pool_name,
                "count": len(texts),
                "metadata_path": str(meta_path.relative_to(PROJECT_ROOT)),
                "embeddings_path": str(emb_path.relative_to(PROJECT_ROOT)),
            }
        else:
            matrix, vectorizer = _encode_with_tfidf(texts)
            mat_path = output_dir / f"train_{pool_name}_tfidf_matrix.joblib"
            vec_path = output_dir / f"train_{pool_name}_tfidf_vectorizer.joblib"
            joblib.dump(matrix, mat_path)
            joblib.dump(vectorizer, vec_path)
            pool_config = {
                "pool_name": pool_name,
                "count": len(texts),
                "metadata_path": str(meta_path.relative_to(PROJECT_ROOT)),
                "matrix_path": str(mat_path.relative_to(PROJECT_ROOT)),
                "vectorizer_path": str(vec_path.relative_to(PROJECT_ROOT)),
            }

        config["pools"][pool_name] = pool_config

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved retrieval index config -> {config_path}")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build System C retrieval indices from train headlines only.")
    parser.add_argument("--split", default="standard", choices=["standard", "topic_hard"])
    parser.add_argument("--backend", default="auto", choices=["auto", "transformer", "tfidf"])
    parser.add_argument("--model-name", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = build_retrieval_index(
        split=args.split,
        model_name=args.model_name,
        backend=args.backend,
        force=args.force,
        batch_size=args.batch_size,
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

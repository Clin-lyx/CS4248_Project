"""Simple semantic similarity scoring for news headlines."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def _load_model(model_name: str = _DEFAULT_MODEL):
    """Load and cache tokenizer/model on the best available device."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(_DEVICE)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _fallback_similarity(headline_a: str, headline_b: str) -> float:
    """Fallback if transformer model is unavailable (e.g., no internet)."""
    matrix = TfidfVectorizer().fit_transform([headline_a, headline_b])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(max(0.0, min(1.0, score)))


def semantic_similarity(headline_a: str, headline_b: str, model_name: str = _DEFAULT_MODEL) -> float:
    """Return a semantic similarity score in [0, 1] for two headlines."""
    if not headline_a or not headline_b:
        raise ValueError("Both headlines must be non-empty strings.")

    try:
        tokenizer, model = _load_model(model_name)
        dev = next(model.parameters()).device
        inputs = tokenizer([headline_a, headline_b], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        score = cosine_similarity(
            embeddings[0:1].cpu().numpy(),
            embeddings[1:2].cpu().numpy(),
        )[0][0]
    except Exception:
        score = _fallback_similarity(headline_a, headline_b)

    return float(max(0.0, min(1.0, (score + 1.0) / 2.0)))


def batch_semantic_similarity(
    texts_a: list[str],
    texts_b: list[str],
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Batched pairwise cosine similarity, returned in [0, 1].

    Much faster than calling semantic_similarity() in a loop because all texts
    are encoded in large GPU batches with a single tokenizer pass each.
    """

    if len(texts_a) != len(texts_b):
        raise ValueError("texts_a and texts_b must have the same length")

    tokenizer, model = _load_model(model_name)
    dev = next(model.parameters()).device

    def _encode(texts: list[str]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            embs = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    embs_a = _encode(texts_a)
    embs_b = _encode(texts_b)

    raw = np.array([
        cosine_similarity([a], [b])[0, 0]
        for a, b in zip(embs_a, embs_b)
    ])
    return np.clip((raw + 1.0) / 2.0, 0.0, 1.0)


if __name__ == "__main__":
    h1 = "Government unveils new climate plan for coastal cities"
    h2 = "Trump announces brand new and wonderful environmental strategy for his coastal properties"
    print(f"Similarity: {semantic_similarity(h1, h2):.4f}")


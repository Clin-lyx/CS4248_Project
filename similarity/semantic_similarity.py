"""Simple semantic similarity scoring for news headlines."""

from __future__ import annotations

from functools import lru_cache

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str = _DEFAULT_MODEL):
    """Load and cache tokenizer/model so repeated calls stay fast."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
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
        inputs = tokenizer([headline_a, headline_b], padding=True, truncation=True, return_tensors="pt")
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


if __name__ == "__main__":
    h1 = "Government unveils new climate plan for coastal cities"
    h2 = "Trump announces brand new and wonderful environmental strategy for his coastal properties"
    print(f"Similarity: {semantic_similarity(h1, h2):.4f}")


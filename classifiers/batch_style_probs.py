"""
Batch sarcasm probabilities for evaluation notebooks.

Uses checkpoints under artifacts/classifiers/{logreg,rnn,transformer}/{split}/,
matching the split strategy used for evaluation (standard vs topic_hard).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

# Prefer the strongest critic as the canonical flip_* columns when multiple exist.
DETECTOR_ORDER: tuple[str, ...] = ("transformer", "rnn", "logreg")


def classifier_dir(project_root: Path, kind: str, split: str) -> Path:
    return project_root / "artifacts" / "classifiers" / kind / split


def available_detectors(project_root: Path, split: str) -> dict[str, Path | None]:
    """Return detector name -> checkpoint directory if present, else None."""
    out: dict[str, Path | None] = {}
    root = project_root.resolve()

    lr = classifier_dir(root, "logreg", split) / "model.joblib"
    out["logreg"] = lr.parent if lr.is_file() else None

    rnn = classifier_dir(root, "rnn", split) / "model.pt"
    out["rnn"] = rnn.parent if rnn.is_file() else None

    tr = classifier_dir(root, "transformer", split)
    out["transformer"] = tr if (tr / "config.json").is_file() else None

    return out


def pick_primary_detector(avail: dict[str, Path | None]) -> str | None:
    for name in DETECTOR_ORDER:
        if avail.get(name) is not None:
            return name
    return None


def _probs_from_predict_dicts(outputs: list[dict]) -> np.ndarray:
    return np.array([float(o["prob_sarcastic"]) for o in outputs], dtype=np.float64)


def batch_prob_sarcastic(
    texts: list[str],
    detector: str,
    *,
    project_root: Path,
    split: str,
    transformer_batch_size: int = 32,
    rnn_batch_size: int = 64,
) -> np.ndarray:
    """Run a single detector on all texts; returns shape (len(texts),)."""
    from classifiers import logreg_classifier, rnn_classifier, transformer_classifier

    avail = available_detectors(project_root, split)
    path = avail.get(detector)
    if path is None:
        raise FileNotFoundError(f"Detector {detector!r} not available for split {split!r}")

    if detector == "logreg":
        out = logreg_classifier.predict(path, texts=texts)
        return _probs_from_predict_dicts(out)
    if detector == "rnn":
        out = rnn_classifier.predict(path, texts=texts, batch_size=rnn_batch_size)
        return _probs_from_predict_dicts(out)
    if detector == "transformer":
        out = transformer_classifier.predict(
            path,
            texts=texts,
            batch_size=transformer_batch_size,
        )
        return _probs_from_predict_dicts(out)
    raise ValueError(f"Unknown detector: {detector}")


def batch_probs_for_all_detectors(
    texts: list[str],
    project_root: Path,
    split: str,
    detectors: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run every available detector in ``detectors`` (default: DETECTOR_ORDER)."""
    avail = available_detectors(project_root, split)
    order = tuple(detectors) if detectors is not None else DETECTOR_ORDER
    out: dict[str, np.ndarray] = {}
    for name in order:
        if avail.get(name) is None:
            continue
        out[name] = batch_prob_sarcastic(texts, name, project_root=project_root, split=split)
    return out

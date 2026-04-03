"""
Training and inference pipeline for TF-IDF + Logistic Regression sarcasm detection.

Uses artifacts/data/cleaned.jsonl and frozen split IDs from
artifacts/splits/standard.json via data.splits helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifiers.common import (
    compute_metrics,
    format_prediction,
    load_split_frames,
    load_texts_for_inference,
    write_predictions,
)

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "classifiers" / "logreg"



def train(
    text_col: str = "text",
    output_dir: Path = None,
    max_features: int = 50000,
    c_value: float = 1.0,
    split: str = "standard",
) -> dict:
    """Train logistic regression classifier with TF-IDF features.

    Returns the metrics dict with 'dev' and 'test' keys.
    """
    if output_dir is None:
        output_dir = ARTIFACT_ROOT

    train_df, dev_df, test_df = load_split_frames(text_col, split=split)

    clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    x_train = train_df[text_col].astype(str).tolist()
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    x_dev = dev_df[text_col].astype(str).tolist()
    y_dev = dev_df["label"].to_numpy(dtype=np.int64)
    x_test = test_df[text_col].astype(str).tolist()
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    clf.fit(x_train, y_train)

    dev_prob = clf.predict_proba(x_dev)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]

    metrics = {
        "dev": compute_metrics(y_dev, dev_prob),
        "test": compute_metrics(y_test, test_prob),
        "config": {
            "model_type": "logreg",
            "text_col": text_col,
            "max_features": max_features,
            "c_value": c_value,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"

    joblib.dump(clf, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


def predict(model_dir: Path = None, texts: list[str] = None) -> list[dict[str, Any]]:
    """Run inference on texts."""
    if model_dir is None:
        model_dir = ARTIFACT_ROOT
    if texts is None:
        texts = []
    
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"LogReg model not found: {model_path}")

    clf: Pipeline = joblib.load(model_path)
    prob_pos = clf.predict_proba(texts)[:, 1]

    outputs: list[dict[str, Any]] = []
    for text, p in zip(texts, prob_pos, strict=True):
        pred_label = int(p >= 0.5)
        outputs.append(format_prediction(text, pred_label, p))
    return outputs


def main() -> None:
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python logreg_classifier.py train")
        print("       python logreg_classifier.py predict <text>")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "train":
        train()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage: python logreg_classifier.py predict <text>")
            return
        text = " ".join(sys.argv[2:])
        preds = predict(texts=[text])
        write_predictions(preds, None)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()


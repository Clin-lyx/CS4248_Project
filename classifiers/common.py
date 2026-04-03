"""
Shared utilities for sarcasm detection classifiers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.splits import get_all_splits, get_split_df, load_cleaned_data


def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and ROC-AUC."""
    preds = (prob_pos >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def load_split_frames(
    text_col: str = "text",
    split: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return train, dev, test DataFrames for a given split strategy.

    Parameters
    ----------
    text_col : str
        Column that contains the input text.
    split : str
        Split name registered in ``data.splits.SPLIT_REGISTRY``
        (e.g. ``"standard"`` or ``"topic_hard"``).
    """
    df = load_cleaned_data()
    train_df, dev_df, test_df = get_all_splits(df, split=split)

    for part_name, part in (("train", train_df), ("dev", dev_df), ("test", test_df)):
        if text_col not in part.columns:
            raise ValueError(f"Missing text column '{text_col}' in {part_name} split.")

    return train_df, dev_df, test_df


def ensure_text_column(df: pd.DataFrame, text_col: str) -> None:
    """Ensure text column exists in DataFrame."""
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Available: {list(df.columns)}")


def load_texts_for_inference(text: str | None, input_jsonl: str | None, text_col: str) -> list[str]:
    """Load texts for inference from either a single text or a JSONL file."""
    if text:
        return [text]

    if input_jsonl:
        path = Path(input_jsonl)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_json(path, lines=True, encoding="utf-8")
        ensure_text_column(df, text_col)
        return df[text_col].astype(str).tolist()

    raise ValueError("Provide either --text or --input-jsonl for prediction.")


def format_prediction(text: str, pred_label: int, prob_sarcastic: float) -> dict[str, Any]:
    """Format a single prediction output."""
    return {
        "text": text,
        "pred_label": pred_label,
        "prob_sarcastic": float(prob_sarcastic),
    }


def write_predictions(preds: list[dict[str, Any]], output_path: Path | None) -> None:
    """Write predictions to JSON lines or print to stdout."""
    if output_path is None:
        import json
        print(json.dumps(preds, indent=2))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved predictions -> {output_path}")


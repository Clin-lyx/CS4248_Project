from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.splits import SPLIT_REGISTRY, get_split_df, load_cleaned_data
from systems.system_a.template_utils import preserves_anchors

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "system_b"
RAW_PAIRS_PATH = ARTIFACT_ROOT / "pseudo_pairs_raw.jsonl"
FILTERED_PAIRS_PATH = ARTIFACT_ROOT / "pseudo_pairs_filtered.jsonl"
MODEL_DIR = ARTIFACT_ROOT / "model"
LOGREG_MODEL_PATH = PROJECT_ROOT / "artifacts" / "classifiers" / "logreg" / "model.joblib"


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def append_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_anchored_data(
    input_path: str | Path = "artifacts/data/cleaned_with_anchors.jsonl",
) -> pd.DataFrame:
    resolved = PROJECT_ROOT / Path(input_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Anchored dataset not found: {resolved}")
    return pd.read_json(resolved, lines=True, encoding="utf-8")


def load_anchored_split_frames(split: str = "standard") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split not in SPLIT_REGISTRY:
        raise KeyError(f"Unknown split '{split}'. Available: {sorted(SPLIT_REGISTRY)}")

    df = load_anchored_data()
    split_path = SPLIT_REGISTRY[split]
    return (
        get_split_df(df, "train", split_path=split_path),
        get_split_df(df, "dev", split_path=split_path),
        get_split_df(df, "test", split_path=split_path),
    )


def infer_direction(label: int) -> str:
    if int(label) == 0:
        return "n2s"
    if int(label) == 1:
        return "s2n"
    raise ValueError(f"Unknown label: {label}")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(text: str) -> str:
    return normalize_space(text).lower()


def looks_too_similar(candidate: str, original: str, max_overlap: float = 0.9) -> bool:
    cand = normalize_text(candidate)
    orig = normalize_text(original)

    if not cand or not orig:
        return False
    if cand == orig:
        return True

    orig_words = set(orig.split())
    cand_words = set(cand.split())
    if not orig_words:
        return False

    overlap = len(orig_words & cand_words) / len(orig_words)
    return overlap > max_overlap


def tokenize_for_edit_distance(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+|[^\w\s]", normalize_space(text).lower())


def normalized_token_edit_distance(text_a: str, text_b: str) -> float:
    toks_a = tokenize_for_edit_distance(text_a)
    toks_b = tokenize_for_edit_distance(text_b)

    if not toks_a and not toks_b:
        return 0.0
    if not toks_a or not toks_b:
        return 1.0

    rows = len(toks_a) + 1
    cols = len(toks_b) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if toks_a[i - 1] == toks_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / max(len(toks_a), len(toks_b))


def semantic_similarity_score(text_a: str, text_b: str) -> float:
    from similarity.semantic_similarity import semantic_similarity

    return float(semantic_similarity(text_a, text_b))


def load_style_scorer(model_path: str | Path = LOGREG_MODEL_PATH):
    resolved = Path(model_path)
    if not resolved.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(resolved)
    except Exception:
        return None


def score_style_probability(text: str, scorer=None) -> float | None:
    if scorer is None:
        scorer = load_style_scorer()
    if scorer is None:
        return None
    try:
        return float(scorer.predict_proba([text])[0][1])
    except Exception:
        return None


def target_style_passes(direction: str, prob_sarcastic: float | None, threshold: float) -> bool:
    if prob_sarcastic is None:
        return True
    if direction == "n2s":
        return prob_sarcastic >= threshold
    if direction == "s2n":
        return prob_sarcastic <= (1.0 - threshold)
    raise ValueError(f"Unknown direction: {direction}")


def build_seq2seq_input(source_text: str, direction: str) -> str:
    task = "rewrite_to_sarcastic" if direction == "n2s" else "rewrite_to_non_sarcastic"
    return f"{task}: {normalize_space(source_text)}"


def select_subset(
    df: pd.DataFrame,
    split_name: str,
    split: str = "standard",
) -> pd.DataFrame:
    if split not in SPLIT_REGISTRY:
        raise KeyError(f"Unknown split '{split}'. Available: {sorted(SPLIT_REGISTRY)}")
    split_path = SPLIT_REGISTRY[split]
    return get_split_df(df, split_name, split_path=split_path)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def count_true(values: list[bool]) -> int:
    return int(sum(1 for v in values if v))


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)

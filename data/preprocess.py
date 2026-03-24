"""
Preprocessing pipeline for the sarcasm headline dataset.

Light normalization only — preserves punctuation, casing, and stylistic cues
that are essential for sarcasm detection and generation downstream.
"""

import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_headline(headline: str) -> str:
    """Light normalization that keeps sarcasm-relevant surface features intact."""
    if not isinstance(headline, str):
        return ""

    headline = unicodedata.normalize("NFKC", headline)

    headline = headline.replace("\u2018", "'").replace("\u2019", "'")
    headline = headline.replace("\u201c", '"').replace("\u201d", '"')
    headline = headline.replace("\u2013", "-").replace("\u2014", "-")
    headline = headline.replace("\u2026", "...")

    headline = re.sub(r"\s+", " ", headline)
    headline = headline.strip()

    return headline


def extract_publisher(url: str) -> str:
    """Derive publisher tag from article_link hostname."""
    if not isinstance(url, str) or not url.strip():
        return "unknown"
    host = (urlparse(url).hostname or "").lower()
    if "onion" in host:
        return "theonion"
    if "huffingtonpost" in host or "huffpost" in host:
        return "huffpost"
    return "other"


def preprocess(
    dataset_path: str | Path = "dataset/Sarcasm_Headlines_Dataset_v2.json",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> normalize -> dedupe -> assign IDs.

    Returns a DataFrame with columns:
        id, headline_raw, text, label, article_link, publisher
    """
    from data.load_dataset import DatasetLoader

    df = DatasetLoader(dataset_path).get_dataframe().copy()

    required = {"headline", "is_sarcastic", "article_link"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in source data: {missing}")

    df["headline_raw"] = df["headline"]
    df["text"] = df["headline"].apply(normalize_headline)
    df["label"] = df["is_sarcastic"]
    df["publisher"] = df["article_link"].apply(extract_publisher)

    n_before = len(df)
    df = df[df["text"] != ""].copy()
    n_empty = n_before - len(df)

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    n_dupes = n_before_dedup - len(df)

    df = df.reset_index(drop=True)
    df["id"] = [f"sar_{i:06d}" for i in range(1, len(df) + 1)]

    df = df[["id", "headline_raw", "text", "label", "article_link", "publisher"]]

    print(f"Preprocess summary:")
    print(f"  Raw rows:           {n_before}")
    print(f"  Empty headlines:    {n_empty}")
    print(f"  Duplicate headlines:{n_dupes}")
    print(f"  Final rows:         {len(df)}")

    return df


def save_jsonl(df: pd.DataFrame, output_path: str | Path = "artifacts/data/cleaned.jsonl") -> Path:
    """Write DataFrame to JSONL."""
    output_path = _PROJECT_ROOT / Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} rows -> {output_path}")
    return output_path


if __name__ == "__main__":
    cleaned = preprocess()
    save_jsonl(cleaned)

"""
Train / dev / test splitting for the sarcasm headline dataset.

Produces a JSON file mapping split names to lists of row IDs.
Teammates load it with ``load_split`` or ``get_split_df``.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPLIT_PATH = "artifacts/splits/standard.json"
_DEFAULT_CLEANED_PATH = "artifacts/data/cleaned.jsonl"

SPLIT_REGISTRY: dict[str, str] = {
    "standard": "artifacts/splits/standard.json",
    "topic_hard": "artifacts/splits/topic_hard.json",
}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_cleaned_data(input_path: str | Path = _DEFAULT_CLEANED_PATH) -> pd.DataFrame:
    input_path = _PROJECT_ROOT / Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {input_path}")

    df = pd.read_json(input_path, lines=True, encoding="utf-8")

    required = {"id", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleaned dataset: {missing}")

    return df


def load_split(
    split_path: str | Path = _DEFAULT_SPLIT_PATH,
) -> dict:
    """Load a saved split JSON and return the full dict (meta + id lists)."""
    split_path = _PROJECT_ROOT / Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run splits.py first."
        )
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_df(
    df: pd.DataFrame,
    split_name: str,
    split_path: str | Path = _DEFAULT_SPLIT_PATH,
) -> pd.DataFrame:
    """Return the subset of *df* belonging to a split ('train', 'dev', or 'test')."""
    split = load_split(split_path)
    if split_name not in split:
        raise KeyError(f"Unknown split '{split_name}'. Available: {[k for k in split if k != 'meta']}")
    ids = set(split[split_name])
    return df[df["id"].isin(ids)].reset_index(drop=True)


def get_all_splits(
    df: pd.DataFrame,
    split: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, dev_df, test_df) for a named split strategy.

    Available splits: "standard", "topic_hard".

    Usage::

        from data.splits import load_cleaned_data, get_all_splits

        df = load_cleaned_data()
        train_df, dev_df, test_df = get_all_splits(df, split="topic_hard")
    """
    if split not in SPLIT_REGISTRY:
        raise KeyError(
            f"Unknown split '{split}'. Available: {sorted(SPLIT_REGISTRY)}"
        )
    path = SPLIT_REGISTRY[split]
    return (
        get_split_df(df, "train", split_path=path),
        get_split_df(df, "dev", split_path=path),
        get_split_df(df, "test", split_path=path),
    )


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def make_standard_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> dict:
    total = train_ratio + dev_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=random_state,
    )

    dev_fraction = dev_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - dev_fraction),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    _validate_no_overlap(train_df, dev_df, test_df)

    def _label_counts(part: pd.DataFrame) -> dict:
        vc = part["label"].value_counts().sort_index()
        return {int(k): int(v) for k, v in vc.items()}

    split = {
        "meta": {
            "ratios": {"train": train_ratio, "dev": dev_ratio, "test": test_ratio},
            "random_state": random_state,
            "total_rows": len(df),
            "counts": {
                "train": {"n": len(train_df), "labels": _label_counts(train_df)},
                "dev": {"n": len(dev_df), "labels": _label_counts(dev_df)},
                "test": {"n": len(test_df), "labels": _label_counts(test_df)},
            },
        },
        "train": sorted(train_df["id"].tolist()),
        "dev": sorted(dev_df["id"].tolist()),
        "test": sorted(test_df["id"].tolist()),
    }

    return split


def _validate_no_overlap(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_ids = set(train_df["id"])
    dev_ids = set(dev_df["id"])
    test_ids = set(test_df["id"])

    assert train_ids.isdisjoint(dev_ids), "train/dev overlap!"
    assert train_ids.isdisjoint(test_ids), "train/test overlap!"
    assert dev_ids.isdisjoint(test_ids), "dev/test overlap!"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_split(df: pd.DataFrame, split: dict) -> None:
    print("=== Standard split summary ===")
    has_publisher = "publisher" in df.columns

    for name in ["train", "dev", "test"]:
        ids = set(split[name])
        part = df[df["id"].isin(ids)]
        lc = part["label"].value_counts().sort_index()
        lp = part["label"].value_counts(normalize=True).sort_index() * 100

        print(f"\n{name.upper()} ({len(part)} rows)")
        for lab in sorted(lc.index):
            tag = "sarcastic" if lab == 1 else "non-sarcastic"
            print(f"  {tag} ({lab}): {lc[lab]} ({lp[lab]:.2f}%)")

        if has_publisher:
            pc = part["publisher"].value_counts()
            pp = part["publisher"].value_counts(normalize=True) * 100
            for pub in sorted(pc.index):
                print(f"  [{pub}]: {pc[pub]} ({pp[pub]:.2f}%)")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(
    split: dict,
    output_path: str | Path = _DEFAULT_SPLIT_PATH,
) -> Path:
    output_path = _PROJECT_ROOT / Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"\nSaved split -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_cleaned_data()
    split = make_standard_split(df)
    summarize_split(df, split)
    save_split(split)

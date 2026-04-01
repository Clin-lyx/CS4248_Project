"""
Topic-hard splitting using TF-IDF -> TruncatedSVD -> KMeans.

Creates a split JSON where each cluster is assigned as a whole to one split.
The assignment uses a greedy objective to stay close to:
  - target split ratios (default 80/10/10)
  - global label ratio
  - reasonable split balance
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT_PATH = "artifacts/data/cleaned.jsonl"
_DEFAULT_OUTPUT_PATH = "artifacts/splits/topic_hard.json"
_DEFAULT_K_VALUES = [20, 30, 40, 50, 60, 80]
_RANDOM_STATE = 42


def load_cleaned_data(input_path: str | Path = _DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load cleaned dataset and validate expected columns."""
    full_path = _PROJECT_ROOT / Path(input_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {full_path}")

    df = pd.read_json(full_path, lines=True, encoding="utf-8")
    required = {"id", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["id", "label", "text"]].dropna(subset=["text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def build_topic_vectors(
    texts: pd.Series,
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    n_components: int = 200,
    random_state: int = _RANDOM_STATE,
):
    """
    Build dense topic vectors via TF-IDF -> TruncatedSVD -> L2 normalize.
    Returns:
        X_topic: np.ndarray [n_samples, n_components]
        vectorizer: fitted TfidfVectorizer
        reducer_pipeline: fitted pipeline (SVD + Normalizer)
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # Keep components safely below feature dimension.
    max_valid_components = max(2, min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1))
    use_components = min(n_components, max_valid_components)

    svd = TruncatedSVD(n_components=use_components, random_state=random_state)
    normalizer = Normalizer(copy=False)
    reducer_pipeline = make_pipeline(svd, normalizer)
    X_topic = reducer_pipeline.fit_transform(X_tfidf)
    return X_topic, vectorizer, reducer_pipeline


def _normalized_entropy(cluster_counts: np.ndarray) -> float:
    probs = cluster_counts / cluster_counts.sum()
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    max_entropy = math.log(len(cluster_counts))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


def evaluate_k_values(
    X_topic: np.ndarray,
    labels_true: pd.Series,
    k_values: list[int] | tuple[int, ...] = _DEFAULT_K_VALUES,
    random_state: int = _RANDOM_STATE,
    silhouette_sample_size: int = 5_000,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Evaluate candidate k values and return metrics plus fitted labels for each k.
    """
    rows: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}
    global_pos_ratio = float(labels_true.mean())

    n_samples = X_topic.shape[0]
    sample_size = min(silhouette_sample_size, n_samples)

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        cluster_labels = model.fit_predict(X_topic)
        labels_by_k[k] = cluster_labels

        sil = silhouette_score(
            X_topic,
            cluster_labels,
            sample_size=sample_size,
            random_state=random_state,
        )
        counts = pd.Series(cluster_labels).value_counts().sort_index()
        count_values = counts.to_numpy()

        # Label skew proxy: mean absolute deviation from global positive ratio.
        df_tmp = pd.DataFrame({"cluster": cluster_labels, "label": labels_true})
        per_cluster_pos = df_tmp.groupby("cluster")["label"].mean()
        mean_abs_label_skew = float((per_cluster_pos - global_pos_ratio).abs().mean())

        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(sil),
                "max_cluster_pct": float(count_values.max() / n_samples),
                "min_cluster_pct": float(count_values.min() / n_samples),
                "entropy_norm": float(_normalized_entropy(count_values)),
                "mean_abs_label_skew": mean_abs_label_skew,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return metrics, labels_by_k


def choose_usable_k(metrics: pd.DataFrame) -> int:
    """
    Choose a usable k with a simple composite score:
      higher silhouette + higher entropy - larger max cluster - stronger label skew
    """
    m = metrics.copy()
    m["usable_score"] = (
        1.0 * m["silhouette"]
        + 0.20 * m["entropy_norm"]
        - 0.25 * m["max_cluster_pct"]
        - 0.15 * m["mean_abs_label_skew"]
    )
    return int(m.sort_values("usable_score", ascending=False).iloc[0]["k"])


def summarize_clusters(df: pd.DataFrame, cluster_col: str = "cluster") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cluster size and label-mix summaries."""
    counts = df[cluster_col].value_counts().sort_index()
    size_table = pd.DataFrame(
        {
            "n": counts,
            "pct": (counts / len(df) * 100).round(2),
        }
    )

    label_mix = pd.crosstab(df[cluster_col], df["label"], normalize="index").round(3)
    label_mix.columns = [f"label_{int(c)}" for c in label_mix.columns]
    return size_table, label_mix


@dataclass
class _SplitState:
    n: int = 0
    pos: int = 0


def assign_clusters_to_splits(
    df: pd.DataFrame,
    split_ratios: dict[str, float] | None = None,
    random_state: int = _RANDOM_STATE,
) -> dict:
    """
    Greedy assignment of whole clusters to splits.
    """
    if split_ratios is None:
        split_ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}

    ratio_total = sum(split_ratios.values())
    if abs(ratio_total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_total}")

    total_n = len(df)
    global_pos_ratio = float(df["label"].mean())
    target_n = {name: int(round(total_n * r)) for name, r in split_ratios.items()}

    cluster_stats = (
        df.groupby("cluster")
        .agg(n=("id", "size"), pos=("label", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(random_state)
    # Small jitter keeps tie-breaking reproducible but non-deterministic by index order.
    cluster_stats["tie_jitter"] = rng.random(len(cluster_stats))
    cluster_stats = cluster_stats.sort_values(["n", "tie_jitter"], ascending=[False, True])

    states = {name: _SplitState() for name in split_ratios}
    cluster_to_split: dict[int, str] = {}

    w_size = 1.0
    w_label = 0.8
    w_overflow = 1.3

    split_names = list(split_ratios.keys())

    def score_candidate(split_name: str, c_n: int, c_pos: int) -> float:
        current = states[split_name]
        new_n = current.n + c_n
        new_pos = current.pos + c_pos

        size_gap = abs(new_n - target_n[split_name]) / total_n

        new_pos_ratio = (new_pos / new_n) if new_n > 0 else global_pos_ratio
        label_gap = abs(new_pos_ratio - global_pos_ratio)

        overflow = max(0, new_n - target_n[split_name]) / total_n
        return (w_size * size_gap) + (w_label * label_gap) + (w_overflow * overflow)

    for row in cluster_stats.itertuples(index=False):
        c_id = int(row.cluster)
        c_n = int(row.n)
        c_pos = int(row.pos)

        best_split = min(split_names, key=lambda s: score_candidate(s, c_n, c_pos))
        cluster_to_split[c_id] = best_split
        states[best_split].n += c_n
        states[best_split].pos += c_pos

    df_assigned = df.copy()
    df_assigned["split"] = df_assigned["cluster"].map(cluster_to_split)

    split_obj = {
        "meta": {
            "method": "topic_hard_tfidf_svd_kmeans",
            "ratios": split_ratios,
            "random_state": random_state,
            "total_rows": total_n,
            "global_label_ratio": {
                "label_0": float((df["label"] == 0).mean()),
                "label_1": float((df["label"] == 1).mean()),
            },
            "cluster_to_split": {str(k): v for k, v in sorted(cluster_to_split.items())},
        },
        "train": sorted(df_assigned.loc[df_assigned["split"] == "train", "id"].tolist()),
        "dev": sorted(df_assigned.loc[df_assigned["split"] == "dev", "id"].tolist()),
        "test": sorted(df_assigned.loc[df_assigned["split"] == "test", "id"].tolist()),
    }
    return split_obj


def save_split(split_obj: dict, output_path: str | Path = _DEFAULT_OUTPUT_PATH) -> Path:
    full_path = _PROJECT_ROOT / Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2)
    return full_path


def print_final_summary(df: pd.DataFrame, split_obj: dict) -> None:
    print("\n=== Final topic-hard split summary ===")

    cluster_to_split = {
        int(cluster_id): split_name
        for cluster_id, split_name in split_obj["meta"]["cluster_to_split"].items()
    }
    df_tmp = df.copy()
    df_tmp["split"] = df_tmp["cluster"].map(cluster_to_split)

    for split_name in ["train", "dev", "test"]:
        part = df_tmp[df_tmp["split"] == split_name]
        n = len(part)
        label_counts = part["label"].value_counts().sort_index()
        label_props = part["label"].value_counts(normalize=True).sort_index()
        clusters = sorted(part["cluster"].unique().tolist())

        print(f"\n{split_name.upper()} ({n} rows)")
        for lab in [0, 1]:
            c = int(label_counts.get(lab, 0))
            p = float(label_props.get(lab, 0.0))
            print(f"  label_{lab}: {c} ({p:.3f})")
        print(f"  clusters: {clusters}")

    split_to_clusters: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    for c, s in cluster_to_split.items():
        split_to_clusters[s].append(c)
    for s in split_to_clusters:
        split_to_clusters[s] = sorted(split_to_clusters[s])

    print("\nCluster assignment per split:")
    for split_name in ["train", "dev", "test"]:
        print(f"  {split_name}: {split_to_clusters[split_name]}")


def main() -> None:
    print("Loading data...")
    df = load_cleaned_data()
    print(f"Loaded {len(df):,} rows")

    print("\nBuilding topic vectors (TF-IDF -> SVD -> normalize)...")
    X_topic, _, _ = build_topic_vectors(df["text"])
    print(f"Topic vector shape: {X_topic.shape}")

    print("\nEvaluating candidate k values...")
    metrics, labels_by_k = evaluate_k_values(
        X_topic=X_topic,
        labels_true=df["label"],
        k_values=_DEFAULT_K_VALUES,
        random_state=_RANDOM_STATE,
    )
    print(metrics.to_string(index=False))

    chosen_k = choose_usable_k(metrics)
    print(f"\nChosen k: {chosen_k}")

    df = df.copy()
    df["cluster"] = labels_by_k[chosen_k]

    size_table, label_mix = summarize_clusters(df, cluster_col="cluster")
    print("\nCluster sizes:")
    print(size_table.to_string())
    print("\nLabel mix per cluster:")
    print(label_mix.to_string())

    split_obj = assign_clusters_to_splits(
        df=df,
        split_ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
        random_state=_RANDOM_STATE,
    )

    output_path = save_split(split_obj, _DEFAULT_OUTPUT_PATH)
    print(f"\nSaved split -> {output_path}")

    print_final_summary(df, split_obj)


if __name__ == "__main__":
    main()

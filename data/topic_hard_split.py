"""
Topic-hard splitting using TF-IDF -> TruncatedSVD -> KMeans.

Creates a split JSON where each cluster is assigned as a whole to one split.
The greedy assignment uses proportional fill so the least-filled split
(relative to its target) receives the next cluster, with overflow and
label-balance corrections.

Output: artifacts/splits/topic_hard.json
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
_DEFAULT_BALANCE_BAND = (0.4, 0.6)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Topic vectors
# ---------------------------------------------------------------------------

def build_topic_vectors(
    texts: pd.Series,
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    n_components: int = 200,
    random_state: int = _RANDOM_STATE,
):
    """
    Build dense topic vectors: TF-IDF -> TruncatedSVD -> L2 normalize.

    Returns (X_tfidf, X_topic, vectorizer, reducer_pipeline).
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    X_tfidf = vectorizer.fit_transform(texts)

    max_valid = max(2, min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1))
    use_components = min(n_components, max_valid)

    svd = TruncatedSVD(n_components=use_components, random_state=random_state)
    normalizer = Normalizer(copy=False)
    reducer = make_pipeline(svd, normalizer)
    X_topic = reducer.fit_transform(X_tfidf)
    return X_tfidf, X_topic, vectorizer, reducer


# ---------------------------------------------------------------------------
# K-value evaluation
# ---------------------------------------------------------------------------

def _normalized_entropy(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    max_entropy = math.log(len(counts))
    return 0.0 if max_entropy == 0 else entropy / max_entropy


def evaluate_k_values(
    X_topic: np.ndarray,
    labels_true: pd.Series,
    k_values: list[int] | tuple[int, ...] = _DEFAULT_K_VALUES,
    random_state: int = _RANDOM_STATE,
    silhouette_sample_size: int = 5_000,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Evaluate candidate k values; return metrics table + cluster labels per k."""
    rows: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}
    global_pos_ratio = float(labels_true.mean())
    sample_size = min(silhouette_sample_size, X_topic.shape[0])

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        cluster_labels = model.fit_predict(X_topic)
        labels_by_k[k] = cluster_labels

        sil = silhouette_score(
            X_topic, cluster_labels,
            sample_size=sample_size, random_state=random_state,
        )
        count_values = pd.Series(cluster_labels).value_counts().sort_index().to_numpy()

        tmp = pd.DataFrame({"cluster": cluster_labels, "label": labels_true})
        per_cluster_pos = tmp.groupby("cluster")["label"].mean()
        mean_abs_label_skew = float((per_cluster_pos - global_pos_ratio).abs().mean())

        rows.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": float(sil),
            "max_cluster_pct": float(count_values.max() / X_topic.shape[0]),
            "min_cluster_pct": float(count_values.min() / X_topic.shape[0]),
            "entropy_norm": float(_normalized_entropy(count_values)),
            "mean_abs_label_skew": mean_abs_label_skew,
        })

    metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return metrics, labels_by_k


def choose_usable_k(metrics: pd.DataFrame) -> int:
    """Pick k via composite: higher silhouette/entropy, lower concentration/skew."""
    m = add_usable_score(metrics)
    return int(m.sort_values("usable_score", ascending=False).iloc[0]["k"])


def add_usable_score(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach the composite usability score used to choose k."""
    m = metrics.copy()
    m["usable_score"] = (
        1.0 * m["silhouette"]
        + 0.20 * m["entropy_norm"]
        - 0.25 * m["max_cluster_pct"]
        - 0.15 * m["mean_abs_label_skew"]
    )
    return m


# ---------------------------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------------------------

def summarize_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    balance_band: tuple[float, float] = _DEFAULT_BALANCE_BAND,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a compact cluster summary table plus per-cluster profile rows.

    The summary table is intended for reports/slides; the cluster profile keeps
    richer per-cluster statistics for downstream inspection.
    """
    lower, upper = balance_band
    cluster_profile = (
        df.groupby(cluster_col)
        .agg(
            cluster_size=("id", "size"),
            sarcasm_ratio=("label", "mean"),
        )
        .reset_index()
        .sort_values(cluster_col)
        .reset_index(drop=True)
    )
    cluster_profile["non_sarcasm_ratio"] = 1.0 - cluster_profile["sarcasm_ratio"]
    cluster_profile["balance_gap"] = (cluster_profile["sarcasm_ratio"] - 0.5).abs()
    cluster_profile["majority_label"] = np.where(
        cluster_profile["sarcasm_ratio"] > 0.5,
        "sarcasm",
        np.where(cluster_profile["sarcasm_ratio"] < 0.5, "non-sarcasm", "tied"),
    )
    cluster_profile["balance_bucket"] = np.where(
        cluster_profile["sarcasm_ratio"] > upper,
        "sarcasm-heavy",
        np.where(
            cluster_profile["sarcasm_ratio"] < lower,
            "non-sarcasm-heavy",
            "near-balanced",
        ),
    )
    cluster_profile["is_near_balanced"] = cluster_profile["sarcasm_ratio"].between(
        lower,
        upper,
        inclusive="both",
    )

    size_values = cluster_profile["cluster_size"]
    summary_table = pd.DataFrame(
        {
            "Item": [
                "Number of clusters chosen",
                "Mean cluster size",
                "Median cluster size",
                "Smallest cluster",
                "Largest cluster",
                f"# sarcasm-heavy clusters (>{int(upper * 100)}% sarcasm)",
                f"# near-balanced clusters ({int(lower * 100)}-{int(upper * 100)}% sarcasm)",
                f"# non-sarcasm-heavy clusters (<{int(lower * 100)}% sarcasm)",
            ],
            "Value": [
                int(cluster_profile[cluster_col].nunique()),
                float(size_values.mean()),
                float(size_values.median()),
                int(size_values.min()),
                int(size_values.max()),
                int((cluster_profile["balance_bucket"] == "sarcasm-heavy").sum()),
                int((cluster_profile["balance_bucket"] == "near-balanced").sum()),
                int((cluster_profile["balance_bucket"] == "non-sarcasm-heavy").sum()),
            ],
        }
    )
    return summary_table, cluster_profile


def _format_summary_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    """Format summary values for compact CLI printing."""
    formatted = summary_table.copy()

    def _format_value(value: float) -> str:
        value = float(value)
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.1f}"

    formatted["Value"] = formatted["Value"].map(_format_value)
    return formatted


def top_terms_by_cluster(
    X_tfidf, cluster_labels: np.ndarray, vectorizer, top_n: int = 12,
) -> pd.DataFrame:
    """Top TF-IDF terms per cluster (by mean TF-IDF within cluster)."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []
    for c in sorted(np.unique(cluster_labels)):
        mean_vec = np.asarray(X_tfidf[cluster_labels == c].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_vec)[::-1][:top_n]
        rows.append({"cluster": int(c), "top_terms": ", ".join(feature_names[top_idx])})
    return pd.DataFrame(rows)


def select_cluster_examples(
    cluster_profile: pd.DataFrame,
    top_terms_df: pd.DataFrame,
    cluster_col: str = "cluster",
    terms_to_show: int = 6,
) -> pd.DataFrame:
    """
    Pick a small, interpretable set of representative clusters.

    This mirrors the notebook presentation: most sarcastic, most non-sarcastic,
    most balanced, smallest, and largest cluster.
    """
    candidates = cluster_profile.merge(top_terms_df, on=cluster_col, how="left")

    def _shorten_terms(term_string: str) -> str:
        terms = [term.strip() for term in str(term_string).split(",") if term.strip()]
        return ", ".join(terms[:terms_to_show])

    selectors = [
        (
            "Most sarcastic cluster",
            lambda frame: frame.sort_values(["sarcasm_ratio", "cluster_size"], ascending=[False, False]),
        ),
        (
            "Most non-sarcastic cluster",
            lambda frame: frame.sort_values(["sarcasm_ratio", "cluster_size"], ascending=[True, False]),
        ),
        (
            "Most balanced cluster",
            lambda frame: frame.sort_values(["balance_gap", "cluster_size"], ascending=[True, False]),
        ),
        (
            "Smallest cluster",
            lambda frame: frame.sort_values(["cluster_size", "balance_gap"], ascending=[True, True]),
        ),
        (
            "Largest cluster",
            lambda frame: frame.sort_values(["cluster_size", "balance_gap"], ascending=[False, True]),
        ),
    ]

    rows = []
    used_clusters: set[int] = set()

    for example_name, ranker in selectors:
        ranked = ranker(candidates.copy())
        chosen_row = None

        for _, candidate in ranked.iterrows():
            cluster_id = int(candidate[cluster_col])
            if cluster_id not in used_clusters:
                used_clusters.add(cluster_id)
                chosen_row = candidate
                break

        if chosen_row is None:
            chosen_row = ranked.iloc[0]

        rows.append(
            {
                "Example": example_name,
                "Cluster": int(chosen_row[cluster_col]),
                "Size": int(chosen_row["cluster_size"]),
                "Sarcasm %": f"{chosen_row['sarcasm_ratio'] * 100:.1f}%",
                "Majority label": chosen_row["majority_label"],
                "Top terms": _shorten_terms(chosen_row["top_terms"]),
            }
        )

    return pd.DataFrame(rows)


def plot_cluster_diagnostics(
    cluster_profile: pd.DataFrame,
    balance_band: tuple[float, float] = _DEFAULT_BALANCE_BAND,
) -> None:
    """Plot cluster size against sarcasm proportion for slide/report use."""
    import matplotlib.pyplot as plt

    lower, upper = balance_band
    colors = np.where(
        cluster_profile["sarcasm_ratio"] > upper,
        "#c0392b",
        np.where(cluster_profile["sarcasm_ratio"] < lower, "#2980b9", "#7f8c8d"),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhspan(
        lower,
        upper,
        color="#ecf0f1",
        alpha=0.9,
        label=f"Near-balanced band ({int(lower * 100)}-{int(upper * 100)}%)",
    )
    ax.scatter(
        cluster_profile["cluster_size"],
        cluster_profile["sarcasm_ratio"],
        c=colors,
        s=70,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.6,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="50% sarcasm")
    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (log scale)")
    ax.set_ylabel("Sarcasm proportion")
    ax.set_title("Topic clusters vary in both size and sarcasm ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Greedy cluster-to-split assignment
# ---------------------------------------------------------------------------

@dataclass
class _SplitState:
    n: int = 0
    pos: int = 0


def assign_clusters_to_splits(
    df: pd.DataFrame,
    split_ratios: dict[str, float] | None = None,
    random_state: int = _RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Greedy whole-cluster assignment using proportional fill.

    The least-filled split (relative to its own target) receives the next
    cluster.  Heavy overflow penalty prevents exceeding any target, and a
    secondary label-gap term nudges toward balanced label ratios.

    Returns (df_with_split_column, cluster_to_split_dict).
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
    cluster_stats["tie_jitter"] = rng.random(len(cluster_stats))
    cluster_stats = cluster_stats.sort_values(
        ["n", "tie_jitter"], ascending=[False, True],
    )

    states = {name: _SplitState() for name in split_ratios}
    split_names = list(split_ratios.keys())
    cluster_to_split: dict[int, str] = {}

    w_fill, w_overflow, w_label = 1.0, 2.0, 0.4

    def _score(split_name: str, c_n: int, c_pos: int) -> float:
        st = states[split_name]
        new_n = st.n + c_n
        new_pos = st.pos + c_pos

        current_fill = (
            st.n / target_n[split_name]
            if target_n[split_name] > 0 else float("inf")
        )
        overflow = max(0.0, (new_n - target_n[split_name]) / target_n[split_name])

        new_pos_ratio = (new_pos / new_n) if new_n > 0 else global_pos_ratio
        label_gap = abs(new_pos_ratio - global_pos_ratio)

        return w_fill * current_fill + w_overflow * overflow + w_label * label_gap

    for row in cluster_stats.itertuples(index=False):
        c_id = int(row.cluster)
        c_n = int(row.n)
        c_pos = int(row.pos)

        best = min(split_names, key=lambda s: _score(s, c_n, c_pos))
        cluster_to_split[c_id] = best
        states[best].n += c_n
        states[best].pos += c_pos

    df_assigned = df.copy()
    df_assigned["split"] = df_assigned["cluster"].map(cluster_to_split)
    return df_assigned, cluster_to_split


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def build_split_json(
    df_assigned: pd.DataFrame,
    cluster_to_split: dict[int, str],
    metrics: pd.DataFrame,
    chosen_k: int,
    split_ratios: dict[str, float],
    random_state: int,
) -> dict:
    """Assemble the JSON-serialisable split object."""
    return {
        "meta": {
            "method": "topic_hard_tfidf_svd_kmeans",
            "ratios": split_ratios,
            "random_state": random_state,
            "total_rows": int(len(df_assigned)),
            "global_label_ratio": {
                "label_0": float((df_assigned["label"] == 0).mean()),
                "label_1": float((df_assigned["label"] == 1).mean()),
            },
            "selected_k": int(chosen_k),
            "k_metrics": metrics.to_dict(orient="records"),
            "cluster_to_split": {
                str(k): v for k, v in sorted(cluster_to_split.items())
            },
        },
        "train": sorted(
            df_assigned.loc[df_assigned["split"] == "train", "id"].tolist()
        ),
        "dev": sorted(
            df_assigned.loc[df_assigned["split"] == "dev", "id"].tolist()
        ),
        "test": sorted(
            df_assigned.loc[df_assigned["split"] == "test", "id"].tolist()
        ),
    }


def save_split(split_obj: dict, output_path: str | Path = _DEFAULT_OUTPUT_PATH) -> Path:
    full_path = _PROJECT_ROOT / Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2)
    return full_path


# ---------------------------------------------------------------------------
# Summary / validation
# ---------------------------------------------------------------------------

def validate_split(df: pd.DataFrame, split_obj: dict) -> None:
    """Assert no overlap and full coverage."""
    train_ids = set(split_obj["train"])
    dev_ids = set(split_obj["dev"])
    test_ids = set(split_obj["test"])

    assert train_ids.isdisjoint(dev_ids), "train/dev overlap!"
    assert train_ids.isdisjoint(test_ids), "train/test overlap!"
    assert dev_ids.isdisjoint(test_ids), "dev/test overlap!"

    total = len(train_ids) + len(dev_ids) + len(test_ids)
    assert total == len(df), f"Assigned {total} != {len(df)} total rows"
    print(f"Validation passed: {total} IDs, no overlaps.")


def print_final_summary(
    df_assigned: pd.DataFrame,
    cluster_to_split: dict[int, str],
    target_ratios: dict[str, float],
    show_cluster_membership: bool = False,
) -> None:
    total = len(df_assigned)
    global_pos = float(df_assigned["label"].mean())
    print("\n=== Final topic-hard split summary ===")
    print(f"Global label_1 ratio: {global_pos:.3f}")

    for split_name in ["train", "dev", "test"]:
        part = df_assigned[df_assigned["split"] == split_name]
        n = len(part)
        actual_ratio = n / total
        target_ratio = target_ratios[split_name]
        label_dist = part["label"].value_counts(normalize=True).sort_index()

        print(f"\n{split_name.upper()} ({n} rows, {actual_ratio:.3f} vs target {target_ratio:.2f})")
        for lab in [0, 1]:
            print(f"  label_{lab}: {float(label_dist.get(lab, 0.0)):.3f}")
        print(f"  n_clusters: {part['cluster'].nunique()}")

        if show_cluster_membership:
            clusters = sorted(part["cluster"].unique().tolist())
            print(f"  clusters: {clusters}")

    if show_cluster_membership:
        split_to_clusters: dict[str, list[int]] = {s: [] for s in target_ratios}
        for c, s in cluster_to_split.items():
            split_to_clusters[s].append(c)
        print("\nCluster assignment per split:")
        for s in ["train", "dev", "test"]:
            print(f"  {s}: {sorted(split_to_clusters[s])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    split_ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}

    print("Loading data...")
    df = load_cleaned_data()
    print(f"Loaded {len(df):,} rows")

    print("\nBuilding topic vectors (TF-IDF -> SVD -> normalize)...")
    X_tfidf, X_topic, vectorizer, _ = build_topic_vectors(df["text"])
    print(f"TF-IDF shape: {X_tfidf.shape}")
    print(f"Topic vector shape: {X_topic.shape}")

    print("\nEvaluating candidate k values...")
    metrics, labels_by_k = evaluate_k_values(
        X_topic=X_topic,
        labels_true=df["label"],
        k_values=_DEFAULT_K_VALUES,
        random_state=_RANDOM_STATE,
    )
    metrics = add_usable_score(metrics)
    print(metrics.to_string(index=False))

    chosen_k = choose_usable_k(metrics)
    print(f"\nChosen k: {chosen_k}")

    df = df.copy()
    df["cluster"] = labels_by_k[chosen_k]

    summary_table, cluster_profile = summarize_clusters(df)
    print("\nCluster summary:")
    print(_format_summary_table(summary_table).to_string(index=False))

    terms = top_terms_by_cluster(X_tfidf, df["cluster"].to_numpy(), vectorizer)
    cluster_examples = select_cluster_examples(cluster_profile, terms)
    print("\nRepresentative cluster examples:")
    print(cluster_examples.to_string(index=False))

    df_assigned, cluster_to_split = assign_clusters_to_splits(
        df=df,
        split_ratios=split_ratios,
        random_state=_RANDOM_STATE,
    )

    split_obj = build_split_json(
        df_assigned=df_assigned,
        cluster_to_split=cluster_to_split,
        metrics=metrics,
        chosen_k=chosen_k,
        split_ratios=split_ratios,
        random_state=_RANDOM_STATE,
    )

    validate_split(df, split_obj)

    output_path = save_split(split_obj, _DEFAULT_OUTPUT_PATH)
    print(f"\nSaved split -> {output_path}")

    print_final_summary(df_assigned, cluster_to_split, split_ratios)


if __name__ == "__main__":
    main()

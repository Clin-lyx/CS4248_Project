from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_b_encdec import rewrite_text
from systems.system_b_utils import (
    ARTIFACT_ROOT,
    count_true,
    infer_direction,
    load_anchored_data,
    load_style_scorer,
    safe_mean,
    score_style_probability,
    semantic_similarity_score,
    select_subset,
)
from systems.system_a.template_utils import preserves_anchors


def evaluate_system_b(
    split_name: str = "dev",
    split: str = "standard",
    limit: int = 0,
    output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    df = load_anchored_data()
    subset = select_subset(df, split_name=split_name, split=split)
    if limit > 0:
        subset = subset.head(limit).reset_index(drop=True)

    style_scorer = load_style_scorer()
    outputs: list[dict] = []
    similarities: list[float] = []
    anchor_flags: list[bool] = []
    style_flags: list[bool] = []

    for _, row in subset.iterrows():
        direction = infer_direction(int(row["label"]))
        rewritten = rewrite_text(row["text"], direction=direction)
        similarity = semantic_similarity_score(row["text"], rewritten)
        anchors_ok = preserves_anchors(row.get("anchors", {}), rewritten)
        prob = score_style_probability(rewritten, scorer=style_scorer)

        style_ok = True
        if prob is not None:
            style_ok = (prob >= 0.5) if direction == "n2s" else (prob < 0.5)

        outputs.append(
            {
                "id": row["id"],
                "direction": direction,
                "source_text": row["text"],
                "target_text": rewritten,
                "semantic_similarity": similarity,
                "anchors_preserved": anchors_ok,
                "prob_sarcastic_target": prob,
                "target_style_pass": style_ok,
            }
        )
        similarities.append(similarity)
        anchor_flags.append(anchors_ok)
        style_flags.append(style_ok)

    if output_path is None:
        output_path = ARTIFACT_ROOT / f"{split_name}_outputs.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_json(output_path, orient="records", lines=True, force_ascii=False)

    summary = {
        "split_name": split_name,
        "split_strategy": split,
        "num_examples": len(outputs),
        "avg_semantic_similarity": safe_mean(similarities),
        "anchor_preservation_rate": count_true(anchor_flags) / max(len(anchor_flags), 1),
        "target_style_pass_rate": count_true(style_flags) / max(len(style_flags), 1),
    }
    metrics_path = output_path.with_name(output_path.stem.replace("_outputs", "_metrics") + ".json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved outputs -> {output_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(json.dumps(summary, indent=2))
    return output_path, metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned System B outputs.")
    parser.add_argument("--split-name", default="dev")
    parser.add_argument("--split-strategy", default="standard")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_system_b(
        split_name=args.split_name,
        split=args.split_strategy,
        limit=args.limit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from similarity.semantic_similarity import batch_semantic_similarity
from systems.system_b_encdec import batch_generate
from systems.system_b_utils import (
    count_true,
    default_eval_output_path,
    default_model_dir,
    infer_direction,
    load_anchored_data,
    load_style_scorer,
    safe_mean,
    select_subset,
    SYSTEM_B_SPLIT_CHOICES,
)
from systems.system_a.template_utils import preserves_anchors


def _batch_style_probabilities(texts: list[str], scorer) -> list[float | None]:
    if scorer is None:
        return [None] * len(texts)
    try:
        probs = scorer.predict_proba(texts)
        return [float(row[1]) for row in probs]
    except Exception:
        return [None] * len(texts)


def evaluate_system_b(
    split_name: str = "test",
    split: str = "standard",
    limit: int = 0,
    output_path: str | Path | None = None,
    *,
    batch_size: int = 16,
    system_b_mode: str = "auto",
    finetuned_model_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    df = load_anchored_data()
    subset = select_subset(df, split_name=split_name, split=split)
    if limit > 0:
        subset = subset.head(limit).reset_index(drop=True)

    resolved_model_dir = Path(finetuned_model_dir) if finetuned_model_dir is not None else default_model_dir(split)
    style_scorer = load_style_scorer(split=split)
    texts = subset["text"].astype(str).tolist()
    directions = [infer_direction(int(label)) for label in subset["label"].tolist()]
    generated = batch_generate(
        texts,
        directions,
        k=1,
        batch_size=batch_size,
        mode=system_b_mode,
        finetuned_model_dir=resolved_model_dir,
    )
    rewritten_texts = [candidates[0] if candidates else original for candidates, original in zip(generated, texts)]
    similarities = batch_semantic_similarity(texts, rewritten_texts).tolist()
    probs = _batch_style_probabilities(rewritten_texts, style_scorer)

    outputs: list[dict] = []
    anchor_flags: list[bool] = []
    style_flags: list[bool] = []

    for idx, row in subset.iterrows():
        direction = directions[idx]
        rewritten = rewritten_texts[idx]
        similarity = float(similarities[idx])
        anchors_ok = preserves_anchors(row.get("anchors", {}), rewritten)
        prob = probs[idx]

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
        anchor_flags.append(anchors_ok)
        style_flags.append(style_ok)

    if output_path is None:
        output_path = default_eval_output_path(split_name=split_name, split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_json(output_path, orient="records", lines=True, force_ascii=False)

    summary = {
        "split_name": split_name,
        "split_strategy": split,
        "finetuned_model_dir": str(resolved_model_dir),
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
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--split",
        "--split-strategy",
        dest="split",
        default="standard",
        choices=SYSTEM_B_SPLIT_CHOICES,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--system-b-mode", default="auto", choices=["auto", "finetuned_local", "prompt_fallback"])
    parser.add_argument("--finetuned-model-dir", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_system_b(
        split_name=args.split_name,
        split=args.split,
        limit=args.limit,
        output_path=args.output,
        batch_size=args.batch_size,
        system_b_mode=args.system_b_mode,
        finetuned_model_dir=args.finetuned_model_dir,
    )


if __name__ == "__main__":
    main()

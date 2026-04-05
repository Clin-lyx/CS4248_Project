"""Legacy OpenAI bootstrap pipeline for System B pseudo-pairs.

Deprecated:
- This file belongs to the earlier API/bootstrap design for creating
  pseudo-pairs with an external model.
- It is kept only as historical/reference code.
- The current team workflow is the manual authoring pipeline centered on:
  `artifacts/data/cleaned_with_anchors.jsonl`,
  `artifacts/splits/standard.json`,
  `artifacts/system_b/pseudo_pairs_filtered.jsonl`,
  `systems/system_b_utils.py`, and
  `systems/system_a/template_utils.py`.

Do not use this file for the current manual contributor workflow unless the
team explicitly decides to revive the API-based path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_b_utils import (
    ARTIFACT_ROOT,
    FILTERED_PAIRS_PATH,
    RAW_PAIRS_PATH,
    finite_or_none,
    infer_direction,
    load_anchored_data,
    load_style_scorer,
    looks_too_similar,
    normalized_token_edit_distance,
    read_jsonl,
    score_style_probability,
    semantic_similarity_score,
    select_subset,
    target_style_passes,
    write_jsonl,
)
from systems.system_a.template_utils import preserves_anchors

DEFAULT_BOOTSTRAP_MODEL = os.getenv("SYSTEM_B_BOOTSTRAP_MODEL", "gpt-5.4")
DEFAULT_RAW_LIMIT = 0
PROMPT_VERSION = "system_b_bootstrap_v1"


def build_generation_instructions() -> str:
    return (
        "You create pseudo-parallel headline rewriting pairs for sarcasm style transfer. "
        "Rewrite the headline into the target style while preserving meaning, topic, named entities, "
        "numbers, dates, and places. Keep the output to a single short headline. Avoid explanations, "
        "avoid commentary, avoid adding new facts, and prefer minimal edits over full rewrites."
    )


def build_generation_input(source_text: str, direction: str, anchors: dict[str, Any] | None) -> str:
    target_style = "sarcastic Onion-style" if direction == "n2s" else "neutral factual"
    anchor_texts = [a.get("text", "") for a in (anchors or {}).get("all", []) if isinstance(a, dict)]
    anchor_line = ", ".join(anchor_texts) if anchor_texts else "None"

    return (
        f"Source headline: {source_text}\n"
        f"Target style: {target_style}\n"
        f"Direction code: {direction}\n"
        f"Required preserved anchors: {anchor_line}\n"
        "Return JSON only with one key: rewritten_headline."
    )


def call_openai_rewrite(
    client,
    source_text: str,
    direction: str,
    anchors: dict[str, Any] | None,
    model_name: str,
    max_output_tokens: int = 120,
) -> str:
    schema = {
        "type": "object",
        "properties": {
            "rewritten_headline": {"type": "string"},
        },
        "required": ["rewritten_headline"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=model_name,
        instructions=build_generation_instructions(),
        input=build_generation_input(source_text, direction, anchors),
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "headline_rewrite",
                "strict": True,
                "schema": schema,
            }
        },
    )
    payload = json.loads(response.output_text)
    return str(payload["rewritten_headline"]).strip()


def generate_raw_pairs(
    split_names: list[str],
    output_path: str | Path = RAW_PAIRS_PATH,
    split: str = "standard",
    model_name: str = DEFAULT_BOOTSTRAP_MODEL,
    limit: int = DEFAULT_RAW_LIMIT,
    start_at: int = 0,
) -> Path:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required to generate pseudo-pairs. "
            "Install it and set OPENAI_API_KEY before running this command."
        ) from exc

    client = OpenAI()
    df = load_anchored_data()

    selected_frames: list[pd.DataFrame] = []
    for split_name in split_names:
        part = select_subset(df, split_name=split_name, split=split).copy()
        part["original_split"] = split_name
        selected_frames.append(part)

    if not selected_frames:
        raise ValueError("No split names provided.")

    work_df = pd.concat(selected_frames, ignore_index=True)
    if start_at:
        work_df = work_df.iloc[start_at:].reset_index(drop=True)
    if limit > 0:
        work_df = work_df.head(limit).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for idx, row in work_df.iterrows():
        direction = infer_direction(int(row["label"]))
        target_text = call_openai_rewrite(
            client=client,
            source_text=row["text"],
            direction=direction,
            anchors=row.get("anchors", {}),
            model_name=model_name,
        )
        records.append(
            {
                "id": row["id"],
                "source_text": row["text"],
                "source_label": int(row["label"]),
                "direction": direction,
                "target_text": target_text,
                "anchors": row.get("anchors", {}),
                "generator_model": model_name,
                "prompt_version": PROMPT_VERSION,
                "original_split": row["original_split"],
                "row_index": int(idx + start_at),
            }
        )

    output_path = write_jsonl(records, output_path)
    print(f"Saved {len(records)} raw pseudo-pairs -> {output_path}")
    return output_path


def filter_pairs(
    input_path: str | Path = RAW_PAIRS_PATH,
    output_path: str | Path = FILTERED_PAIRS_PATH,
    min_similarity: float = 0.55,
    min_target_style_confidence: float = 0.65,
    max_edit_ratio: float = 0.7,
    skip_style_filter: bool = False,
) -> Path:
    raw_records = read_jsonl(input_path)
    style_scorer = None if skip_style_filter else load_style_scorer()

    accepted: list[dict[str, Any]] = []
    for record in raw_records:
        source_text = record["source_text"]
        target_text = str(record["target_text"]).strip()
        direction = record["direction"]
        anchors = record.get("anchors", {})

        similarity = semantic_similarity_score(source_text, target_text)
        edit_ratio = normalized_token_edit_distance(source_text, target_text)
        source_prob = finite_or_none(score_style_probability(source_text, scorer=style_scorer))
        target_prob = finite_or_none(score_style_probability(target_text, scorer=style_scorer))
        anchors_ok = preserves_anchors(anchors, target_text)

        rejection_reason = None
        if not target_text:
            rejection_reason = "empty_target"
        elif looks_too_similar(target_text, source_text):
            rejection_reason = "too_similar"
        elif not anchors_ok:
            rejection_reason = "anchor_loss"
        elif similarity < min_similarity:
            rejection_reason = "low_similarity"
        elif edit_ratio > max_edit_ratio:
            rejection_reason = "edit_ratio_too_high"
        elif not target_style_passes(direction, target_prob, min_target_style_confidence):
            rejection_reason = "weak_style_flip"

        enriched = dict(record)
        enriched.update(
            {
                "semantic_similarity": similarity,
                "edit_ratio": edit_ratio,
                "style_score_source": source_prob,
                "style_score_target": target_prob,
                "anchors_preserved": anchors_ok,
                "accepted": rejection_reason is None,
                "rejection_reason": rejection_reason,
            }
        )

        if rejection_reason is None:
            accepted.append(enriched)

    output_path = write_jsonl(accepted, output_path)
    print(f"Saved {len(accepted)} filtered pseudo-pairs -> {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and filter pseudo-pairs for System B.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate raw pseudo-pairs with an OpenAI model.")
    gen.add_argument("--splits", default="train", help="Comma-separated split names, e.g. train or train,dev")
    gen.add_argument("--split-strategy", default="standard", help="Split strategy name from data.splits.SPLIT_REGISTRY")
    gen.add_argument("--output", default=str(RAW_PAIRS_PATH))
    gen.add_argument("--model", default=DEFAULT_BOOTSTRAP_MODEL)
    gen.add_argument("--limit", type=int, default=DEFAULT_RAW_LIMIT)
    gen.add_argument("--start-at", type=int, default=0)

    filt = sub.add_parser("filter", help="Filter raw pseudo-pairs.")
    filt.add_argument("--input", default=str(RAW_PAIRS_PATH))
    filt.add_argument("--output", default=str(FILTERED_PAIRS_PATH))
    filt.add_argument("--min-similarity", type=float, default=0.55)
    filt.add_argument("--min-target-style-confidence", type=float, default=0.65)
    filt.add_argument("--max-edit-ratio", type=float, default=0.7)
    filt.add_argument("--skip-style-filter", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
        generate_raw_pairs(
            split_names=split_names,
            output_path=args.output,
            split=args.split_strategy,
            model_name=args.model,
            limit=args.limit,
            start_at=args.start_at,
        )
        return

    if args.command == "filter":
        filter_pairs(
            input_path=args.input,
            output_path=args.output,
            min_similarity=args.min_similarity,
            min_target_style_confidence=args.min_target_style_confidence,
            max_edit_ratio=args.max_edit_ratio,
            skip_style_filter=args.skip_style_filter,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

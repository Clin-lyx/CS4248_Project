"""
Convert slim bottom-up JSONL (e.g. Bottom_Up_1000_pairs.jsonl) into the full
manual pseudo-pair schema from SYSTEM_B_MANUAL_WORKFLOW.md.

- Replaces string-list `anchors` with the canonical `anchors` object from
  artifacts/data/cleaned_with_anchors.jsonl (required for preserves_anchors).
- Sets `source_text` from cleaned data (headline ground truth).
- Fills semantic_similarity, edit_ratio, anchors_preserved.
- Keeps only train IDs in standard.json unless --allow-non-train.

Example:

  .venv\\Scripts\\python.exe systems\\ingest_bottom_up_jsonl.py ^
    artifacts/system_b/Bottom_Up_1000_pairs.jsonl ^
    artifacts/system_b/bottom_up_manual_ready.jsonl ^
    --prompt-version manual_batch_bottom_up_1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_a.template_utils import normalize_space, preserves_anchors
from systems.system_b_utils import (
    FILTERED_PAIRS_PATH,
    infer_direction,
    normalized_token_edit_distance,
    semantic_similarity_score,
)


def load_json_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def load_cleaned_by_id(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["id"]] = r
    return out


def load_train_ids() -> set[str]:
    split_path = PROJECT_ROOT / "artifacts" / "splits" / "standard.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return set(payload["train"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert bottom-up JSONL to manual workflow schema.")
    parser.add_argument("input_jsonl", type=Path, help="Slim JSONL with id, source_text, source_label, direction, target_text")
    parser.add_argument("output_jsonl", type=Path, help="Output path (full schema rows)")
    parser.add_argument(
        "--prompt-version",
        required=True,
        help="Batch tag, e.g. manual_batch_020_ids_...",
    )
    parser.add_argument(
        "--generator-model",
        default="chatgpt_manual_gpt5_4",
        help="Provenance string stored in generator_model",
    )
    parser.add_argument(
        "--skip-existing-ids",
        action="store_true",
        help=f"Skip IDs already present in {FILTERED_PAIRS_PATH.name}",
    )
    parser.add_argument(
        "--no-strict-anchors",
        action="store_true",
        help="Emit rows even if anchors are not preserved (sets accepted=false, rejection_reason set)",
    )
    parser.add_argument(
        "--allow-non-train",
        action="store_true",
        help="Do not require IDs to be in standard.json train split",
    )
    args = parser.parse_args()
    strict_anchors = not args.no_strict_anchors

    cleaned_path = PROJECT_ROOT / "artifacts/data/cleaned_with_anchors.jsonl"
    cleaned = load_cleaned_by_id(cleaned_path)
    train_ids = load_train_ids()
    existing: set[str] = set()
    if args.skip_existing_ids and FILTERED_PAIRS_PATH.exists():
        existing = load_json_ids(FILTERED_PAIRS_PATH)

    stats = {
        "read": 0,
        "written": 0,
        "skipped_no_cleaned": 0,
        "skipped_not_train": 0,
        "skipped_duplicate": 0,
        "skipped_direction_mismatch": 0,
        "skipped_anchor": 0,
        "warn_source_mismatch": 0,
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open(encoding="utf-8") as fin, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            raw = json.loads(line)
            rid = raw["id"]
            target_text = raw["target_text"]

            if rid in existing:
                stats["skipped_duplicate"] += 1
                continue

            c = cleaned.get(rid)
            if not c:
                stats["skipped_no_cleaned"] += 1
                print(f"[skip] no cleaned row: {rid}", file=sys.stderr)
                continue

            if not args.allow_non_train and rid not in train_ids:
                stats["skipped_not_train"] += 1
                print(f"[skip] not in train split: {rid}", file=sys.stderr)
                continue

            source_text = c["text"]
            label = int(c["label"])
            direction = infer_direction(label)
            if raw.get("direction") != direction:
                stats["skipped_direction_mismatch"] += 1
                print(
                    f"[skip] direction/label mismatch for {rid}: "
                    f"file says {raw.get('direction')!r}, cleaned label {label} -> {direction}",
                    file=sys.stderr,
                )
                continue

            if normalize_space(raw.get("source_text", "")) != normalize_space(source_text):
                stats["warn_source_mismatch"] += 1
                print(
                    f"[warn] source_text differs from cleaned for {rid}; using cleaned text",
                    file=sys.stderr,
                )

            anchors = c.get("anchors") or {}
            anchor_ok = preserves_anchors(anchors, target_text)
            if strict_anchors and not anchor_ok:
                stats["skipped_anchor"] += 1
                print(f"[skip] anchors not preserved: {rid}", file=sys.stderr)
                continue

            sim = semantic_similarity_score(source_text, target_text)
            edit = normalized_token_edit_distance(source_text, target_text)

            row = {
                "id": rid,
                "source_text": source_text,
                "source_label": label,
                "direction": direction,
                "target_text": target_text,
                "anchors": anchors,
                "generator_model": args.generator_model,
                "prompt_version": args.prompt_version,
                "original_split": "train",
                "semantic_similarity": sim,
                "edit_ratio": edit,
                "style_score_source": None,
                "style_score_target": None,
                "anchors_preserved": anchor_ok,
                "accepted": anchor_ok,
                "rejection_reason": None if anchor_ok else "anchors_not_preserved",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

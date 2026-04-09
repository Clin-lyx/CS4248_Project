from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(path)}


def summarize_case(case_id: str, full_row: dict[str, Any], base_row: dict[str, Any], label: str) -> str:
    full_scores = full_row.get("scores", {})
    base_scores = base_row.get("scores", {})
    return "\n".join(
        [
            f"### {case_id} — {label}",
            f"- input: {full_row['input_text']}",
            f"- full output: {full_row['output_text']}",
            f"- baseline output: {base_row['output_text']}",
            f"- full valid: {full_scores.get('is_valid')} | baseline valid: {base_scores.get('is_valid')}",
            f"- full style: {full_scores.get('target_style_score')} | baseline style: {base_scores.get('target_style_score')}",
            f"- full semantic: {full_scores.get('semantic_similarity')} | baseline semantic: {base_scores.get('semantic_similarity')}",
            f"- full anchors: {full_scores.get('anchor_retention_rate')} | baseline anchors: {base_scores.get('anchor_retention_rate')}",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Week 4 System C trade-off stories.")
    ap.add_argument("--full", default="results/system_c_best_standard_dev.jsonl")
    ap.add_argument("--baseline", default="results/system_c_best_standard_dev_no_retrieval.jsonl")
    ap.add_argument("--output", default="analysis/system_c_tradeoff_examples.md")
    args = ap.parse_args()

    full = load_map(PROJECT_ROOT / args.full)
    base = load_map(PROJECT_ROOT / args.baseline)

    helped: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    hurt: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []

    for case_id, full_row in full.items():
        if case_id not in base:
            continue
        base_row = base[case_id]
        fs = full_row.get("scores", {})
        bs = base_row.get("scores", {})

        full_valid = bool(fs.get("is_valid", False))
        base_valid = bool(bs.get("is_valid", False))
        full_final = float(full_row.get("metadata", {}).get("final_score", 0.0) or 0.0)
        base_final = float(base_row.get("metadata", {}).get("final_score", 0.0) or 0.0)

        if full_valid and not base_valid:
            helped.append((case_id, full_row, base_row, "full valid / baseline invalid"))
        elif (not full_valid) and base_valid:
            hurt.append((case_id, full_row, base_row, "full invalid / baseline valid"))
        elif full_final > base_final + 0.15:
            helped.append((case_id, full_row, base_row, "higher full final score"))
        elif base_final > full_final + 0.15:
            hurt.append((case_id, full_row, base_row, "higher baseline final score"))

    lines = [
        "# System C trade-off stories",
        "",
        f"Compared full run `{args.full}` against baseline `{args.baseline}`.",
        "",
        "## Cases where full System C helps",
        "",
    ]
    for item in helped[:20]:
        lines.append(summarize_case(*item))

    lines.extend(["", "## Cases where full System C hurts", ""])
    for item in hurt[:20]:
        lines.append(summarize_case(*item))

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved trade-off report -> {out}")


if __name__ == "__main__":
    main()

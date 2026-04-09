from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocess import normalize_headline
from systems.system_a.template_utils import preserves_anchors


def normalize_lower_text(text: Any) -> str:
    return normalize_headline("" if text is None else str(text)).lower()


def normalize_anchors(anchors: Any) -> Any:
    if isinstance(anchors, list):
        return [normalize_lower_text(item) for item in anchors]

    if not isinstance(anchors, dict):
        return anchors

    normalized: dict[str, Any] = {}
    for key, value in anchors.items():
        if isinstance(value, list):
            items: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    updated = dict(item)
                    if "text" in updated:
                        updated["text"] = normalize_lower_text(updated["text"])
                    items.append(updated)
                else:
                    items.append(normalize_lower_text(item))
            normalized[key] = items
        elif key == "text":
            normalized[key] = normalize_lower_text(value)
        else:
            normalized[key] = value
    return normalized


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)

    if "source_text" in updated:
        updated["source_text"] = normalize_lower_text(updated["source_text"])
    if "target_text" in updated:
        updated["target_text"] = normalize_lower_text(updated["target_text"])
    if "anchors" in updated:
        updated["anchors"] = normalize_anchors(updated["anchors"])

    if "anchors_preserved" in updated and "anchors" in updated and "target_text" in updated:
        updated["anchors_preserved"] = preserves_anchors(updated["anchors"], updated["target_text"])

    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize pseudo-pair JSONL text with preprocess.normalize_headline and lowercase output.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "system_b" / "bottom_up_manual_ready_v2.jsonl",
        help="Input JSONL file to normalize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "system_b" / "bottom_up_manual_ready_v2_lowercase.jsonl",
        help="Output JSONL file for normalized rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0
    anchor_failures = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {input_path}: {exc}") from exc

            normalized = normalize_row(row)
            rows_read += 1

            if (
                "anchors_preserved" in normalized
                and normalized["anchors_preserved"] is False
            ):
                anchor_failures += 1

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Read {rows_read} rows from {input_path}")
    print(f"Wrote {rows_written} rows to {output_path}")
    print(f"Anchor failures after normalization: {anchor_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

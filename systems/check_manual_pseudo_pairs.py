from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_a.template_utils import preserves_anchors
from systems.system_b_utils import FILTERED_PAIRS_PATH

REQUIRED_FIELDS = {
    "id",
    "source_text",
    "source_label",
    "direction",
    "target_text",
    "anchors",
    "generator_model",
    "prompt_version",
    "original_split",
    "semantic_similarity",
    "edit_ratio",
    "style_score_source",
    "style_score_target",
    "anchors_preserved",
    "accepted",
    "rejection_reason",
}


def load_train_ids() -> set[str]:
    split_path = PROJECT_ROOT / "artifacts" / "splits" / "standard.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return set(payload["train"])


def main() -> int:
    path = Path(FILTERED_PAIRS_PATH)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON on line {line_no}: {exc}")
                return 1

    train_ids = load_train_ids()
    seen_ids: set[str] = set()
    direction_counts: Counter[str] = Counter()
    batch_counts: Counter[str] = Counter()

    missing_fields = 0
    duplicate_ids = 0
    non_train_rows = 0
    anchor_failures = 0

    for row in rows:
        missing = REQUIRED_FIELDS - set(row.keys())
        if missing:
            missing_fields += 1
            print(f"Missing fields for {row.get('id', '<unknown>')}: {sorted(missing)}")

        row_id = row.get("id")
        if row_id in seen_ids:
            duplicate_ids += 1
            print(f"Duplicate id: {row_id}")
        else:
            seen_ids.add(row_id)

        if row.get("original_split") != "train" or row_id not in train_ids:
            non_train_rows += 1
            print(f"Non-train row: {row_id}")

        anchors_ok = preserves_anchors(row.get("anchors", {}), row.get("target_text", ""))
        if not anchors_ok or not row.get("anchors_preserved", False):
            anchor_failures += 1
            print(f"Anchor failure: {row_id}")

        direction_counts.update([str(row.get("direction"))])
        batch_counts.update([str(row.get("prompt_version"))])

    print(f"Rows: {len(rows)}")
    print(f"Directions: {dict(direction_counts)}")
    print("Prompt versions:")
    for name, count in sorted(batch_counts.items()):
        print(f"  {name}: {count}")

    print("Checks:")
    print(f"  missing_fields={missing_fields}")
    print(f"  duplicate_ids={duplicate_ids}")
    print(f"  non_train_rows={non_train_rows}")
    print(f"  anchor_failures={anchor_failures}")

    if any([missing_fields, duplicate_ids, non_train_rows, anchor_failures]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

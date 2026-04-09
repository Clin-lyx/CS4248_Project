from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_b_utils import read_jsonl, write_jsonl

TRAIN_PAIRS_PATH = PROJECT_ROOT / "artifacts" / "system_b" / "pseudo_pairs_filtered.jsonl"
DEV_PAIRS_PATH = PROJECT_ROOT / "artifacts" / "system_b" / "pseudo_pairs_filtered_dev_normalized.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "system_b" / "pseudo_pairs_author_balanced.jsonl"


def _assign_train_authors(rows: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for idx, row in enumerate(rows):
        updated = dict(row)
        updated["author_id"] = "author_1" if idx < 2200 else "author_2"
        enriched.append(updated)
    return enriched


def _assign_dev_author(rows: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        updated = dict(row)
        updated["author_id"] = "author_3"
        enriched.append(updated)
    return enriched


def _assign_training_splits(rows: list[dict], *, dev_ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["author_id"]), str(row["direction"]))
        grouped.setdefault(key, []).append(dict(row))

    assigned: list[dict] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        rng.shuffle(bucket)
        dev_size = max(1, int(len(bucket) * dev_ratio))
        for idx, row in enumerate(bucket):
            row["training_split"] = "dev" if idx < dev_size else "train"
            assigned.append(row)

    assigned.sort(key=lambda row: (row["id"], row["author_id"], row["direction"]))
    return assigned


def build_author_balanced_pairs(
    train_path: str | Path = TRAIN_PAIRS_PATH,
    dev_path: str | Path = DEV_PAIRS_PATH,
    output_path: str | Path = OUTPUT_PATH,
    *,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    train_rows = _assign_train_authors(read_jsonl(train_path))
    dev_rows = _assign_dev_author(read_jsonl(dev_path))
    all_rows = train_rows + dev_rows
    assigned_rows = _assign_training_splits(all_rows, dev_ratio=dev_ratio, seed=seed)

    output_path = write_jsonl(assigned_rows, output_path)

    counts_by_author = Counter(row["author_id"] for row in assigned_rows)
    counts_by_training_split = Counter(row["training_split"] for row in assigned_rows)
    counts_by_author_split = Counter((row["author_id"], row["training_split"]) for row in assigned_rows)
    counts_by_author_direction_split = Counter(
        (row["author_id"], row["direction"], row["training_split"]) for row in assigned_rows
    )

    print(f"Saved author-balanced pseudo-pairs -> {output_path}")
    print(json.dumps(
        {
            "num_rows": len(assigned_rows),
            "dev_ratio": dev_ratio,
            "seed": seed,
            "by_author": dict(sorted(counts_by_author.items())),
            "by_training_split": dict(sorted(counts_by_training_split.items())),
            "by_author_split": {
                f"{author}:{split}": count
                for (author, split), count in sorted(counts_by_author_split.items())
            },
            "by_author_direction_split": {
                f"{author}:{direction}:{split}": count
                for (author, direction, split), count in sorted(counts_by_author_direction_split.items())
            },
        },
        indent=2,
    ))
    return Path(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an author-balanced combined pseudo-pair training file.")
    parser.add_argument("--train-input", default=str(TRAIN_PAIRS_PATH))
    parser.add_argument("--dev-input", default=str(DEV_PAIRS_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_author_balanced_pairs(
        train_path=args.train_input,
        dev_path=args.dev_input,
        output_path=args.output,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

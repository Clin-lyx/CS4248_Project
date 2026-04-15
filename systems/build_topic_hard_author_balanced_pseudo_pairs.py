from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "artifacts" / "system_b" / "pseudo_pairs_author_balanced.jsonl"
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "splits" / "topic_hard.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "system_b" / "pseudo_pairs_author_balanced_topic_hard.jsonl"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def load_split_ids(path: str | Path, split_name: str) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {path}")
    return {str(item) for item in payload[split_name]}


def allocate_proportional_quotas(counts: dict[str, int], target_total: int) -> dict[str, int]:
    total_available = sum(counts.values())
    if target_total > total_available:
        raise ValueError(f"Requested {target_total} rows from only {total_available} available rows.")

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []
    assigned = 0

    for key in sorted(counts):
        count = counts[key]
        exact = (target_total * count) / total_available
        base = min(count, int(exact))
        quotas[key] = base
        assigned += base
        remainders.append((exact - base, count - base, key))

    remaining = target_total - assigned
    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for _, capacity, key in remainders:
        if remaining <= 0:
            break
        if capacity <= 0:
            continue
        quotas[key] += 1
        remaining -= 1

    if remaining != 0:
        raise RuntimeError("Failed to allocate proportional quotas exactly.")
    return quotas


def select_balanced_rows(rows: list[dict[str, Any]], *, target_size: int | None, seed: int) -> list[dict[str, Any]]:
    if target_size is None or len(rows) <= target_size:
        return [dict(row) for row in rows]

    grouped_by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_by_direction.setdefault(str(row["direction"]), []).append(dict(row))

    directions = sorted(grouped_by_direction)
    if len(directions) != 2:
        raise ValueError(f"Expected exactly 2 direction buckets, found {directions}.")

    max_balanced_total = min(len(grouped_by_direction[direction]) for direction in directions) * len(directions)
    actual_target_size = min(target_size, max_balanced_total)
    actual_target_size -= actual_target_size % len(directions)
    if actual_target_size <= 0:
        raise ValueError("No rows available after balancing.")

    per_direction_target = actual_target_size // len(directions)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for direction in directions:
        by_author: dict[str, list[dict[str, Any]]] = {}
        for row in grouped_by_direction[direction]:
            by_author.setdefault(str(row["author_id"]), []).append(dict(row))

        quotas = allocate_proportional_quotas(
            {author_id: len(bucket) for author_id, bucket in by_author.items()},
            per_direction_target,
        )

        for author_id in sorted(by_author):
            bucket = by_author[author_id]
            rng.shuffle(bucket)
            selected.extend(bucket[: quotas[author_id]])

    return selected


def assign_training_splits(rows: list[dict[str, Any]], *, dev_ratio: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["author_id"]), str(row["direction"]))
        grouped.setdefault(key, []).append(dict(row))

    assigned: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        rng.shuffle(bucket)
        dev_size = max(1, int(len(bucket) * dev_ratio))
        for idx, row in enumerate(bucket):
            row["training_split"] = "dev" if idx < dev_size else "train"
            assigned.append(row)

    assigned.sort(key=lambda row: (row["id"], row["author_id"], row["direction"]))
    return assigned


def build_topic_hard_pairs(
    input_path: str | Path = INPUT_PATH,
    split_path: str | Path = SPLIT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    *,
    excluded_split_name: str = "test",
    target_size: int | None = 6000,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    rows = read_jsonl(input_path)
    excluded_ids = load_split_ids(split_path, excluded_split_name)
    filtered_rows = [dict(row) for row in rows if str(row["id"]) not in excluded_ids]
    selected_rows = select_balanced_rows(filtered_rows, target_size=target_size, seed=seed)
    assigned_rows = assign_training_splits(selected_rows, dev_ratio=dev_ratio, seed=seed)
    output_path = write_jsonl(assigned_rows, output_path)

    counts_by_direction = Counter(row["direction"] for row in assigned_rows)
    counts_by_source_label = Counter(row["source_label"] for row in assigned_rows)
    counts_by_author = Counter(row["author_id"] for row in assigned_rows)
    counts_by_training_split = Counter(row["training_split"] for row in assigned_rows)
    counts_by_author_direction = Counter((row["author_id"], row["direction"]) for row in assigned_rows)
    counts_by_author_direction_split = Counter(
        (row["author_id"], row["direction"], row["training_split"]) for row in assigned_rows
    )

    print(f"Saved topic-hard filtered pseudo-pairs -> {output_path}")
    print(json.dumps(
        {
            "source_rows": len(rows),
            "excluded_split_name": excluded_split_name,
            "excluded_id_count": len(excluded_ids),
            "overlap_removed": len(rows) - len(filtered_rows),
            "filtered_rows": len(filtered_rows),
            "target_size_requested": target_size,
            "target_size_actual": len(assigned_rows),
            "dev_ratio": dev_ratio,
            "seed": seed,
            "by_direction": dict(sorted(counts_by_direction.items())),
            "by_source_label": {str(key): value for key, value in sorted(counts_by_source_label.items())},
            "by_author": dict(sorted(counts_by_author.items())),
            "by_training_split": dict(sorted(counts_by_training_split.items())),
            "by_author_direction": {
                f"{author_id}:{direction}": count
                for (author_id, direction), count in sorted(counts_by_author_direction.items())
            },
            "by_author_direction_split": {
                f"{author_id}:{direction}:{training_split}": count
                for (author_id, direction, training_split), count in sorted(counts_by_author_direction_split.items())
            },
        },
        indent=2,
    ))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter author-balanced pseudo-pairs against a split holdout and rebalance them."
    )
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--split-path", default=str(SPLIT_PATH))
    parser.add_argument("--excluded-split-name", default="test")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--target-size", type=int, default=6000)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_topic_hard_pairs(
        input_path=args.input,
        split_path=args.split_path,
        output_path=args.output,
        excluded_split_name=args.excluded_split_name,
        target_size=args.target_size,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

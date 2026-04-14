from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_b_utils import (
    build_seq2seq_input,
    default_model_dir,
    default_pair_dataset_path,
    default_train_metrics_path,
    read_jsonl,
    SYSTEM_B_SPLIT_CHOICES,
)


class PseudoPairDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "input_text": build_seq2seq_input(row["source_text"], row["direction"]),
            "target_text": row["target_text"],
            "id": row["id"],
            "direction": row["direction"],
        }


def split_rows(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    training_split_rows = [r for r in rows if r.get("training_split") in {"train", "dev"}]
    if training_split_rows:
        train_rows = [r for r in training_split_rows if r.get("training_split") == "train"]
        dev_rows = [r for r in training_split_rows if r.get("training_split") == "dev"]
        if not train_rows:
            raise ValueError("No training rows found in training_split='train'.")
        if not dev_rows:
            raise ValueError("No dev rows found in training_split='dev'.")
        return train_rows, dev_rows

    train_rows = [r for r in rows if r.get("original_split") == "train"]
    dev_rows = [r for r in rows if r.get("original_split") == "dev"]

    if dev_rows:
        return train_rows, dev_rows

    if not train_rows:
        raise ValueError("No training rows found in the selected pseudo-pair file.")

    rng = random.Random(seed)
    shuffled = train_rows[:]
    rng.shuffle(shuffled)

    dev_size = max(1, int(len(shuffled) * dev_ratio))
    dev_rows = shuffled[:dev_size]
    train_rows = shuffled[dev_size:]
    return train_rows, dev_rows


def make_collate_fn(tokenizer, max_source_length: int, max_target_length: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        input_texts = [item["input_text"] for item in batch]
        target_texts = [item["target_text"] for item in batch]

        model_inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        target_batch = tokenizer(
            text_target=target_texts,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )

        labels = target_batch["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    return collate


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train_system_b(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str = "google/flan-t5-base",
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    max_source_length: int = 96,
    max_target_length: int = 48,
    dev_ratio: float = 0.1,
    seed: int = 42,
    split: str = "standard",
) -> Path:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is required to train System B. "
            "Install transformers and sentencepiece before running training."
        ) from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    resolved_input_path = Path(input_path) if input_path is not None else default_pair_dataset_path(split)
    resolved_output_dir = Path(output_dir) if output_dir is not None else default_model_dir(split)
    metrics_path = default_train_metrics_path(split)

    rows = read_jsonl(resolved_input_path)
    train_rows, dev_rows = split_rows(rows, dev_ratio=dev_ratio, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_ds = PseudoPairDataset(train_rows)
    dev_ds = PseudoPairDataset(dev_rows)
    collate = make_collate_fn(tokenizer, max_source_length, max_target_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_dev_loss = float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        dev_loss = evaluate_loss(model, dev_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)

    metrics = {
        "model_name": model_name,
        "pair_split": split,
        "input_path": str(resolved_input_path),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "history": history,
        "best_dev_loss": best_dev_loss,
        "config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "seed": seed,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved System B model -> {resolved_output_dir}")
    print(f"Saved training metrics -> {metrics_path}")
    return resolved_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train System B on split-aware pseudo-pairs.")
    parser.add_argument(
        "--split",
        default="standard",
        choices=SYSTEM_B_SPLIT_CHOICES,
        help="Which pseudo-pair set to train on. Used to resolve the default --input and --output-dir.",
    )
    parser.add_argument("--input", default=None, help="Override the pseudo-pair JSONL path.")
    parser.add_argument("--output-dir", default=None, help="Override the fine-tuned model output directory.")
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=96)
    parser.add_argument("--max-target-length", type=int, default=48)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_system_b(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        split=args.split,
    )


if __name__ == "__main__":
    main()


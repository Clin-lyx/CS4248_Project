"""
Training and inference pipeline for transformer-based sarcasm detection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifiers.common import (
    compute_metrics,
    format_prediction,
    load_split_frames,
    write_predictions,
)

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "classifiers" / "transformer"



class HeadlineDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def evaluate(model: AutoModelForSequenceClassification, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels.numpy())

    prob_arr = np.concatenate(probs_all)
    label_arr = np.concatenate(labels_all).astype(np.int64)
    metrics = compute_metrics(label_arr, prob_arr)
    return metrics, prob_arr, label_arr


def train(
    model_name: str = "distilbert-base-uncased",
    text_col: str = "text",
    output_dir: Path = None,
    max_length: int = 64,
    batch_size: int = 16,
    lr: float = 2e-5,
    epochs: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df, dev_df, test_df = load_split_frames(text_col)

    train_texts = train_df[text_col].astype(str).tolist()
    dev_texts = dev_df[text_col].astype(str).tolist()
    test_texts = test_df[text_col].astype(str).tolist()

    y_train = train_df["label"].astype(int).tolist()
    y_dev = dev_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()

    if output_dir is None:
        output_dir = ARTIFACT_ROOT
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = HeadlineDataset(train_texts, y_train, tokenizer, max_length=max_length)
    dev_ds = HeadlineDataset(dev_texts, y_dev, tokenizer, max_length=max_length)
    test_ds = HeadlineDataset(test_texts, y_test, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_dev_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())

        dev_metrics, _, _ = evaluate(model, dev_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, dev_f1={dev_metrics['f1']:.4f}")

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Transformer training failed to produce a model state.")

    model.load_state_dict(best_state)
    dev_metrics, _, _ = evaluate(model, dev_loader, device)
    test_metrics, _, _ = evaluate(model, test_loader, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "dev": dev_metrics,
        "test": test_metrics,
        "train_config": {
            "model_name": model_name,
            "text_col": text_col,
            "max_length": max_length,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model + tokenizer -> {output_dir}")
    print(f"Saved metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


def predict(
    model_dir: Path = None,
    texts: list[str] = None,
    max_length: int = 64,
    batch_size: int = 16,
    fallback_model_name: str = "distilbert-base-uncased",
) -> list[dict[str, Any]]:
    if model_dir is None:
        model_dir = ARTIFACT_ROOT
    if texts is None:
        texts = []

    if model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        print(
            f"Local transformer model directory not found: {model_dir}. "
            f"Falling back to pretrained model '{fallback_model_name}'."
        )
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            fallback_model_name,
            num_labels=2,
        )

    ds = HeadlineDataset(texts, [0] * len(texts), tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch_inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for p in probs:
                p = float(p)
                outputs.append(format_prediction("", int(p >= 0.5), p))

    for i, text in enumerate(texts):
        outputs[i]["text"] = text
    return outputs


def main() -> None:
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transformer_classifier.py train")
        print("       python transformer_classifier.py predict <text>")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "train":
        train()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage: python transformer_classifier.py predict <text>")
            return
        text = " ".join(sys.argv[2:])
        preds = predict(texts=[text])
        write_predictions(preds, None)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()


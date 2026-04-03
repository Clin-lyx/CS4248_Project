"""
Training and inference pipeline for BiLSTM (RNN) sarcasm detection.

Uses artifacts/data/cleaned.jsonl and frozen split IDs from
artifacts/splits/standard.json via data.splits helpers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifiers.common import (
    compute_metrics,
    format_prediction,
    load_split_frames,
    write_predictions,
)
from data.splits import get_split_df, load_cleaned_data

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "classifiers" / "rnn"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")



def simple_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int) -> dict[str, int]:
    freq: dict[str, int] = {}
    for text in texts:
        for tok in simple_tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in simple_tokenize(text)]
    return ids if ids else [vocab["<UNK>"]]


class TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int]) -> None:
        self.encoded = [encode_text(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.encoded[idx], self.labels[idx]


def collate_batch(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    max_len = int(lengths.max().item())

    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return tokens, lengths, labels


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Mean-pool valid time steps only.
        mask = (tokens != 0).unsqueeze(-1)
        masked = out * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom

        logits = self.fc(self.dropout(pooled)).squeeze(-1)
        return logits


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            logits = model(tokens, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    prob_arr = np.concatenate(all_probs)
    label_arr = np.concatenate(all_labels).astype(np.int64)
    return compute_metrics(label_arr, prob_arr), prob_arr, label_arr


def train(
    text_col: str = "text",
    output_dir: Path = None,
    emb_dim: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    min_freq: int = 2,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    split: str = "standard",
) -> dict:
    """Train BiLSTM classifier. Returns the metrics dict."""
    if output_dir is None:
        output_dir = ARTIFACT_ROOT

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df, dev_df, test_df = load_split_frames(text_col, split=split)

    train_texts = train_df[text_col].astype(str).tolist()
    dev_texts = dev_df[text_col].astype(str).tolist()
    test_texts = test_df[text_col].astype(str).tolist()

    y_train = train_df["label"].astype(int).tolist()
    y_dev = dev_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()

    vocab = build_vocab(train_texts, min_freq=min_freq)

    train_ds = TextLabelDataset(train_texts, y_train, vocab)
    dev_ds = TextLabelDataset(dev_texts, y_dev, vocab)
    test_ds = TextLabelDataset(test_texts, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(len(vocab), emb_dim, hidden_dim, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_dev_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for tokens, lengths, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(tokens, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        dev_metrics, _, _ = evaluate(model, dev_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, dev_f1={dev_metrics['f1']:.4f}")

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("RNN training failed to produce a model state.")

    model.load_state_dict(best_state)
    dev_metrics, _, _ = evaluate(model, dev_loader, device)
    test_metrics, _, _ = evaluate(model, test_loader, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"

    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "model_type": "rnn",
            "text_col": text_col,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "min_freq": min_freq,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "dev": dev_metrics,
        "test": test_metrics,
        "train_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model -> {checkpoint_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


def predict(model_dir: Path = None, texts: list[str] = None, batch_size: int = 64) -> list[dict[str, Any]]:
    if model_dir is None:
        model_dir = ARTIFACT_ROOT
    if texts is None:
        texts = []
    
    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RNN model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab: dict[str, int] = checkpoint["vocab"]
    config = checkpoint["config"]

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        emb_dim=int(config["emb_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    ds = TextLabelDataset(texts=texts, labels=[0] * len(texts), vocab=vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for tokens, lengths, _ in loader:
            logits = model(tokens, lengths)
            probs = torch.sigmoid(logits).numpy()
            for p in probs:
                p = float(p)
                pred_label = int(p >= 0.5)
                outputs.append(format_prediction("", pred_label, p))

    for i, text in enumerate(texts):
        outputs[i]["text"] = text
    return outputs


def main() -> None:
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rnn_classifier.py train")
        print("       python rnn_classifier.py predict <text>")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "train":
        train()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage: python rnn_classifier.py predict <text>")
            return
        text = " ".join(sys.argv[2:])
        preds = predict(texts=[text])
        write_predictions(preds, None)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()


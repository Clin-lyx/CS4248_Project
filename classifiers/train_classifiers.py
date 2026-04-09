"""
Train sarcasm classifiers for a frozen split (standard or topic_hard).

Checkpoints are written under artifacts/classifiers/<kind>/<split>/ so evaluation
notebooks and batch_style_probs can load the matching weights.

Examples::

    py -3 classifiers/train_classifiers.py --split standard --models all
    py -3 classifiers/train_classifiers.py --split topic_hard --models logreg rnn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.splits import SPLIT_REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LogReg / RNN / Transformer sarcasm classifiers.")
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_REGISTRY),
        default="standard",
        help="Split strategy (must match test_systems*.ipynb SPLIT_NAME).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("logreg", "rnn", "transformer", "all"),
        default=["all"],
        help="Which models to train. Use 'all' alone, or list one or more of logreg, rnn, transformer.",
    )
    args = parser.parse_args()

    want = set(args.models)
    if "all" in want:
        if len(want) > 1:
            parser.error("When using 'all', do not specify other model names.")
        want = {"logreg", "rnn", "transformer"}
    else:
        want = {m for m in want if m != "all"}

    print(f"Split: {args.split}")
    print(f"Models: {sorted(want)}")
    print(f"Outputs: artifacts/classifiers/<kind>/{args.split}/\n")

    if "logreg" in want:
        from classifiers.logreg_classifier import train as train_logreg

        print("--- LogReg (TF-IDF) ---")
        train_logreg(split=args.split)
        print()

    if "rnn" in want:
        from classifiers.rnn_classifier import train as train_rnn

        print("--- RNN (BiLSTM) ---")
        train_rnn(split=args.split)
        print()

    if "transformer" in want:
        from classifiers.transformer_classifier import train as train_transformer

        print("--- Transformer ---")
        train_transformer(split=args.split)
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

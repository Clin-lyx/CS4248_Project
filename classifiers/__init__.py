"""Sarcasm-detection classifier pipelines.

Heavy dependencies (torch, transformers) load only when train entrypoints or
submodules that need them are imported — not on ``import classifiers``.
"""

from __future__ import annotations

from typing import Any, Callable

_lazy_train_fns: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]] | None = None


def _ensure_trains() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    global _lazy_train_fns
    if _lazy_train_fns is None:
        from classifiers.logreg_classifier import train as logreg_train
        from classifiers.rnn_classifier import train as rnn_train
        from classifiers.transformer_classifier import train as transformer_train

        _lazy_train_fns = (logreg_train, rnn_train, transformer_train)
    return _lazy_train_fns


def __getattr__(name: str) -> Any:
    if name == "logreg_train":
        return _ensure_trains()[0]
    if name == "rnn_train":
        return _ensure_trains()[1]
    if name == "transformer_train":
        return _ensure_trains()[2]
    if name == "CLASSIFIERS":
        lt, rt, tt = _ensure_trains()
        return {"logreg": lt, "rnn": rt, "transformer": tt}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CLASSIFIERS", "logreg_train", "rnn_train", "transformer_train"]

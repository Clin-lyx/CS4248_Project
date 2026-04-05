from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.decoding_configs import DEFAULT_CONFIG
from generation.prompts import build_prompt
from systems.system_b_utils import MODEL_DIR, build_seq2seq_input, normalize_text


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is required for System B inference. "
            "Install transformers and sentencepiece before running this module."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.eval()
    return tokenizer, model


def _decode_candidates(tokenizer, outputs) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()
        if not text:
            continue

        text = " ".join(text.split())
        text = text.rstrip(" .")
        text = text + "."

        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        candidates.append(text)

    return candidates


def _prompt_fallback_candidates(
    input_text: str,
    direction: str,
    k: int,
    model_name: str,
    decoding_config: dict | None,
) -> list[str]:
    tokenizer, model = _load_model_and_tokenizer(model_name)
    config = DEFAULT_CONFIG.copy()
    if decoding_config:
        config.update(decoding_config)

    prompt = build_prompt(input_text, direction)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    raw_k = max(k * 4, 12)
    outputs = model.generate(
        **inputs,
        num_return_sequences=raw_k,
        **config,
    )
    candidates = _decode_candidates(tokenizer, outputs)
    return candidates[:k] or [input_text]


def _finetuned_candidates(
    input_text: str,
    direction: str,
    k: int,
    model_dir: str | Path,
    decoding_config: dict | None,
) -> list[str]:
    tokenizer, model = _load_model_and_tokenizer(str(model_dir))
    config = DEFAULT_CONFIG.copy()
    if decoding_config:
        config.update(decoding_config)

    seq2seq_input = build_seq2seq_input(input_text, direction)
    inputs = tokenizer(
        seq2seq_input,
        return_tensors="pt",
        truncation=True,
        max_length=96,
    )

    outputs = model.generate(
        **inputs,
        num_return_sequences=max(k, 1),
        **config,
    )
    candidates = _decode_candidates(tokenizer, outputs)
    return candidates[:k] or [input_text]


def generate_candidates(
    input_text: str,
    direction: str,
    k: int = 5,
    decoding_config: dict | None = None,
    mode: str = "auto",
    finetuned_model_dir: str | Path = MODEL_DIR,
    prompt_fallback_model: str = "google/flan-t5-base",
) -> list[str]:
    resolved_model_dir = Path(finetuned_model_dir)

    if mode not in {"auto", "finetuned_local", "prompt_fallback"}:
        raise ValueError(f"Unknown System B mode: {mode}")

    if mode in {"auto", "finetuned_local"} and resolved_model_dir.exists():
        return _finetuned_candidates(
            input_text=input_text,
            direction=direction,
            k=k,
            model_dir=resolved_model_dir,
            decoding_config=decoding_config,
        )

    if mode == "finetuned_local" and not resolved_model_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned System B model not found at {resolved_model_dir}. "
            "Train it with systems/system_b_train.py first."
        )

    return _prompt_fallback_candidates(
        input_text=input_text,
        direction=direction,
        k=k,
        model_name=prompt_fallback_model,
        decoding_config=decoding_config,
    )


def rewrite_text(
    input_text: str,
    direction: str,
    mode: str = "auto",
    **kwargs,
) -> str:
    return generate_candidates(
        input_text=input_text,
        direction=direction,
        k=1,
        mode=mode,
        **kwargs,
    )[0]

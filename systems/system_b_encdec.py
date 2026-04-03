import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from generation.prompts import build_prompt
from generation.decoding_configs import DEFAULT_CONFIG

MODEL_NAME = "google/flan-t5-large"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


def _normalize(text):
    return " ".join(text.strip().lower().split())


def _looks_too_similar(candidate, original):
    cand = _normalize(candidate)
    orig = _normalize(original)

    if cand == orig:
        return True

    # if most of the original appears unchanged, reject it
    orig_words = set(orig.split())
    cand_words = set(cand.split())

    if len(orig_words) == 0:
        return False

    overlap = len(orig_words & cand_words) / len(orig_words)

    return overlap > 0.9


def generate_candidates(input_text, direction, k=5, decoding_config=None):
    if decoding_config is None:
        decoding_config = DEFAULT_CONFIG.copy()

    prompt = build_prompt(input_text, direction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k_: v.to(device) for k_, v in inputs.items()}

    raw_k = k * 4 if k <= 3 else max(k * 2, 12)

    outputs = model.generate(
        **inputs,
        num_return_sequences=raw_k,
        **decoding_config,
    )

    candidates = []
    seen = set()

    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()

        if not text:
            continue

        # clean repeated punctuation/whitespace
        text = " ".join(text.split())
        text = text.rstrip(" .")
        text = text + "."

        norm = _normalize(text)

        # remove duplicates
        if norm in seen:
            continue

        # remove unchanged / nearly unchanged outputs
        if _looks_too_similar(text, input_text):
            continue

        seen.add(norm)
        candidates.append(text)

        if len(candidates) >= k:
            break

    # fallback if filtering was too aggressive
    if len(candidates) == 0:
        candidates.append(input_text)

    return candidates
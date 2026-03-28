from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from generation.prompts import build_prompt
from generation.decoding_configs import DEFAULT_CONFIG

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def _normalize(text):
    return " ".join(text.strip().lower().split())


def generate_candidates(input_text, direction, k=5, decoding_config=None):
    if decoding_config is None:
        decoding_config = DEFAULT_CONFIG

    prompt = build_prompt(input_text, direction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    outputs = model.generate(
        **inputs,
        num_return_sequences=k,
        **decoding_config,
    )

    candidates = []
    seen = set()

    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()

        if not text:
            continue

        norm = _normalize(text)

        # remove unchanged output
        if norm == _normalize(input_text):
            continue

        # remove duplicates
        if norm in seen:
            continue

        seen.add(norm)
        candidates.append(text)

    return candidates
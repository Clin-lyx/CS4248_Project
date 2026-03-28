from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from generation.prompts import build_prompt
from generation.decoding_configs import DEFAULT_CONFIG

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_candidates(input_text, direction, k=1, decoding_config=None):
    if decoding_config is None:
        decoding_config = DEFAULT_CONFIG

    prompt = build_prompt(input_text, direction)

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        num_return_sequences=k,
        **decoding_config
    )

    candidates = []

    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        candidates.append(text)

    return candidates
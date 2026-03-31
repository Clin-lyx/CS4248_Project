import json
import re
from pathlib import Path
from typing import Any


_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates.json"


def load_templates() -> dict[str, Any]:
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_anchor_texts(anchor_dict: dict[str, Any] | None) -> list[str]:
    if not isinstance(anchor_dict, dict):
        return []
    return [
        a["text"]
        for a in anchor_dict.get("all", [])
        if isinstance(a, dict) and "text" in a
    ]


def preserves_anchors(anchor_dict: dict[str, Any] | None, rewritten_text: str) -> bool:
    anchor_texts = get_anchor_texts(anchor_dict)
    rewritten_lower = rewritten_text.lower()
    return all(anchor.lower() in rewritten_lower for anchor in anchor_texts)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_punctuation_spacing(text: str) -> str:
    text = re.sub(r"\s+([,:;.!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return normalize_space(text)


def starts_like_headline(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 0


def strip_prefixes(text: str, prefixes: list[str]) -> str:
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def strip_suffixes(text: str, suffixes: list[str]) -> str:
    lowered = text.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            return text[: len(text) - len(suffix)].strip()
    return text


def soften_words(text: str, replacements: dict[str, str]) -> str:
    out = text
    for word, repl in replacements.items():
        pattern = rf"\b{re.escape(word)}\b"
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return clean_punctuation_spacing(out)


def has_question_like_form(text: str) -> bool:
    lowered = text.lower()
    return "?" in text or lowered.startswith("how ") or lowered.startswith("why ") or lowered.startswith("what ")


def has_number_anchor(anchor_dict: dict[str, Any] | None) -> bool:
    if not isinstance(anchor_dict, dict):
        return False
    return any(a.get("type") == "number" for a in anchor_dict.get("all", []) if isinstance(a, dict))


def has_entity_anchor(anchor_dict: dict[str, Any] | None) -> bool:
    if not isinstance(anchor_dict, dict):
        return False
    return any(a.get("type") == "entity" for a in anchor_dict.get("all", []) if isinstance(a, dict))


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = clean_punctuation_spacing(item)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def neutral_to_sarcastic_candidates(text: str, anchor_dict: dict[str, Any] | None) -> list[str]:
    cfg = load_templates()
    body = normalize_space(text)
    lowered = body.lower()

    base_prefixes = cfg.get("neutral_to_sarcastic_prefixes", [])
    base_suffixes = cfg.get("neutral_to_sarcastic_suffixes", [])
    question_suffixes = cfg.get("neutral_to_sarcastic_question_suffixes", [])
    list_suffixes = cfg.get("neutral_to_sarcastic_list_suffixes", [])
    person_prefixes = cfg.get("neutral_to_sarcastic_person_prefixes", [])
    policy_prefixes = cfg.get("neutral_to_sarcastic_policy_prefixes", [])

    is_question_like = has_question_like_form(body)
    is_list_like = bool(re.match(r"^\d+\b", lowered)) or ":" in body
    has_number = has_number_anchor(anchor_dict)
    has_entity = has_entity_anchor(anchor_dict)

    political_words = [
        "trump", "obama", "congress", "senate", "white house",
        "policy", "border", "republican", "democrat", "leader"
    ]
    is_policy_like = any(word in lowered for word in political_words)

    candidates = []

    if is_question_like:
        candidates.extend(f"{body} {suffix}" for suffix in question_suffixes)
        candidates.extend(f"{prefix} {body}" for prefix in base_prefixes)

    elif is_list_like:
        candidates.extend(f"{body} {suffix}" for suffix in list_suffixes)
        candidates.extend(f"{prefix} {body}" for prefix in base_prefixes)

    elif is_policy_like:
        candidates.extend(f"{prefix} {body}" for prefix in policy_prefixes)
        candidates.extend(f"{body} {suffix}" for suffix in base_suffixes)

    elif has_entity:
        candidates.extend(f"{prefix} {body}" for prefix in person_prefixes)
        candidates.extend(f"{body} {suffix}" for suffix in base_suffixes)

    elif has_number:
        candidates.extend(f"{body} {suffix}" for suffix in base_suffixes)
        candidates.extend(f"{prefix} {body}" for prefix in base_prefixes)

    else:
        candidates.extend(f"{prefix} {body}" for prefix in base_prefixes)
        candidates.extend(f"{body} {suffix}" for suffix in base_suffixes)

    return dedupe_keep_order(candidates)

def sarcastic_to_neutral_candidates(text: str, anchor_dict: dict[str, Any] | None) -> list[str]:
    cfg = load_templates()
    body = normalize_space(text)

    c1 = strip_prefixes(body, cfg.get("sarcastic_prefixes_to_strip", []))
    c2 = soften_words(c1, cfg.get("sarcastic_words_to_soften", {}))
    c3 = strip_suffixes(c2, cfg.get("sarcastic_suffixes_to_strip", []))
    c3 = clean_punctuation_spacing(c3)

    candidates = [c1, c2, c3]
    return dedupe_keep_order(candidates)


def choose_safe_candidate(
    original_text: str,
    anchor_dict: dict[str, Any] | None,
    candidates: list[str],
) -> str:
    original_clean = clean_punctuation_spacing(original_text)

    safe_candidates = []
    for candidate in candidates:
        candidate = clean_punctuation_spacing(candidate)
        if starts_like_headline(candidate) and preserves_anchors(anchor_dict, candidate):
            safe_candidates.append(candidate)

    if not safe_candidates:
        return original_clean

    # Prefer candidates that actually changed the text
    changed = [c for c in safe_candidates if c != original_clean]
    if changed:
        return changed[0]

    return safe_candidates[0]
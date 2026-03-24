"""
Anchor extraction for sarcasm headline rewriting.

Anchors are spans (entities, numbers, capitalized tokens) that downstream
systems must preserve when rewriting a headline. Each anchor is a dict with
text, character offsets (start/end on the input string), and a type tag.
"""

import re
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Lazy spaCy loading
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


# ---------------------------------------------------------------------------
# Number regex patterns (ordered: most specific first)
# ---------------------------------------------------------------------------
MONEY_RE = re.compile(
    r"[$£€]\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|trillion|[MBTmbt])?\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\b")

_SPECIFIC_NUM_PATTERNS = [MONEY_RE, PERCENT_RE, ORDINAL_RE, YEAR_RE]

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}

_STOP_CAPS = frozenset({
    "A", "An", "The", "In", "On", "At", "To", "For", "Of", "And", "But",
    "Or", "Is", "It", "By", "As", "If", "So", "No", "Up", "He", "She",
    "We", "My", "Do", "Be", "I",
})


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def _make_span(text: str, start: int, end: int, anchor_type: str) -> dict:
    return {"text": text, "start": start, "end": end, "type": anchor_type}


def _overlaps_any(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(not (end <= ts or start >= te) for ts, te in taken)


def _dedup_spans(spans: list[dict]) -> list[dict]:
    """Remove duplicate spans by (start, end). Keep first occurrence."""
    seen: set[tuple[int, int]] = set()
    out = []
    for s in spans:
        key = (s["start"], s["end"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Per-category extractors
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> list[dict]:
    """
    Greedy-first: match money/percent/ordinal/year first, mask those
    character ranges, then run the general number pattern on the remainder.
    """
    spans: list[dict] = []
    taken: list[tuple[int, int]] = []

    for pat in _SPECIFIC_NUM_PATTERNS:
        for m in pat.finditer(text):
            if not _overlaps_any(m.start(), m.end(), taken):
                spans.append(_make_span(m.group(), m.start(), m.end(), "number"))
                taken.append((m.start(), m.end()))

    for m in NUMBER_RE.finditer(text):
        if not _overlaps_any(m.start(), m.end(), taken):
            spans.append(_make_span(m.group(), m.start(), m.end(), "number"))
            taken.append((m.start(), m.end()))

    spans.sort(key=lambda s: s["start"])
    return spans


def extract_entities(doc) -> list[dict]:
    spans = []
    for ent in doc.ents:
        if ent.label_ in NER_LABELS:
            spans.append(_make_span(ent.text, ent.start_char, ent.end_char, "entity"))
    return spans


def extract_capitalized(doc, taken: list[tuple[int, int]]) -> list[dict]:
    """
    Fallback: tokens that are titlecase or uppercase, aren't stopwords,
    and weren't already captured by NER or numbers.
    """
    spans = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.text in _STOP_CAPS:
            continue
        if not (tok.text[0].isupper() and len(tok.text) >= 2):
            continue
        if tok.i == 0:
            continue
        start, end = tok.idx, tok.idx + len(tok.text)
        if _overlaps_any(start, end, taken):
            continue
        spans.append(_make_span(tok.text, start, end, "capital"))
    return spans


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_anchors(text: str, doc=None) -> dict:
    """
    Extract all anchor spans from a headline string.

    Returns dict with keys: entities, numbers, capitals, all.
    Each value is a list of {text, start, end, type} dicts.
    """
    empty = {"entities": [], "numbers": [], "capitals": [], "all": []}
    if not isinstance(text, str) or not text.strip():
        return empty

    if doc is None:
        doc = _get_nlp()(text)

    numbers = extract_numbers(text)
    entities = extract_entities(doc)

    taken = [(s["start"], s["end"]) for s in numbers + entities]
    capitals = extract_capitalized(doc, taken)

    all_anchors = _dedup_spans(entities + numbers + capitals)
    all_anchors.sort(key=lambda s: s["start"])

    return {
        "entities": entities,
        "numbers": numbers,
        "capitals": capitals,
        "all": all_anchors,
    }


# ---------------------------------------------------------------------------
# Batch processing (uses nlp.pipe for speed)
# ---------------------------------------------------------------------------

def add_anchors(
    df: pd.DataFrame,
    text_col: str = "text",
    batch_size: int = 512,
) -> pd.DataFrame:
    """Add an 'anchors' column to df using batched spaCy processing."""
    df = df.copy()
    nlp = _get_nlp()
    texts = df[text_col].tolist()

    anchors_list: list[dict] = []
    for doc, text in zip(nlp.pipe(texts, batch_size=batch_size), texts):
        anchors_list.append(extract_anchors(text, doc=doc))

    df["anchors"] = anchors_list
    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_jsonl(
    df: pd.DataFrame,
    output_path: str | Path = "artifacts/data/cleaned_with_anchors.jsonl",
) -> Path:
    output_path = _PROJECT_ROOT / Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} rows -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI: anchorize cleaned.jsonl
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cleaned_path = _PROJECT_ROOT / "artifacts" / "data" / "cleaned.jsonl"
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found. Run preprocess.py first."
        )

    print(f"Loading {cleaned_path} ...")
    df = pd.read_json(cleaned_path, lines=True, encoding="utf-8")
    print(f"  {len(df)} rows loaded.")

    print("Extracting anchors (batched spaCy) ...")
    df = add_anchors(df)

    n_with = int(df["anchors"].apply(lambda a: len(a["all"]) > 0).sum())
    print(f"  {n_with}/{len(df)} rows have at least one anchor.")

    save_jsonl(df)

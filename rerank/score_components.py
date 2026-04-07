from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_a.template_utils import get_anchor_texts, preserves_anchors
from systems.system_b_utils import (
    finite_or_none,
    looks_too_similar,
    normalized_token_edit_distance,
    score_style_probability,
    semantic_similarity_score,
)


def retained_anchor_texts(anchor_dict: dict[str, Any] | None, candidate_text: str) -> list[str]:
    candidate_lower = str(candidate_text).lower()
    return [anchor for anchor in get_anchor_texts(anchor_dict) if anchor.lower() in candidate_lower]


def dropped_anchor_texts(anchor_dict: dict[str, Any] | None, candidate_text: str) -> list[str]:
    kept = set(retained_anchor_texts(anchor_dict, candidate_text))
    return [anchor for anchor in get_anchor_texts(anchor_dict) if anchor not in kept]


def anchor_retention_rate(anchor_dict: dict[str, Any] | None, candidate_text: str) -> float:
    anchors = get_anchor_texts(anchor_dict)
    if not anchors:
        return 1.0
    kept = retained_anchor_texts(anchor_dict, candidate_text)
    return float(len(kept) / max(len(anchors), 1))


def lexical_overlap_ratio(input_text: str, candidate_text: str) -> float:
    source_words = set(str(input_text).lower().split())
    target_words = set(str(candidate_text).lower().split())
    if not source_words:
        return 0.0
    return float(len(source_words & target_words) / len(source_words))


def target_style_score(direction: str, prob_sarcastic: float | None) -> float | None:
    if prob_sarcastic is None:
        return None
    if direction == "n2s":
        return float(prob_sarcastic)
    if direction == "s2n":
        return float(1.0 - prob_sarcastic)
    raise ValueError(f"Unknown direction: {direction}")


def validate_candidate(
    input_text: str,
    candidate_text: str,
    anchor_dict: dict[str, Any] | None,
    direction: str,
    prob_sarcastic: float | None,
    semantic_similarity: float | None,
    edit_distance: float | None,
    *,
    style_threshold: float = 0.5,
    semantic_threshold: float = 0.45,
    max_edit_distance: float | None = None,
    reject_identity: bool = True,
    reject_anchor_loss: bool = True,
    enforce_semantic_constraint: bool = True,
    enforce_style_threshold: bool = True,
) -> tuple[bool, str | None]:
    candidate = str(candidate_text or "").strip()
    if not candidate:
        return False, "empty"

    if reject_identity and candidate.lower() == str(input_text).strip().lower():
        return False, "identity"

    if reject_identity and looks_too_similar(candidate, input_text, max_overlap=0.97):
        return False, "too_similar_to_input"

    if reject_anchor_loss and not preserves_anchors(anchor_dict, candidate):
        return False, "anchor_loss"

    if enforce_style_threshold:
        target_score = target_style_score(direction, prob_sarcastic)
        if target_score is not None and target_score < style_threshold:
            return False, "wrong_target_style"

    if enforce_semantic_constraint and semantic_similarity is not None and semantic_similarity < semantic_threshold:
        return False, "low_semantic_similarity"

    if max_edit_distance is not None and edit_distance is not None and edit_distance > max_edit_distance:
        return False, "too_different"

    return True, None


def compute_candidate_scores(
    input_text: str,
    candidate_text: str,
    direction: str,
    anchor_dict: dict[str, Any] | None,
    *,
    style_scorer=None,
    style_threshold: float = 0.5,
    semantic_threshold: float = 0.45,
    max_edit_distance: float | None = None,
    reject_identity: bool = True,
    reject_anchor_loss: bool = True,
    enforce_semantic_constraint: bool = True,
    enforce_style_threshold: bool = True,
) -> dict[str, Any]:
    candidate_text = str(candidate_text or "").strip()

    prob = score_style_probability(candidate_text, scorer=style_scorer)
    style_score = target_style_score(direction, prob)

    semantic = None
    if candidate_text:
        try:
            semantic = semantic_similarity_score(input_text, candidate_text)
        except Exception:
            semantic = None

    edit_distance = None
    if candidate_text:
        try:
            edit_distance = normalized_token_edit_distance(input_text, candidate_text)
        except Exception:
            edit_distance = None

    anchor_ok = preserves_anchors(anchor_dict, candidate_text) if candidate_text else False
    anchor_rate = anchor_retention_rate(anchor_dict, candidate_text)
    retained = retained_anchor_texts(anchor_dict, candidate_text)
    dropped = dropped_anchor_texts(anchor_dict, candidate_text)
    lexical_overlap = lexical_overlap_ratio(input_text, candidate_text) if candidate_text else 0.0

    is_valid, rejection_reason = validate_candidate(
        input_text=input_text,
        candidate_text=candidate_text,
        anchor_dict=anchor_dict,
        direction=direction,
        prob_sarcastic=prob,
        semantic_similarity=semantic,
        edit_distance=edit_distance,
        style_threshold=style_threshold,
        semantic_threshold=semantic_threshold,
        max_edit_distance=max_edit_distance,
        reject_identity=reject_identity,
        reject_anchor_loss=reject_anchor_loss,
        enforce_semantic_constraint=enforce_semantic_constraint,
        enforce_style_threshold=enforce_style_threshold,
    )

    return {
        "style_prob_sarcastic": finite_or_none(prob),
        "target_style_score": finite_or_none(style_score),
        "semantic_similarity": finite_or_none(semantic),
        "edit_distance": finite_or_none(edit_distance),
        "anchor_retention_rate": finite_or_none(anchor_rate),
        "anchors_preserved": bool(anchor_ok),
        "lexical_overlap": finite_or_none(lexical_overlap),
        "retained_anchors": retained,
        "dropped_anchors": dropped,
        "is_valid": bool(is_valid),
        "rejection_reason": rejection_reason,
    }

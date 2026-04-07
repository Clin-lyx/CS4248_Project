from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rerank.score_components import compute_candidate_scores

DEFAULT_RERANK_CONFIG: dict[str, Any] = {
    "style_weight": 0.40,
    "semantic_weight": 0.35,
    "anchor_weight": 0.20,
    "edit_weight": 0.05,
    "style_threshold": 0.50,
    "semantic_threshold": 0.45,
    "max_edit_distance": None,
    "reject_identity": True,
    "reject_anchor_loss": True,
    "enforce_semantic_constraint": True,
    "enforce_style_threshold": True,
    "invalid_penalty": 1.0,
}


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_RERANK_CONFIG)
    if config:
        merged.update(config)
    return merged


def _final_score_from_components(scores: dict[str, Any], config: dict[str, Any]) -> float:
    style = float(scores.get("target_style_score") or 0.0)
    semantic = float(scores.get("semantic_similarity") or 0.0)
    anchor = float(scores.get("anchor_retention_rate") or 0.0)
    edit = float(scores.get("edit_distance") or 1.0)

    final_score = (
        config["style_weight"] * style
        + config["semantic_weight"] * semantic
        + config["anchor_weight"] * anchor
        - config["edit_weight"] * edit
    )
    if not scores.get("is_valid", False):
        final_score -= float(config.get("invalid_penalty", 1.0))
    return float(final_score)


def _coerce_candidate(candidate: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, dict):
        payload = dict(candidate)
        if "output_text" not in payload and "text" in payload:
            payload["output_text"] = payload["text"]
        payload.setdefault("metadata", {})
        return payload
    return {"output_text": str(candidate), "metadata": {}}


def score_and_rank_candidates(
    input_text: str,
    candidates: list[str | dict[str, Any]],
    direction: str,
    anchor_dict: dict[str, Any] | None,
    *,
    style_scorer=None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = _merge_config(config)
    ranked: list[dict[str, Any]] = []

    for candidate in candidates:
        item = _coerce_candidate(candidate)
        scores = compute_candidate_scores(
            input_text=input_text,
            candidate_text=item.get("output_text", ""),
            direction=direction,
            anchor_dict=anchor_dict,
            style_scorer=style_scorer,
            style_threshold=cfg["style_threshold"],
            semantic_threshold=cfg["semantic_threshold"],
            max_edit_distance=cfg["max_edit_distance"],
            reject_identity=cfg["reject_identity"],
            reject_anchor_loss=cfg["reject_anchor_loss"],
            enforce_semantic_constraint=cfg["enforce_semantic_constraint"],
            enforce_style_threshold=cfg["enforce_style_threshold"],
        )
        item["scores"] = scores
        item["final_score"] = _final_score_from_components(scores, cfg)
        ranked.append(item)

    ranked.sort(
        key=lambda row: (
            int(bool(row["scores"].get("is_valid", False))),
            float(row.get("final_score", -999.0)),
            float(row["scores"].get("semantic_similarity") or 0.0),
        ),
        reverse=True,
    )

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked


def select_best_candidate(ranked_candidates: list[dict[str, Any]], input_text: str) -> dict[str, Any]:
    if not ranked_candidates:
        return {
            "output_text": input_text,
            "scores": {
                "is_valid": False,
                "rejection_reason": "no_candidates",
            },
            "metadata": {"fallback_used": True, "fallback_reason": "no_candidates"},
            "final_score": -999.0,
            "rank": 1,
        }

    for item in ranked_candidates:
        if item["scores"].get("is_valid", False):
            item.setdefault("metadata", {})
            item["metadata"].setdefault("fallback_used", False)
            return item

    fallback = ranked_candidates[0]
    fallback.setdefault("metadata", {})
    fallback["metadata"]["fallback_used"] = True
    fallback["metadata"]["fallback_reason"] = "all_candidates_invalid"
    return fallback


def rerank_candidates(
    input_text: str,
    candidates: list[str | dict[str, Any]],
    direction: str,
    anchor_dict: dict[str, Any] | None,
    *,
    style_scorer=None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = score_and_rank_candidates(
        input_text=input_text,
        candidates=candidates,
        direction=direction,
        anchor_dict=anchor_dict,
        style_scorer=style_scorer,
        config=config,
    )
    best = select_best_candidate(ranked, input_text)
    return best, ranked

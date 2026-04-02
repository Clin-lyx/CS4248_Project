from __future__ import annotations

from typing import Any

from .template_utils import (
    choose_safe_candidate,
    neutral_to_sarcastic_candidates,
    sarcastic_to_neutral_candidates,
)


class SystemATemplate:
    """
    Rule-based baseline for sarcasm headline rewriting.

    Directions:
    - n2s: neutral -> sarcastic
    - s2n: sarcastic -> neutral
    """

    def get_candidates(
        self,
        text: str,
        anchor_dict: dict[str, Any] | None,
        direction: str,
    ) -> list[str]:
        if direction == "n2s":
            return neutral_to_sarcastic_candidates(text, anchor_dict)
        if direction == "s2n":
            return sarcastic_to_neutral_candidates(text, anchor_dict)
        raise ValueError(f"Unknown direction: {direction}")

    def rewrite(
        self,
        text: str,
        anchor_dict: dict[str, Any] | None,
        direction: str,
    ) -> str:
        candidates = self.get_candidates(text, anchor_dict, direction)
        return choose_safe_candidate(text, anchor_dict, candidates)

    def rewrite_row(self, row, direction: str) -> str:
        return self.rewrite(
            text=row["text"],
            anchor_dict=row.get("anchors", {}),
            direction=direction,
        )

    def rewrite_row_by_label(self, row) -> str:
        label = row["label"]
        if label == 0:
            return self.rewrite_row(row, direction="n2s")
        if label == 1:
            return self.rewrite_row(row, direction="s2n")
        raise ValueError(f"Unknown label: {label}")
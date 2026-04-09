from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from similarity.semantic_similarity import batch_semantic_similarity
from systems.system_b_utils import (
    MODEL_DIR,
    count_true,
    safe_mean,
    target_style_passes,
    load_anchored_data,
    select_subset,
)
from systems.system_a.template_utils import preserves_anchors
from systems.system_c_rer_pipeline import DEFAULT_PROMPT_FALLBACK_MODEL, run_system_c


def _batch_style_probabilities(texts: list[str], scorer) -> list[float | None]:
    if scorer is None:
        return [None] * len(texts)
    try:
        probs = scorer.predict_proba(texts)
        return [float(row[1]) for row in probs]
    except Exception:
        return [None] * len(texts)


def _load_best_rows_from_jsonl(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"System C best-output file not found: {resolved}")

    try:
        best_rows = pd.read_json(resolved, lines=True, encoding="utf-8")
    except ValueError as exc:
        raise ValueError(f"System C best-output file is not valid JSONL: {resolved}") from exc

    required = {"id", "direction", "input_text", "output_text"}
    missing = sorted(required - set(best_rows.columns))
    if missing:
        raise ValueError(
            f"System C best-output file is missing required fields {missing}: {resolved}"
        )
    return best_rows


def _make_eval_paths(
    split_name: str,
    split_strategy: str,
    *,
    no_retrieval: bool = False,
    no_rerank: bool = False,
    no_semantic_constraint: bool = False,
) -> tuple[Path, Path]:
    suffix = f"_{split_strategy}_{split_name}"
    tags: list[str] = []
    if no_retrieval:
        tags.append("no_retrieval")
    if no_rerank:
        tags.append("no_rerank")
    if no_semantic_constraint:
        tags.append("no_semantic_constraint")
    if tags:
        suffix += "_" + "_".join(tags)

    output_path = PROJECT_ROOT / "results" / f"system_c_outputs{suffix}.jsonl"
    metrics_path = PROJECT_ROOT / "results" / f"system_c_eval_metrics{suffix}.json"
    return output_path, metrics_path


def evaluate_system_c(
    split_name: str = "dev",
    split: str = "standard",
    limit: int = 0,
    output_path: str | Path | None = None,
    *,
    k_retrieve: int = 5,
    k_generate: int = 5,
    system_b_mode: str = "auto",
    finetuned_model_dir: str | Path | None = None,
    prompt_fallback_model: str | None = None,
    no_retrieval: bool = False,
    no_rerank: bool = False,
    no_semantic_constraint: bool = False,
    force_rebuild_index: bool = False,
    best_input: str | Path | None = None,
) -> tuple[Path, Path]:
    if best_input is not None:
        best_rows = _load_best_rows_from_jsonl(best_input)
    else:
        best_path, _, _ = run_system_c(
            split_name=split_name,
            split_strategy=split,
            limit=limit,
            k_retrieve=k_retrieve,
            k_generate=k_generate,
            system_b_mode=system_b_mode,
            finetuned_model_dir=finetuned_model_dir or MODEL_DIR,
            prompt_fallback_model=prompt_fallback_model or DEFAULT_PROMPT_FALLBACK_MODEL,
            no_retrieval=no_retrieval,
            no_rerank=no_rerank,
            no_semantic_constraint=no_semantic_constraint,
            force_rebuild_index=force_rebuild_index,
            tag=None,
        )
        best_rows = pd.read_json(best_path, lines=True, encoding="utf-8")
    source_df = load_anchored_data()
    subset = select_subset(source_df, split_name=split_name, split=split)[["id", "anchors"]]
    anchors_by_id = dict(zip(subset["id"], subset["anchors"]))
    style_scorer = None

    target_texts = best_rows["output_text"].astype(str).tolist()
    source_texts = best_rows["input_text"].astype(str).tolist()
    scores_present = "scores" in best_rows.columns
    needs_similarity = (not scores_present) or best_rows["scores"].apply(
        lambda v: not isinstance(v, dict) or v.get("semantic_similarity") is None
    ).any()
    needs_style = (not scores_present) or best_rows["scores"].apply(
        lambda v: not isinstance(v, dict) or v.get("style_prob_sarcastic") is None
    ).any()

    batched_similarities = batch_semantic_similarity(source_texts, target_texts).tolist() if needs_similarity else []
    if needs_style:
        from systems.system_b_utils import load_style_scorer

        style_scorer = load_style_scorer(split=split)
        batched_probs = _batch_style_probabilities(target_texts, style_scorer)
    else:
        batched_probs = []

    scored_rows: list[dict] = []
    similarities: list[float] = []
    anchor_flags: list[bool] = []
    style_flags: list[bool] = []

    for _, row in best_rows.iterrows():
        scores = row.get("scores", {}) or {}
        direction = row["direction"]
        target_text = row["output_text"]
        prob = scores.get("style_prob_sarcastic")
        if prob is None:
            prob = batched_probs[len(scored_rows)]

        similarity = scores.get("semantic_similarity")
        if similarity is None:
            similarity = batched_similarities[len(scored_rows)]

        anchor_raw = scores.get("anchors_preserved")
        if anchor_raw is None:
            anchor_dict = anchors_by_id.get(row["id"], {})
            anchors_ok = preserves_anchors(anchor_dict, target_text)
        else:
            anchors_ok = bool(anchor_raw)
        style_ok = target_style_passes(direction, prob, threshold=0.5)

        scored_rows.append(
            {
                "id": row["id"],
                "direction": direction,
                "source_text": row["input_text"],
                "target_text": target_text,
                "semantic_similarity": similarity,
                "anchors_preserved": anchors_ok,
                "prob_sarcastic_target": prob,
                "target_style_pass": style_ok,
            }
        )
        if similarity is not None:
            similarities.append(float(similarity))
        anchor_flags.append(anchors_ok)
        style_flags.append(style_ok)

    if output_path is None:
        output_path, metrics_path = _make_eval_paths(
            split_name,
            split,
            no_retrieval=no_retrieval,
            no_rerank=no_rerank,
            no_semantic_constraint=no_semantic_constraint,
        )
    else:
        output_path = Path(output_path)
        metrics_path = output_path.with_name(output_path.stem.replace("_outputs", "_metrics") + ".json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored_rows).to_json(output_path, orient="records", lines=True, force_ascii=False)

    summary = {
        "split_name": split_name,
        "split_strategy": split,
        "num_examples": len(scored_rows),
        "avg_semantic_similarity": safe_mean(similarities),
        "anchor_preservation_rate": count_true(anchor_flags) / max(len(anchor_flags), 1),
        "target_style_pass_rate": count_true(style_flags) / max(len(style_flags), 1),
        "no_retrieval": no_retrieval,
        "no_rerank": no_rerank,
        "no_semantic_constraint": no_semantic_constraint,
        "system_b_mode": None if best_input is not None else system_b_mode,
        "k_retrieve": None if best_input is not None else k_retrieve,
        "k_generate": None if best_input is not None else k_generate,
        "best_input": str(best_input) if best_input is not None else None,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved outputs -> {output_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(json.dumps(summary, indent=2))
    return output_path, metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate System C outputs with System B-style metrics.")
    parser.add_argument("--split-name", default="dev")
    parser.add_argument("--split-strategy", default="standard", choices=["standard", "topic_hard"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--k-retrieve", type=int, default=5)
    parser.add_argument("--k-generate", type=int, default=5)
    parser.add_argument("--system-b-mode", default="auto", choices=["auto", "finetuned_local", "prompt_fallback"])
    parser.add_argument("--finetuned-model-dir", default=None)
    parser.add_argument("--prompt-fallback-model", default=None)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-semantic-constraint", action="store_true")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--best-input", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_system_c(
        split_name=args.split_name,
        split=args.split_strategy,
        limit=args.limit,
        output_path=args.output,
        k_retrieve=args.k_retrieve,
        k_generate=args.k_generate,
        system_b_mode=args.system_b_mode,
        finetuned_model_dir=args.finetuned_model_dir,
        prompt_fallback_model=args.prompt_fallback_model,
        no_retrieval=args.no_retrieval,
        no_rerank=args.no_rerank,
        no_semantic_constraint=args.no_semantic_constraint,
        force_rebuild_index=args.force_rebuild_index,
        best_input=args.best_input,
    )


if __name__ == "__main__":
    main()

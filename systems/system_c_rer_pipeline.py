from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.prompts import build_prompt
from rerank.rerank import DEFAULT_RERANK_CONFIG, rerank_candidates
from retrieval.build_index import build_retrieval_index
from retrieval.retrieve import retrieve_neighbors
from systems.system_a.template_utils import (
    neutral_to_sarcastic_candidates,
    sarcastic_to_neutral_candidates,
)
from systems.system_b_encdec import batch_generate, generate_candidates
from systems.system_b_utils import (
    MODEL_DIR,
    infer_direction,
    load_anchored_data,
    load_style_scorer,
    normalize_text,
    select_subset,
    write_jsonl,
)

SYSTEM_NAME = "system_c_rer"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_PROMPT_FALLBACK_MODEL = "google/flan-t5-base"

DECODE_CONFIG_BANK: dict[str, dict[str, Any]] = {
    "beam_like": {
        "max_new_tokens": 24,
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
    },
    "low_temp": {
        "max_new_tokens": 24,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
    },
    "diverse_sample": {
        "max_new_tokens": 24,
        "do_sample": True,
        "temperature": 1.3,
        "top_p": 0.95,
        "top_k": 50,
    },
}


def _load_prompt_model(model_name: str = DEFAULT_PROMPT_FALLBACK_MODEL):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _decode_unique(tokenizer, outputs) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()
        text = " ".join(text.split()).rstrip(" .")
        if not text:
            continue
        text = text + "."
        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        candidates.append(text)
    return candidates


def _build_retrieval_conditioned_prompt(input_text: str, direction: str, prototypes: list[str]) -> str:
    base = build_prompt(input_text, direction)
    if not prototypes:
        return base
    bullets = "\n".join(f"- {p}" for p in prototypes[:3])
    extra = (
        "\n\nTarget-style prototype headlines to imitate only for tone and headline shape "
        "(do not copy their topic or facts):\n"
        f"{bullets}\n\n"
        "Now rewrite the input headline while preserving the input topic, entities, dates and numbers."
    )
    return base + extra


def _generate_prompt_conditioned_candidates(
    input_text: str,
    direction: str,
    prototypes: list[str],
    k: int,
    model_name: str,
    decoding_config: dict[str, Any],
) -> list[str]:
    tokenizer, model = _load_prompt_model(model_name)
    prompt = _build_retrieval_conditioned_prompt(input_text, direction, prototypes)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    raw_k = max(k * 3, 8)
    outputs = model.generate(**inputs, num_return_sequences=raw_k, **decoding_config)
    return _decode_unique(tokenizer, outputs)[:k]


SARCASTIC_MARKERS = [
    "heroically",
    "finally",
    "bravely",
    "somehow",
    "miraculously",
    "absolutely",
    "once and for all",
]


def _retrieval_edit_candidates(
    input_text: str,
    direction: str,
    prototypes: list[dict[str, Any]],
    anchor_dict: dict[str, Any] | None = None,
) -> list[str]:
    candidates: list[str] = []

    # Always include conservative edit candidates from System A's rule library so
    # System C remains usable even when System B generation is unavailable.
    try:
        if direction == "n2s":
            candidates.extend(neutral_to_sarcastic_candidates(input_text, anchor_dict))
        else:
            candidates.extend(sarcastic_to_neutral_candidates(input_text, anchor_dict))
    except Exception:
        pass

    if direction == "n2s":
        base = input_text.rstrip(".")
        for marker in SARCASTIC_MARKERS[: min(3, len(prototypes) or 1)]:
            candidates.append(f"{base} {marker}.")
        # Retrieval-guided deadpan variants that preserve the original topic.
        candidates.append(f"Report: {base} for some reason.")
        candidates.append(f"Sources confirm {base}." )
        return candidates

    # For sarcastic->neutral, do not copy retrieved headlines wholesale. Instead,
    # create light neutralizations of the source to preserve topic/meaning.
    neutralized = input_text.rstrip(".")
    candidates.append(neutralized + ".")
    candidates.append(neutralized.replace("?", "").replace("!", "") + ".")
    return candidates


def build_candidate_pool(
    input_text: str,
    direction: str,
    retrieved: list[dict[str, Any]],
    *,
    anchor_dict: dict[str, Any] | None = None,
    k_generate: int = 5,
    system_b_mode: str = "auto",
    finetuned_model_dir: str | Path = MODEL_DIR,
    prompt_fallback_model: str = DEFAULT_PROMPT_FALLBACK_MODEL,
) -> list[dict[str, Any]]:
    all_candidates: list[dict[str, Any]] = []
    seen_norms: set[str] = set()
    retrieved_texts = [r["text"] for r in retrieved]
    retrieved_ids = [r["id"] for r in retrieved]

    for cfg_name, cfg in DECODE_CONFIG_BANK.items():
        try:
            outputs = generate_candidates(
                input_text=input_text,
                direction=direction,
                k=k_generate,
                decoding_config=cfg,
                mode=system_b_mode,
                finetuned_model_dir=finetuned_model_dir,
                prompt_fallback_model=prompt_fallback_model,
            )
        except Exception:
            outputs = []

        for text in outputs:
            norm = normalize_text(text)
            if not norm or norm in seen_norms:
                continue
            seen_norms.add(norm)
            all_candidates.append(
                {
                    "output_text": text,
                    "candidate_source": "generator_only",
                    "metadata": {
                        "candidate_source": "generator_only",
                        "decoding_config_name": cfg_name,
                        "retrieved_ids": retrieved_ids,
                        "retrieved_texts": retrieved_texts,
                    },
                }
            )

        if retrieved_texts:
            try:
                prompt_outputs = _generate_prompt_conditioned_candidates(
                    input_text=input_text,
                    direction=direction,
                    prototypes=retrieved_texts,
                    k=max(2, min(3, k_generate)),
                    model_name=prompt_fallback_model,
                    decoding_config=cfg,
                )
            except Exception:
                prompt_outputs = []

            for text in prompt_outputs:
                norm = normalize_text(text)
                if not norm or norm in seen_norms:
                    continue
                seen_norms.add(norm)
                all_candidates.append(
                    {
                        "output_text": text,
                        "candidate_source": "generator_with_prototypes",
                        "metadata": {
                            "candidate_source": "generator_with_prototypes",
                            "decoding_config_name": cfg_name,
                            "retrieved_ids": retrieved_ids,
                            "retrieved_texts": retrieved_texts,
                        },
                    }
                )

    for text in _retrieval_edit_candidates(input_text, direction, retrieved, anchor_dict=anchor_dict):
        norm = normalize_text(text)
        if not norm or norm in seen_norms:
            continue
        seen_norms.add(norm)
        all_candidates.append(
            {
                "output_text": text,
                "candidate_source": "retrieval_edit",
                "metadata": {
                    "candidate_source": "retrieval_edit",
                    "decoding_config_name": None,
                    "retrieved_ids": retrieved_ids,
                    "retrieved_texts": retrieved_texts,
                },
            }
        )

    if not all_candidates:
        all_candidates.append(
            {
                "output_text": input_text,
                "candidate_source": "fallback_identity",
                "metadata": {
                    "candidate_source": "fallback_identity",
                    "decoding_config_name": None,
                    "retrieved_ids": retrieved_ids,
                    "retrieved_texts": retrieved_texts,
                },
            }
        )

    return all_candidates


def build_candidate_pools_no_retrieval(
    rows: list[dict[str, Any]],
    *,
    k_generate: int,
    system_b_mode: str,
    finetuned_model_dir: str | Path,
    prompt_fallback_model: str,
    batch_size: int = 16,
) -> list[list[dict[str, Any]]]:
    texts = [row["text"] for row in rows]
    directions = [infer_direction(int(row["label"])) for row in rows]

    all_candidate_pools = []
    seen_norms_per_row: list[set[str]] = [set() for _ in rows]
    for _ in rows:
        all_candidate_pools.append([])

    for cfg_name, cfg in DECODE_CONFIG_BANK.items():
        try:
            batched_outputs = batch_generate(
                texts=texts,
                directions=directions,
                k=k_generate,
                decoding_config=cfg,
                batch_size=batch_size,
                mode=system_b_mode,
                finetuned_model_dir=finetuned_model_dir,
                prompt_fallback_model=prompt_fallback_model,
            )
        except Exception:
            batched_outputs = [[] for _ in rows]

        for idx, outputs in enumerate(batched_outputs):
            seen_norms = seen_norms_per_row[idx]
            for text in outputs:
                norm = normalize_text(text)
                if not norm or norm in seen_norms:
                    continue
                seen_norms.add(norm)
                all_candidate_pools[idx].append(
                    {
                        "output_text": text,
                        "candidate_source": "generator_only",
                        "metadata": {
                            "candidate_source": "generator_only",
                            "decoding_config_name": cfg_name,
                            "retrieved_ids": [],
                            "retrieved_texts": [],
                        },
                    }
                )

    for idx, row in enumerate(rows):
        direction = directions[idx]
        anchor_dict = row.get("anchors", {})
        seen_norms = seen_norms_per_row[idx]
        edit_candidates = _retrieval_edit_candidates(
            row["text"],
            direction,
            [],
            anchor_dict=anchor_dict,
        )
        for text in edit_candidates:
            norm = normalize_text(text)
            if not norm or norm in seen_norms:
                continue
            seen_norms.add(norm)
            all_candidate_pools[idx].append(
                {
                    "output_text": text,
                    "candidate_source": "retrieval_edit",
                    "metadata": {
                        "candidate_source": "retrieval_edit",
                        "decoding_config_name": None,
                        "retrieved_ids": [],
                        "retrieved_texts": [],
                    },
                }
            )

        if not all_candidate_pools[idx]:
            all_candidate_pools[idx].append(
                {
                    "output_text": row["text"],
                    "candidate_source": "fallback_identity",
                    "metadata": {
                        "candidate_source": "fallback_identity",
                        "decoding_config_name": None,
                        "retrieved_ids": [],
                        "retrieved_texts": [],
                    },
                }
            )

    return all_candidate_pools


def _make_results_paths(
    split_name: str,
    split_strategy: str,
    tag: str | None = None,
    *,
    no_retrieval: bool = False,
    no_rerank: bool = False,
    no_semantic_constraint: bool = False,
) -> tuple[Path, Path, Path]:
    suffix = f"_{split_strategy}_{split_name}"
    auto_tags: list[str] = []
    if no_retrieval:
        auto_tags.append("no_retrieval")
    if no_rerank:
        auto_tags.append("no_rerank")
    if no_semantic_constraint:
        auto_tags.append("no_semantic_constraint")
    if tag:
        auto_tags.append(tag)
    if auto_tags:
        suffix += "_" + "_".join(auto_tags)
    best_path = RESULTS_DIR / f"system_c_best{suffix}.jsonl"
    cand_path = RESULTS_DIR / f"system_c_candidates{suffix}.jsonl"
    metrics_path = RESULTS_DIR / f"system_c_metrics{suffix}.json"
    return best_path, cand_path, metrics_path


def run_system_c(
    *,
    split_name: str = "dev",
    split_strategy: str = "standard",
    limit: int = 0,
    k_retrieve: int = 5,
    k_generate: int = 5,
    system_b_mode: str = "auto",
    finetuned_model_dir: str | Path = MODEL_DIR,
    prompt_fallback_model: str = DEFAULT_PROMPT_FALLBACK_MODEL,
    no_retrieval: bool = False,
    no_rerank: bool = False,
    no_semantic_constraint: bool = False,
    force_rebuild_index: bool = False,
    tag: str | None = None,
) -> tuple[Path, Path, Path]:
    if not no_retrieval:
        build_retrieval_index(split=split_strategy, force=force_rebuild_index)

    df = load_anchored_data()
    subset = select_subset(df, split_name=split_name, split=split_strategy)
    if limit and limit > 0:
        subset = subset.head(limit).reset_index(drop=True)
    subset_rows = subset.to_dict("records")

    style_scorer = load_style_scorer(split=split_strategy)
    rerank_config = dict(DEFAULT_RERANK_CONFIG)
    if no_semantic_constraint:
        rerank_config["enforce_semantic_constraint"] = False

    best_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    prebuilt_candidate_pools = None
    if no_retrieval:
        prebuilt_candidate_pools = build_candidate_pools_no_retrieval(
            subset_rows,
            k_generate=k_generate,
            system_b_mode=system_b_mode,
            finetuned_model_dir=finetuned_model_dir,
            prompt_fallback_model=prompt_fallback_model,
        )

    for row_idx, row in enumerate(subset_rows):
        input_text = row["text"]
        direction = infer_direction(int(row["label"]))
        anchor_dict = row.get("anchors", {})
        retrieved = [] if no_retrieval else retrieve_neighbors(
            query_text=input_text,
            direction=direction,
            k=k_retrieve,
            split=split_strategy,
            exclude_ids={row["id"]},
        )

        if no_retrieval and prebuilt_candidate_pools is not None:
            candidates = prebuilt_candidate_pools[row_idx]
        else:
            candidates = build_candidate_pool(
                input_text=input_text,
                direction=direction,
                retrieved=retrieved,
                anchor_dict=anchor_dict,
                k_generate=k_generate,
                system_b_mode=system_b_mode,
                finetuned_model_dir=finetuned_model_dir,
                prompt_fallback_model=prompt_fallback_model,
            )

        if no_rerank:
            ranked = []
            for i, cand in enumerate(candidates, start=1):
                record = dict(cand)
                record["rank"] = i
                record["scores"] = {"is_valid": True, "rejection_reason": None}
                record["final_score"] = 0.0
                ranked.append(record)
            best = ranked[0]
        else:
            best, ranked = rerank_candidates(
                input_text=input_text,
                candidates=candidates,
                direction=direction,
                anchor_dict=anchor_dict,
                style_scorer=style_scorer,
                config=rerank_config,
            )

        best_rows.append(
            {
                "id": row["id"],
                "direction": direction,
                "input_text": input_text,
                "output_text": best["output_text"],
                "system_name": SYSTEM_NAME,
                "scores": best.get("scores", {}),
                "metadata": {
                    **best.get("metadata", {}),
                    "candidate_rank": best.get("rank"),
                    "final_score": best.get("final_score"),
                    "split_name": split_name,
                    "split_strategy": split_strategy,
                    "k_retrieve": k_retrieve,
                    "k_generate": k_generate,
                    "system_b_mode": system_b_mode,
                    "no_retrieval": no_retrieval,
                    "no_rerank": no_rerank,
                    "no_semantic_constraint": no_semantic_constraint,
                    "retrieved_ids": [r["id"] for r in retrieved],
                    "retrieved_texts": [r["text"] for r in retrieved],
                },
            }
        )

        for item in ranked:
            candidate_rows.append(
                {
                    "id": row["id"],
                    "direction": direction,
                    "input_text": input_text,
                    "output_text": item["output_text"],
                    "system_name": SYSTEM_NAME,
                    "scores": item.get("scores", {}),
                    "metadata": {
                        **item.get("metadata", {}),
                        "candidate_rank": item.get("rank"),
                        "final_score": item.get("final_score"),
                        "split_name": split_name,
                        "split_strategy": split_strategy,
                        "k_retrieve": k_retrieve,
                        "k_generate": k_generate,
                        "system_b_mode": system_b_mode,
                        "no_retrieval": no_retrieval,
                        "no_rerank": no_rerank,
                        "no_semantic_constraint": no_semantic_constraint,
                    },
                }
            )

    best_path, cand_path, metrics_path = _make_results_paths(
        split_name,
        split_strategy,
        tag=tag,
        no_retrieval=no_retrieval,
        no_rerank=no_rerank,
        no_semantic_constraint=no_semantic_constraint,
    )
    write_jsonl(best_rows, best_path)
    write_jsonl(candidate_rows, cand_path)

    valid_best = sum(1 for row in best_rows if row["scores"].get("is_valid", False))
    metrics = {
        "split_name": split_name,
        "split_strategy": split_strategy,
        "num_examples": len(best_rows),
        "valid_best_rate": valid_best / max(len(best_rows), 1),
        "no_retrieval": no_retrieval,
        "no_rerank": no_rerank,
        "no_semantic_constraint": no_semantic_constraint,
        "system_b_mode": system_b_mode,
        "k_retrieve": k_retrieve,
        "k_generate": k_generate,
        "rerank_config": rerank_config,
        "decode_config_bank": DECODE_CONFIG_BANK,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Also maintain canonical handoff filenames expected by teammates / report code.
    canonical_best = RESULTS_DIR / "system_c_best.jsonl"
    canonical_cand = RESULTS_DIR / "system_c_candidates.jsonl"
    canonical_cfg = RESULTS_DIR / "system_c_final_config.json"
    write_jsonl(best_rows, canonical_best)
    write_jsonl(candidate_rows, canonical_cand)
    canonical_cfg.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved canonical best outputs -> {canonical_best}")
    print(f"Saved canonical candidate logs -> {canonical_cand}")
    print(f"Saved canonical config -> {canonical_cfg}")
    print(f"Saved best outputs -> {best_path}")
    print(f"Saved candidate logs -> {cand_path}")
    print(f"Saved metrics -> {metrics_path}")
    return best_path, cand_path, metrics_path


def write_retrieval_sanity_file(
    *,
    split_name: str = "dev",
    split_strategy: str = "standard",
    limit: int = 200,
    k_retrieve: int = 5,
    force_rebuild_index: bool = False,
) -> Path:
    build_retrieval_index(split=split_strategy, force=force_rebuild_index)
    df = load_anchored_data()
    subset = select_subset(df, split_name=split_name, split=split_strategy).head(limit).reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for _, row in subset.iterrows():
        direction = infer_direction(int(row["label"]))
        neighbors = retrieve_neighbors(
            query_text=row["text"],
            direction=direction,
            k=k_retrieve,
            split=split_strategy,
            exclude_ids={row["id"]},
        )
        rows.append(
            {
                "id": row["id"],
                "direction": direction,
                "input_text": row["text"],
                "retrieved": neighbors,
            }
        )

    output_path = ANALYSIS_DIR / f"system_c_retrieval_sanity_{split_strategy}_{split_name}_{limit}.jsonl"
    write_jsonl(rows, output_path)
    print(f"Saved retrieval sanity file -> {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run System C retrieve-edit-rerank pipeline.")
    parser.add_argument("--split-name", default="dev")
    parser.add_argument("--split-strategy", default="standard", choices=["standard", "topic_hard"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--k-retrieve", type=int, default=5)
    parser.add_argument("--k-generate", type=int, default=5)
    parser.add_argument("--system-b-mode", default="auto", choices=["auto", "finetuned_local", "prompt_fallback"])
    parser.add_argument("--finetuned-model-dir", default=str(MODEL_DIR))
    parser.add_argument("--prompt-fallback-model", default=DEFAULT_PROMPT_FALLBACK_MODEL)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-semantic-constraint", action="store_true")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--write-retrieval-sanity", action="store_true")
    parser.add_argument("--sanity-limit", type=int, default=200)
    parser.add_argument("--tag", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.write_retrieval_sanity:
        write_retrieval_sanity_file(
            split_name=args.split_name,
            split_strategy=args.split_strategy,
            limit=args.sanity_limit,
            k_retrieve=args.k_retrieve,
            force_rebuild_index=args.force_rebuild_index,
        )
        return

    run_system_c(
        split_name=args.split_name,
        split_strategy=args.split_strategy,
        limit=args.limit,
        k_retrieve=args.k_retrieve,
        k_generate=args.k_generate,
        system_b_mode=args.system_b_mode,
        finetuned_model_dir=args.finetuned_model_dir,
        prompt_fallback_model=args.prompt_fallback_model,
        no_retrieval=args.no_retrieval,
        no_rerank=args.no_rerank,
        no_semantic_constraint=args.no_semantic_constraint,
        force_rebuild_index=args.force_rebuild_index,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()

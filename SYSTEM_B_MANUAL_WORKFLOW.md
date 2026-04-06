# System B Manual Pseudo-Pair Workflow

This is the active workflow for adding System B pseudo-pairs.

The old API/bootstrap script at `systems/system_b_pseudo_pairs.py` is legacy reference code only. Do not use it for the current team process.

## Active Files

- `artifacts/data/cleaned_with_anchors.jsonl`
  - Source headlines, labels, and anchors.
- `artifacts/splits/standard.json`
  - Use this to keep work on the `train` split only.
- `artifacts/system_b/pseudo_pairs_filtered.jsonl`
  - Final manual pseudo-pair dataset. Append accepted rows here.
- `systems/system_b_utils.py`
  - Helpers for `semantic_similarity` and `edit_ratio`.
- `systems/system_a/template_utils.py`
  - Use `preserves_anchors(...)` to check anchor preservation.
- `systems/check_manual_pseudo_pairs.py`
  - Validation helper for the manual dataset.

## Manual Workflow

1. Choose unused IDs from the `train` split only.
2. Infer direction from the label:
   - `0 -> n2s`
   - `1 -> s2n`
3. Write one target headline manually.
4. Preserve anchors such as names, places, organizations, and numbers.
5. Do not invent new facts.
6. Keep the output headline-length and close to the source meaning.
7. Compute metadata:
   - `semantic_similarity`
   - `edit_ratio`
   - `anchors_preserved`
8. Append rows to `artifacts/system_b/pseudo_pairs_filtered.jsonl`.
9. Run a cleanup pass on weak rows before merging.

## Required Row Schema

Each JSONL row should contain:

- `id`
- `source_text`
- `source_label`
- `direction`
- `target_text`
- `anchors`
- `generator_model`
- `prompt_version`
- `original_split`
- `semantic_similarity`
- `edit_ratio`
- `style_score_source`
- `style_score_target`
- `anchors_preserved`
- `accepted`
- `rejection_reason`

Defaults for the current manual workflow:

- `generator_model = "chatgpt_manual_gpt5_4"`
- `original_split = "train"`
- `accepted = true`
- `rejection_reason = null`
- `style_score_source = null`
- `style_score_target = null`

## Batch Naming

Use this format for `prompt_version`:

- `manual_batch_<NNN>_ids_<start>_<end>_<note>`

Example:

- `manual_batch_010_ids_561_821_balanced_600`

## Collaboration Rules

- Split work by non-overlapping ID ranges.
- Only use `train` examples.
- Keep one batch name per teammate batch.
- Run validation before merging.

Recommended teammate split style:

- Person A: one ID range
- Person B: next non-overlapping ID range
- Person C: next non-overlapping ID range

## Acceptability Checklist

- Anchors preserved
- No invented facts
- Headline length preserved
- `s2n` removes irony without changing the event
- `n2s` adds sarcasm without changing the core claim

## Validation

Run:

```powershell
.venv\Scripts\python.exe systems\check_manual_pseudo_pairs.py
```

This checks:

- JSONL loads cleanly
- required fields exist
- IDs are unique
- rows are from the `train` split only
- anchors are preserved
- counts by direction
- counts by `prompt_version`

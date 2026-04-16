# CS4248 Project — Sarcasm Style Transfer in News Headlines

## Table of Contents

- [Overview](#overview)
- [Task](#task)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Core Systems](#core-systems)
  - [System A — Template Baseline](#system-a--template-baseline)
  - [System B — Fine-tuned Encoder–Decoder](#system-b--fine-tuned-encoderdecoder)
  - [System C — Retrieve–Edit–Rerank](#system-c--retrieveeditrerank)
- [Setup](#setup)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
  - [Train Classifiers](#train-classifiers)
  - [Train System B](#train-system-b)
- [Evaluation](#evaluation)
  - [Evaluate System B](#evaluate-system-b)
  - [Build Retrieval Index](#build-retrieval-index)
  - [Run System C](#run-system-c)
  - [Evaluate System C](#evaluate-system-c)
- [Standard vs Topic-Hard Workflows](#standard-vs-topic-hard-workflows)
- [Important Artifacts](#important-artifacts)
- [Notebooks](#notebooks)
- [Notes and Limitations](#notes-and-limitations)

---

## Overview

This repository implements **bidirectional sarcasm style transfer** for news headlines.

Given a headline and its label, the task is to rewrite it into the **opposite style** while preserving the original topic and key factual content. The project compares three systems:

- **System A**: rule-based template baseline
- **System B**: fine-tuned Flan-T5 encoder–decoder
- **System C**: retrieve–edit–rerank pipeline built on top of System B

In the final report, **System C without retrieval** gives the best overall balance between style transfer and content preservation.

---

## Task

**Input:** a news headline with its original sarcasm label

**Output:** a rewritten headline that preserves the original topic and core meaning while reversing the sarcasm style

Directions:

- `n2s`: neutral → sarcastic
- `s2n`: sarcastic → neutral

The main evaluation dimensions are:

- style correctness
- content preservation
- fluency / faithfulness

---

## Dataset

We use the **Sarcasm Headlines Dataset v2**.

Each example contains:

- `headline`
- `is_sarcastic`
- `article_link`

After light normalization and duplicate removal, the cleaned dataset contains **28,503** examples.

The repository supports two split strategies:

- **`standard`** — stratified train/dev/test split
- **`topic_hard`** — topic-disjoint split to test generalization beyond topical cues

Main data artifacts:

- `artifacts/data/cleaned.jsonl`
- `artifacts/data/cleaned_with_anchors.jsonl`
- `artifacts/splits/standard.json`
- `artifacts/splits/topic_hard.json`

---

## Repository Structure

```text
.
├── CS4248_Project_standard.ipynb
├── CS4248_Project_topic_hard.ipynb
├── DATA.md
├── SYSTEM_B_MANUAL_WORKFLOW.md
├── analysis/
├── artifacts/
│   ├── classifiers/
│   ├── data/
│   ├── human_eval/
│   ├── splits/
│   └── system_b/
├── classifiers/
├── data/
├── dataset/
├── experiments/
├── generation/
├── notebooks/
├── rerank/
├── retrieval/
├── scripts/
├── similarity/
├── systems/
└── requirements.txt
```

### Main folders

- `data/` — preprocessing, anchors, standard split, topic-hard split
- `dataset/` — raw dataset files
- `classifiers/` — split-aware sarcasm detectors
- `systems/` — Systems A, B, and C
- `retrieval/` — retrieval index and nearest-neighbor search
- `rerank/` — candidate scoring and reranking
- `similarity/` — semantic similarity utilities
- `artifacts/` — processed data, splits, trained models, pseudo-pairs, outputs
- `notebooks/` — analysis and support notebooks
- root notebooks — end-to-end standard and topic-hard workflows

---

## Core Systems

### System A — Template Baseline

System A is a rule-based baseline built from templates and simple neutralization rules.

It is conservative and usually preserves content well, but often fails to flip style strongly enough.

Relevant files:

- `systems/system_a/system_a_template.py`
- `systems/system_a/template_utils.py`
- `systems/system_a/templates.json`

<br>

### System B — Fine-tuned Encoder–Decoder

System B fine-tunes `google/flan-t5-base` on pseudo-parallel rewrite pairs.

The current training script is **file-based** and reads pseudo-pairs from a JSONL file. It saves the model and `train_metrics.json`.

Relevant files:

- `systems/system_b_train.py`
- `systems/system_b_encdec.py`
- `systems/system_b_evaluate.py`
- `systems/system_b_utils.py`

Generation modes supported by System B:

- `auto`
- `finetuned_local`
- `prompt_fallback`

<br>

### System C — Retrieve–Edit–Rerank

System C builds a candidate pool from:

- multiple decoding strategies
- optional retrieval prototypes
- conservative edit candidates

It then reranks them using:

- style score
- semantic similarity
- anchor preservation
- edit control

Relevant files:

- `systems/system_c_rer_pipeline.py`
- `systems/system_c_evaluate.py`
- `retrieval/build_index.py`
- `retrieval/retrieve.py`
- `rerank/rerank.py`
- `rerank/score_components.py`

---

## Setup

Create and activate a virtual environment, then install dependencies.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Data Pipeline

Build the processed dataset, anchors, and splits from the repository root:

```bash
python -m data.preprocess
python -m data.anchors
python -m data.splits
python -m data.topic_hard_split
```

This produces the main data artifacts used throughout the project.

## Training

### Train Classifiers

Train the split-aware sarcasm detectors used for evaluation and reranking.

```bash
python classifiers/train_classifiers.py --split standard --models all
python classifiers/train_classifiers.py --split topic_hard --models all
```

Classifier artifacts are stored under:

- `artifacts/classifiers/logreg/`
- `artifacts/classifiers/rnn/`
- `artifacts/classifiers/transformer/`

### Train System B

The current `system_b_train.py` script is **file-based**, not split-flag based.

Example:

```bash
python systems/system_b_train.py \
  --input artifacts/system_b/pseudo_pairs_author_balanced.jsonl \
  --output-dir artifacts/system_b/model \
  --model-name google/flan-t5-base \
  --epochs 4 \
  --batch-size 8
```

Useful arguments:

- `--input`
- `--output-dir`
- `--model-name`
- `--batch-size`
- `--epochs`
- `--lr`
- `--weight-decay`
- `--max-source-length`
- `--max-target-length`
- `--dev-ratio`
- `--seed`

For topic-hard experiments, point `--input` and `--output-dir` to the topic-hard pseudo-pair file and desired output directory.

---

## Evaluation

### Evaluate System B

```bash
python systems/system_b_evaluate.py \
  --split-name test \
  --split-strategy standard \
  --batch-size 16
```

Useful arguments:

- `--split-name`
- `--split-strategy`
- `--limit`
- `--output`
- `--batch-size`
- `--system-b-mode`
- `--finetuned-model-dir`

This writes JSONL outputs and metrics JSON files.

### Build Retrieval Index

Before retrieval-based System C runs, build the retrieval index:

```bash
python retrieval/build_index.py --split standard
python retrieval/build_index.py --split topic_hard
```

### Run System C

```bash
python systems/system_c_rer_pipeline.py \
  --split-name test \
  --split-strategy standard
```

Useful arguments:

- `--split-name`
- `--split-strategy`
- `--limit`
- `--k-retrieve`
- `--k-generate`
- `--system-b-mode`
- `--finetuned-model-dir`
- `--prompt-fallback-model`
- `--no-retrieval`
- `--no-rerank`
- `--no-semantic-constraint`
- `--force-rebuild-index`
- `--write-retrieval-sanity`
- `--sanity-limit`
- `--tag`

Examples:

```bash
# System C without retrieval
python systems/system_c_rer_pipeline.py \
  --split-name test \
  --split-strategy standard \
  --no-retrieval

# System C with retrieval on topic-hard
python systems/system_c_rer_pipeline.py \
  --split-name test \
  --split-strategy topic_hard
```

The script writes split-specific outputs to `results/` and also maintains canonical files:

- `results/system_c_best.jsonl`
- `results/system_c_candidates.jsonl`
- `results/system_c_final_config.json`

### Evaluate System C

```bash
python systems/system_c_evaluate.py \
  --split-name test \
  --split-strategy standard
```

Useful arguments:

- `--split-name`
- `--split-strategy`
- `--limit`
- `--output`
- `--k-retrieve`
- `--k-generate`
- `--system-b-mode`
- `--finetuned-model-dir`
- `--prompt-fallback-model`
- `--no-retrieval`
- `--no-rerank`
- `--no-semantic-constraint`
- `--force-rebuild-index`
- `--best-input`

Examples:

```bash
# Evaluate System C without retrieval
python systems/system_c_evaluate.py \
  --split-name test \
  --split-strategy standard \
  --no-retrieval

# Evaluate an existing best-output file
python systems/system_c_evaluate.py \
  --split-name test \
  --split-strategy standard \
  --best-input results/system_c_best.jsonl
```

This writes:

- `results/system_c_outputs_*.jsonl`
- `results/system_c_eval_metrics_*.json`

---

## Standard vs Topic-Hard Workflows

The repository supports two main experiment settings:

- `standard`
- `topic_hard`

The easiest way to reproduce the end-to-end workflows is through the two root notebooks:

- `CS4248_Project_standard.ipynb`
- `CS4248_Project_topic_hard.ipynb`

These notebooks cover:

- dependency setup
- classifier training
- System B training
- System B evaluation
- System C evaluation with and without retrieval
- System A output generation
- comparison tables and exported files

---

## Important Artifacts

### Data

- `artifacts/data/cleaned.jsonl`
- `artifacts/data/cleaned_with_anchors.jsonl`

### Splits

- `artifacts/splits/standard.json`
- `artifacts/splits/topic_hard.json`

### Classifiers

- `artifacts/classifiers/logreg/`
- `artifacts/classifiers/rnn/`
- `artifacts/classifiers/transformer/`

### System B

- `artifacts/system_b/pseudo_pairs_filtered.jsonl`
- `artifacts/system_b/pseudo_pairs_author_balanced.jsonl`
- `artifacts/system_b/pseudo_pairs_author_balanced_topic_hard.jsonl`
- `artifacts/system_b/train_metrics.json`

### System C

- `results/system_c_best*.jsonl`
- `results/system_c_candidates*.jsonl`
- `results/system_c_metrics*.json`
- `results/system_c_outputs*.jsonl`
- `results/system_c_eval_metrics*.json`

---

## Notebooks

Support notebooks under `notebooks/` include:

- `data_analysis.ipynb`
- `finalize_split.ipynb`
- `system_a.ipynb`
- `test_systems.ipynb`
- `test_systems_topic_hard.ipynb`
- `topic_clustering_tfidf_kmeans.ipynb`
- `train_classifiers.ipynb`

Use the **root notebooks** for full end-to-end workflows and `notebooks/` for focused analysis or experimentation.

---
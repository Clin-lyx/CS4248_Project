# Data pipeline (Member 1)

How to build artifacts, load them in Python, and wire splits into training or evaluation.

Run commands from the **repository root** (`CS4248_Project`) with the project virtualenv active.

```powershell
.\.venv\Scripts\Activate.ps1
```

Or call the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m data.preprocess
```

---

## Dependencies

Install Member 1 stack (see `requirements-m1.txt`), then the spaCy English model:

```powershell
pip install -r requirements-m1.txt
python -m spacy download en_core_web_sm
```

If `import spacy` / `spacy.load` fails with pydantic errors, upgrade the spaCy stack together:

```powershell
pip install -U spacy thinc weasel confection pydantic
```

---

## Artifact overview

| Artifact | Produced by | Purpose |
|----------|-------------|---------|
| `artifacts/data/cleaned.jsonl` | `data.preprocess` | Canonical rows: stable `id`, `text`, `label`, provenance |
| `artifacts/data/cleaned_with_anchors.jsonl` | `data.anchors` | Same as cleaned + `anchors` column (spans for constraints / eval) |
| `artifacts/splits/standard.json` | `data.splits` | Frozen train/dev/test **IDs** (stratified by `label`) |

**Order:** preprocess → (optional) anchors → splits. Splits reference `id` values from `cleaned.jsonl`.

---

## One-shot CLI (rebuild everything)

```powershell
python -m data.preprocess
python -m data.anchors
python -m data.splits
```

---

## Record schemas

### `cleaned.jsonl` (one JSON object per line)

| Field | Type | Notes |
|-------|------|--------|
| `id` | string | `sar_000001`, … stable after preprocess |
| `headline_raw` | string | Original headline from source JSONL |
| `text` | string | Lightly normalized headline (default model input) |
| `label` | int | `0` = non-sarcastic, `1` = sarcastic |
| `article_link` | string | Source URL |
| `publisher` | string | `theonion` \| `huffpost` \| `other` (from hostname) |

### `cleaned_with_anchors.jsonl`

Same columns as above, plus:

| Field | Type | Notes |
|-------|------|--------|
| `anchors` | object | `entities`, `numbers`, `capitals`, `all` — each value is a list of `{text, start, end, type}` |

Offsets refer to the **`text`** field (character indices, Python slice semantics: `[start:end]`).

### `artifacts/splits/standard.json`

- `meta`: ratios, `random_state`, row counts, per-split label counts  
- `train`, `dev`, `test`: sorted lists of `id` strings  

---

## Python: load cleaned data

```python
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
df = pd.read_json(ROOT / "artifacts/data/cleaned.jsonl", lines=True, encoding="utf-8")
```

Or use the shared helper (resolves paths from repo root):

```python
from data.splits import load_cleaned_data

df = load_cleaned_data()
df_anchored = pd.read_json("artifacts/data/cleaned_with_anchors.jsonl", lines=True, encoding="utf-8")
```

---

## Python: apply the frozen split

```python
from data.splits import load_cleaned_data, get_split_df

df = load_cleaned_data()
train_df = get_split_df(df, "train")
dev_df = get_split_df(df, "dev")
test_df = get_split_df(df, "test")
```

Only `id` values present in `cleaned.jsonl` are valid. If you regenerate preprocess output, **IDs may change** — then regenerate anchors and splits and recommit `standard.json`.

Raw ID lists without filtering the DataFrame:

```python
from data.splits import load_split

bundle = load_split()
train_ids = set(bundle["train"])
```

---

## Python: anchors only (no full file)

```python
from data.anchors import extract_anchors, add_anchors

spans = extract_anchors("NASA says Mars is pretty neat")
# or batch a DataFrame:
df2 = add_anchors(df, text_col="text", batch_size=512)
```

---

## Notebooks

If the kernel cwd is `notebooks/`, add the repo root to `sys.path` before `from data....` imports (see `notebooks/data_analysis.ipynb`), or open Jupyter with **project root** as the working directory.

---

## Caveat for models and evaluation

In this dataset, **label and publisher are almost perfectly aligned** (HuffPost ↔ non-sarcastic, The Onion ↔ sarcastic). Treat strong classifier scores as potentially **topic- or site-driven** unless you also use a topic-hard split or other leakage checks from the project roadmap.

---

## Source file

Raw Kaggle-style JSONL (not committed as the canonical training table):

- `dataset/Sarcasm_Headlines_Dataset_v2.json` — fields `headline`, `is_sarcastic`, `article_link`

`DatasetLoader` in `data/load_dataset.py` reads that path relative to the repo root.

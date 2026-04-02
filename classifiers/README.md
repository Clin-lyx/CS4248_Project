# Classifier Pipelines

This folder contains three sarcasm-detection training + inference pipelines that use `artifacts/data/cleaned.jsonl` and split IDs from `artifacts/splits/standard.json`.

## Files

- `classifiers/logreg_classifier.py`
  - TF-IDF + Logistic Regression baseline
- `classifiers/rnn_classifier.py`
  - BiLSTM sequence classifier
- `classifiers/transformer_classifier.py`
  - Transformer fine-tuning (default: `distilbert-base-uncased`)

All pipelines output:

- `pred_label` (0/1)
- `prob_sarcastic`

## Artifact locations

- `artifacts/classifiers/logreg/`
- `artifacts/classifiers/rnn/`
- `artifacts/classifiers/transformer/`

## Quick start

```powershell
python classifiers/logreg_classifier.py train
python classifiers/rnn_classifier.py train
python classifiers/transformer_classifier.py train
```

Inference examples:

```powershell
python classifiers/logreg_classifier.py predict wow this definitely sounds normal
python classifiers/rnn_classifier.py predict Trump finally achieves world peace
python classifiers/transformer_classifier.py predict scientists shocked by obvious outcome
```

## Note for transformer_classifier.py

Artifacts are too large to be included in the repo, so you will need to train the model yourself.

You can still use the `predict` command without training and it will fallback to use the pretrained `distilbert-base-uncased` model, but it will not be fine-tuned for sarcasm detection and will likely perform poorly.

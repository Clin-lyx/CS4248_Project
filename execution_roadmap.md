# Roadmap and Work Split for Bidirectional Sarcasm Headline Rewriting

## Final recommended project scope

**Exact version of the project to do**  
Build a **bidirectional headline rewriting system** that flips sarcasm label (non-sarcastic ↔ sarcastic) **while preserving meaning via controllable minimal edits**, and select the best rewrite using a **detector-guided reranker** with explicit semantic and edit constraints.

**One sentence: what we are building**  
A controllable “generate → score → select” pipeline that rewrites headlines into the opposite sarcasm style with **low semantic drift** and **high interpretability**, backed by leakage-aware evaluation.

### Must-have components
- **Data + splitting**
  - Standard random split (for fast iteration).
  - **Topic-hard split** (cluster headlines by semantic embedding; hold out clusters to stress-test leakage/topic reliance).
  - Fixed seeds + split files committed (reproducibility).
- **Two directions implemented**
  - Non-sarcastic → sarcastic (N→S)
  - Sarcastic → non-sarcastic (S→N)
- **Multiple systems (minimum 3)**
  - Identity baseline (copy input).
  - Template baseline (hand-designed sarcasm / de-sarcasm transforms).
  - Main system (controlled candidate generation + reranking).
- **Sarcasm detector**
  - Lightweight baseline detector (TF–IDF + logistic regression) for interpretability.
  - One stronger detector (small transformer fine-tune) if feasible; used primarily as critic/evaluator.
  - You must explicitly treat classifier-style scores as *imperfect* per style-transfer evaluation literature. citeturn0search1
- **Evaluation that matches style transfer**
  - Style strength (detector score), content preservation (semantic similarity + entity/number retention), fluency/headline-likeness (LM perplexity or lightweight acceptability + human ratings). citeturn0search1turn1search0turn1search1
- **Microscopic analysis**
  - Error taxonomy + manual inspection of a curated failure set.
  - Detector-vs-human disagreement slice (at least 15–30 examples).
- **Small but well-structured human evaluation**
  - 100–200 inputs total, 2 directions, 3 systems (template vs main vs optional prompt), 3-dimension ratings (style, meaning, fluency).

### Optional extensions (only if core works by end of Week 2)
- Retrieve-and-edit candidate proposal module (adds diversity while staying headline-like). citeturn1search2
- “Sarcasm mechanism” tagging (hyperbole / rhetorical question / ironic praise) and per-mechanism analysis (excellent discussion material; low compute). citeturn0search3
- A lightweight edit-based style-transfer model trained on synthetic pairs (riskier; only if time remains). citeturn0search2

---

## System roadmap

Below is the roadmap **in chronological implementation order**, with ownership and done-criteria.

### Stage zero: Repo, contracts, and evaluation scaffolding
**Goal**  
Make the project “runnable end-to-end” early, so every later improvement is measurable.

**Concrete tasks**
- Create a single repo structure: `data/`, `splits/`, `systems/`, `metrics/`, `experiments/`, `analysis/`, `human_eval/`.
- Define a unified JSONL format for examples:
  - `id`, `text`, `label`, `split`, `topic_cluster_id`, plus extracted anchors (entities, numbers).
- Implement a single CLI entrypoint:
  - `run_system(system_name, split, direction) -> outputs.jsonl`.
  - `evaluate(outputs.jsonl) -> metrics.json + plots`.
- Set random seeds; register packages/versions.

**Expected output**
- One script can run Identity baseline on a 100-sample subset and print metrics.

**Who can own it**
- Reproducibility/infra owner.

**Dependencies**
- None.

**Done when**
- New teammate can clone repo and reproduce the same Identity outputs + metrics on the same subset.

---

### Stage one: Dataset analysis + leakage-aware splits
**Goal**  
Build defensible splits and identify confounds early.

**Concrete tasks**
- Confirm dataset assumptions (sources, class balance, fields). The dataset is curated from satire site **entity["organization","The Onion","satirical news site, us"]** and real-news **entity["organization","HuffPost","news site, us"]**, ~28K headlines with ~13K sarcastic and includes source links. citeturn0search0turn0search4
- Build **topic embeddings** (Sentence-BERT or similar) for clustering; SBERT is designed for fast cosine similarity search and clustering. citeturn1search1turn1search5
- Create:
  - Standard split (e.g., 80/10/10).
  - Topic-hard split (cluster → split by cluster; ensure both labels appear in each split as much as possible).
- Write leakage checks:
  - Train TF–IDF classifier on standard split; evaluate on topic-hard test; measure drop.
  - Extract top-weight n-grams and manually label them (topic, stylistic cue, named entity, etc.).

**Expected output**
- `/splits/standard.json`, `/splits/topic_hard.json`, plus a short “Corpus Confounds” note with 10–20 top lexical cues.

**Who can own it**
- Data/splits owner + analysis owner.

**Dependencies**
- Stage zero.

**Done when**
- Splits are frozen and checked into repo, and a leakage diagnostic figure/table exists.

---

### Stage two: Sarcasm detectors (for evaluation + reranking)
**Goal**  
Have at least one reliable-enough critic, plus an interpretable baseline.

**Concrete tasks**
- Detector A (must-have): TF–IDF + logistic regression (fast, interpretable).
- Detector B (recommended): small transformer classifier fine-tuned on headlines (e.g., DistilBERT/RoBERTa-base if compute allows).
- Calibration sanity checks:
  - Identity baseline should *not* flip style score much.
  - Template baseline should flip style score somewhat.

**Expected output**
- Saved detector checkpoints + `predict_style(texts) -> prob_sarcastic`.

**Who can own it**
- Detector owner.

**Dependencies**
- Stage one splits.

**Done when**
- You can report detector accuracy on standard vs topic-hard, and use detector probabilities as a scoring function.

---

### Stage three: Baseline generators (identity + templates)
**Goal**  
Establish credible baselines fast (they also uncover evaluation bugs).

**Concrete tasks**
- Identity baseline (copy).
- Template baseline (must-have):
  - N→S: add one of a small set of sarcasm framings (rhetorical question tail, “of course”, hyperbole marker), **without changing anchors**.
  - S→N: remove sarcasm markers (intensifiers, rhetorical tails, sarcasm-y adverbs), smooth to neutral phrasing.
- Record every rule/template and link each to a “sarcasm mechanism” hypothesis (useful later in Discussion).

**Expected output**
- `systems/template.py` produces 5–10 candidates per input + 1 “best guess” output.

**Who can own it**
- Baselines owner.

**Dependencies**
- Stage zero format; anchor extractor from Stage one (for entity/number preservation rules).

**Done when**
- Template outputs exist for 200 examples in both directions, and obvious meaning-drift is rare.

---

### Stage four: Main system — controlled candidate generation + reranking
**Goal**  
Produce higher-quality rewrites than templates with measurable trade-offs.

**Concrete tasks**
- Candidate generator (must-have):
  - Generate 20–40 candidates per input using **controlled micro-edits** (insert/replace/delete) around evaluative framing, while **locking anchors** (entities, numbers).
  - Keep an explicit **edit budget** (e.g., ≤2 content-word edits) to enforce “minimal edit” and interpretability.
- Scorers (must-have):
  - Style: detector probability for target style.
  - Content: SBERT cosine similarity. citeturn1search1turn1search5
  - Optional but recommended content metric: BERTScore (robust semantic similarity metric for generation). citeturn1search0
  - Edit cost: token-level Levenshtein distance or % tokens changed (your controllability axis).
- Reranker (must-have):
  - Weighted score: `w_style*style + w_content*content - w_edit*edit_cost (+ w_fluency*fluency if you add it)`.
  - Tune weights on dev set; report the chosen weights (design justification).
- Hard filters (must-have):
  - Entity/number retention thresholds.
  - Minimum semantic similarity threshold.

**Expected output**
- For each input: `k` candidates + `best` selection + per-candidate score breakdown (for later error analysis).

**Who can own it**
- Main system owner (+ support from detector + metrics owners).

**Dependencies**
- Stage two detectors, Stage one anchor extraction, Stage zero evaluation harness.

**Done when**
- On a dev subset, the main system beats template baseline on:
  - style flip rate **and**
  - semantic preservation **and/or**
  - human preference (pilot test ~20 items).

---

### Stage five: Human evaluation + error analysis package
**Goal**  
Lock in rubric-aligned evidence: macro + micro + language-focused discussion.

**Concrete tasks**
- Human eval design (must-have):
  - 3 axes: style correctness, meaning preservation, fluency/headline-likeness.
  - Sample: 100–200 inputs; include both directions.
  - Systems: at least Template vs Main (+ optional prompt baseline).
- Error analysis (must-have):
  - Build taxonomy (meaning drift, anchor loss, unnatural headline, “sarcasm marker only”, detector fooled).
  - Create a curated set of 30–50 failure cases with annotations and model score breakdown.

**Expected output**
- Annotator form, instructions, dataset, inter-annotator agreement, summary tables, and a qualitative appendix set.

**Who can own it**
- Human eval + analysis owner.

**Dependencies**
- Stage four outputs locked.

**Done when**
- You have a final table of human ratings and a completed error casebook.

---

## Work split for six people

Assign roles so work is parallel from Day 1, but converges cleanly by Week 2.

### Role A: Data + splits lead
**Main responsibility**  
Own dataset preprocessing, anchor extraction, and leakage-aware splits.

**Specific deliverables**
- Frozen `standard` + `topic-hard` split files.
- Topic clustering notebook/script + summary stats.
- Anchor extractor (NER + number patterns) usable by generators and evaluators.

**Week focus**
- Week 1: splits + leakage diagnostics.
- Week 2: finalize anchor constraints and topic slice definitions.
- Week 3: contribute to meso-level analyses (topic bins, entity bins).
- Week 4: help write Corpus Analysis section + limitations.

**Dependencies**
- Needs repo skeleton (Role F) early.

**Backup tasks if blocked**
- Build frequency-based “topic proxy” bins (TF–IDF top terms) if clustering is slow.

---

### Role B: Detector + critic lead
**Main responsibility**  
Train and validate sarcasm detectors; expose robust scoring APIs.

**Specific deliverables**
- TF–IDF baseline detector + interpretability report (top features).
- One stronger detector if feasible.
- `score_style(text)` function and calibration plots.

**Week focus**
- Week 1: TF–IDF detector + baseline metrics (standard vs topic-hard).
- Week 2: stronger detector + integrate into reranker.
- Week 3: disagreement analysis set (detector vs human).
- Week 4: Discussion write-up on detector limitations (classifier-as-metric caution). citeturn0search1

**Dependencies**
- Needs split files from Role A.

**Backup tasks if blocked**
- Stick with TF–IDF + add probability calibration / threshold tuning.

---

### Role C: Baseline rewriting lead
**Main responsibility**  
Deliver strong, interpretable baselines that are easy to explain.

**Specific deliverables**
- Identity baseline.
- Template baseline for both directions.
- Documentation: “template ↔ hypothesized sarcasm mechanism”.

**Week focus**
- Week 1: implement baseline templates; generate pilot outputs.
- Week 2: refine templates with anchor safety rules.
- Week 3: run full baseline generation and provide examples for report.
- Week 4: contribute qualitative comparisons + examples.

**Dependencies**
- Needs anchors from Role A (entity/number lock).

**Backup tasks if blocked**
- Implement a “minimal delete” S→N baseline (remove sarcasm markers only) to guarantee meaning retention.

---

### Role D: Main system engineering lead
**Main responsibility**  
Build the controlled candidate generator + reranker pipeline.

**Specific deliverables**
- Candidate generation module (multiple strategies; edit budget).
- Reranker with configurable weights + dev tuning.
- Per-candidate score logging (debuggability).

**Week focus**
- Week 1: skeleton generator producing candidates without reranking.
- Week 2: integrate scorers + reranking; achieve clear gains on dev.
- Week 3: run ablations and export full outputs for analysis/human eval.
- Week 4: finalize error breakdown tooling.

**Dependencies**
- Needs detectors (Role B) and similarity metrics (Role E).

**Backup tasks if blocked**
- Reduce to fewer strategies (e.g., 2–3 best ones) but keep reranking + constraints.

---

### Role E: Evaluation + analysis lead
**Main responsibility**  
Own metrics, plots, and error analysis pipeline; design human eval.

**Specific deliverables**
- Metric suite: style score, SBERT similarity, BERTScore, edit distance. citeturn1search0turn1search5
- Meso-slice dashboards (topic bins, entity bins, length bins).
- Human eval protocol + aggregation scripts.

**Week focus**
- Week 1: implement evaluation harness; sanity-check metrics on identity outputs.
- Week 2: integrate with reranker outputs; set up ablation comparison tables.
- Week 3: run human evaluation + error taxonomy.
- Week 4: write Experiments + Discussion figures/tables.

**Dependencies**
- Needs system outputs (Role C/D), and detector API (Role B).

**Backup tasks if blocked**
- If BERTScore integration slows: rely on SBERT similarity + entity/number overlap as primary content metrics.

---

### Role F: Reproducibility + report integration lead
**Main responsibility**  
Make everything reproducible and report-ready; manage experiment logging.

**Specific deliverables**
- Experiment config system (YAML/JSON configs), run logs, seeds.
- Standardized output folders and naming scheme.
- Report skeleton + figure/table placeholders; final integration.

**Week focus**
- Week 1: repo scaffolding + run scripts; enforce “one-command reproduce”.
- Week 2: set up automated ablation runs.
- Week 3: consolidate tables/figures + ensure consistent captions.
- Week 4: final write-up, limitations, and AI-tool audit trail.

**Dependencies**
- Needs early alignment with all roles on file formats.

**Backup tasks if blocked**
- Maintain a single “results spreadsheet” and enforce manual logging discipline.

---

## Execution and experiments

**Four-week execution plan**

### Week one
**Must-do tasks**
- Repo scaffolding + end-to-end pipeline running on a 100-example subset.
- Standard + topic-hard splits frozen.
- TF–IDF detector trained + evaluated on both splits.
- Identity + template baselines implemented for both directions.

**Useful but optional tasks**
- Stronger transformer detector.
- Pilot human eval rubric on 10 examples.

**Checkpoint by end of week**
- One table: detector accuracy (standard vs topic-hard) + baseline generator style flip rate on a tiny subset.

**Danger signs**
- No frozen splits by Day 5.
- Baselines aren’t producing outputs in both directions.
- No metrics computed end-to-end.

**Cut if behind**
- Stronger detector (keep TF–IDF).
- Any seq2seq training talk.

---

### Week two
**Must-do tasks**
- Implement main candidate generator with edit budget + anchor lock.
- Implement SBERT similarity + edit distance + reranker (weights tuned on dev).
- Generate outputs for all dev examples for 3 systems (identity/template/main).

**Useful but optional tasks**
- Add BERTScore metric (if painless). citeturn1search0
- Add two extra candidate strategies (rhetorical question, ironic praise).

**Checkpoint by end of week**
- Main system measurably improves the style/content trade-off vs template on dev.

**Danger signs**
- Main system “wins” style score only by destroying meaning (low similarity).
- Reranker weights are unstable (huge swings).

**Cut if behind**
- Reduce candidate count and strategies; keep reranking + constraints.

---

### Week three
**Must-do tasks**
- Full runs on test sets (standard + topic-hard).
- Ablations (remove reranker; remove content filter; change edit budget).
- Launch human evaluation (collect ratings).

**Useful but optional tasks**
- Detector-vs-human disagreement study.
- Meso-level slicing by topic clusters and entity count.

**Checkpoint by end of week**
- Final results table draft + first two plots (trade-off curve; split comparison).

**Danger signs**
- Human eval not started by mid-week.
- No qualitative examples selected.

**Cut if behind**
- Extra analyses; keep human eval + one strong error taxonomy.

---

### Week four
**Must-do tasks**
- Finish human eval aggregation + agreement.
- Write error analysis (30–50 examples) with categories and fixes.
- Finalize report figures/tables and write-up.

**Useful but optional tasks**
- Retrieve-and-edit extension pilot (only if everything else is done). citeturn1search2

**Checkpoint by end of week**
- Submission-quality report with reproducible code + locked results.

**Danger signs**
- Too many “we tried but didn’t finish” sections.
- Missing Discussion answers tied to evidence.

**Cut if behind**
- Extensions, extra models, extra plots.

---

**Experiment roadmap**

Run experiments from fastest sanity checks to final runs—do not skip the early gates.

1) **Sanity E0: Identity baseline**
- **Purpose:** verify evaluation pipeline isn’t broken.
- **Variant:** copy input.
- **Cost/difficulty:** trivial.
- **Decision it helps:** validates metrics and detectors.

2) **Sanity E1: Detector leakage check**
- **Purpose:** quantify topic leakage risk.
- **Variant:** TF–IDF detector evaluated on standard vs topic-hard.
- **Cost:** low.
- **Decision:** whether you must emphasize leakage-aware evaluation (you should). citeturn0search0turn0search1

3) **Sanity E2: Template baseline quick win**
- **Purpose:** confirm the task is feasible and “flip-able.”
- **Variant:** templates for both directions.
- **Cost:** low.
- **Decision:** establishes baseline trade-off.

4) **Dev E3: Main generator without reranking**
- **Purpose:** test candidate quality before complicated scoring.
- **Variant:** generate N candidates, take first/heuristic pick.
- **Cost:** low–medium.
- **Decision:** whether candidate strategies are viable.

5) **Dev E4: Add reranking (style + content + edit)**
- **Purpose:** validate the core idea: selection improves trade-offs.
- **Variant:** full reranker.
- **Cost:** medium.
- **Decision:** finalize weights and filters.

6) **Dev E5: Edit budget sweep**
- **Purpose:** controllability curve (excellent for report).
- **Variant:** edit budget {1,2,3}.
- **Cost:** medium.
- **Decision:** choose the default budget for final system; provides Discussion evidence. citeturn0search2

7) **Test E6: Full evaluation on two splits**
- **Purpose:** final macro results.
- **Variant:** identity/template/main on standard and topic-hard.
- **Cost:** medium.
- **Decision:** final table.

8) **Human E7: Small human eval**
- **Purpose:** validate metrics and produce rubric-aligned evidence.
- **Variant:** template vs main (+ optional prompt).
- **Cost:** medium organizational cost.
- **Decision:** final claims.

---

## MVP and extensions

**Minimal viable project**

This is the smallest version that still looks like a **solid CS4248 submission**.

### MVP includes
- **Systems**
  - Identity baseline
  - Template baseline
  - Main system: controlled candidate generation + reranking
- **Evaluation**
  - Style success: sarcasm detector probability/accuracy (with leakage caveats). citeturn0search1
  - Content preservation: SBERT cosine similarity; entity/number retention.
  - Edit cost: token-level edit distance.
  - Two splits: standard + topic-hard.
- **Analysis**
  - Leakage diagnostic (standard vs topic-hard detector performance).
  - Error taxonomy with 30–50 curated examples.
  - Small human evaluation (100–150 inputs total), 3 axes (style/meaning/fluency).

### MVP explicitly should NOT attempt
- Training a full seq2seq generator end-to-end (too risky in 4 weeks).
- Large-scale prompt engineering rabbit holes without controlled evaluation.
- Too many model variants (cap at 3–4 systems + ablations).

---

**Ambitious but realistic extension**

Only attempt these if the main system is stable by the end of Week 2 and full test runs exist.

1) **Retrieve-and-edit candidate proposal (adds diversity, often better headline shape)**
- Worth doing because it naturally fits “generate candidates then rerank” and has strong precedent in retrieve-edit-rerank frameworks. citeturn1search2
- Not core because it adds complexity in retrieval quality and potential memorization concerns.

2) **Sarcasm-mechanism tagging and per-mechanism analysis**
- Worth doing because it produces a language-focused Discussion with minimal extra compute, aligning with rhetorical-device perspectives on sarcasm. citeturn0search3
- Not core because it’s analysis-heavy; do it after results are locked.

---

## Deliverables checklist

By the end, you should have these artifacts committed/exported.

### Code modules
- `data_loader.py` + preprocessing
- `split_builder.py` (standard + topic-hard)
- `anchors.py` (entities/numbers extraction)
- `detectors/` (TF–IDF + optional transformer)
- `systems/identity.py`
- `systems/template.py`
- `systems/main_generate_rerank.py`
- `metrics/style.py`, `metrics/sbert_sim.py`, `metrics/bertscore.py` (if used), `metrics/edit_distance.py`
- `analysis/error_taxonomy.ipynb` (or script)
- `run_experiment.py` (single entrypoint)

### Datasets / splits
- Frozen split files + README describing how they were built.
- Topic cluster assignments per example.

### Saved outputs
- Outputs JSONL for each (system × direction × split).
- Candidate logs for main system (top-k candidates + scores).

### Tables
- Macro results table (two splits).
- Ablation table.
- Human eval summary table.

### Figures
- Trade-off plot: style vs content (and/or vs edit distance).
- Split sensitivity plot: standard vs topic-hard performance change.
- Error distribution bar chart (failure categories).

### Human evaluation materials
- Annotation form (Google Form / spreadsheet)
- Instructions + examples
- Rater assignment plan
- Raw ratings + aggregation script

### Report sections (drafts by end of Week 3)
- Corpus analysis + leakage checks
- Method (systems + reranker)
- Experiments (tables/figures)
- Discussion (answers to 4–6 research questions with evidence)

---

## Common failure modes

1) **Relying on a sarcasm classifier as the only success metric**
- Why it fails: style-transfer evaluation is known to be inconsistent and classifier-only metrics can be misleading; you need content + fluency + human validation. citeturn0search1
- Avoid by: always reporting the triad (style/content/fluency) + human eval.

2) **Meaning drift (especially N→S)**
- Avoid by: anchor locking (entities/numbers), minimum semantic similarity thresholds using SBERT/BERTScore. citeturn1search0turn1search5

3) **Over-scope: too many models**
- Avoid by: committing to the 3-system storyline (identity/template/main) early.

4) **No leakage-aware story**
- Dataset is sourced from different publishers and can induce spurious cues; you need at least one hard split and a leakage diagnostic narrative. citeturn0search0turn0search4

5) **Starting human evaluation too late**
- Avoid by: preparing the rubric in Week 1 and launching in Week 3 no matter what.

6) **Weak qualitative analysis**
- Avoid by: maintaining a living “error casebook” as soon as Week 2 outputs exist.

7) **Reproducibility gaps**
- Avoid by: frozen splits, fixed seeds, saved outputs, and experiment configs from Week 1.

---

## Final advice

**Most important design principle**  
Treat sarcasm rewriting as **multi-objective controllable editing**: optimize **style flip** while explicitly constraining **content preservation** and **edit cost**, and report the trade-offs (not just one score). citeturn0search1turn0search2

**Most important thing to avoid**  
Do not let your project become “we generated text and a classifier liked it”—that’s fragile and hard to defend under leakage and metric validity concerns. citeturn0search1

**Best default plan if you become uncertain**  
Freeze scope to: **identity + template + controlled-generate-and-rerank**, evaluate on **standard + topic-hard splits**, run a **small human study**, and invest remaining time into **error analysis + language-focused discussion**.
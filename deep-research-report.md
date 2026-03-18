# Research Memo: Controllable, Leakage-Aware Sarcasm Style Transfer for News Headlines

## Executive recommendation

**Recommended single best framing**  
Build a **controllable, meaning-preserving sarcasm style transfer system for headlines** framed explicitly as **minimal-edit rewriting under leakage-aware evaluation**. The project is: given a headline, rewrite it into the *opposite* sarcasm style (sarcastic ↔ non-sarcastic) while preserving key propositional content (entities, event predicate, numbers) and staying headline-like (length, syntax, register).

This framing is well-matched to your dataset because it is **not parallel** (no gold paired rewrites), and because sarcasm is a **pragmatics + rhetoric** phenomenon rather than a purely lexical one (valence reversal, incongruity, rhetorical questions, hyperbole). citeturn4search8turn2search8 It is also well-matched to your course constraints because it naturally supports: multiple systems (baseline + improved pipeline), ablations, macro + micro evaluation, and language-focused discussion questions.

**One main contribution to aim for**  
A **multi-objective “generate → score → select” rewriting pipeline** that is:
- **controllable** (explicit rewrite strategies and “sarcasm intensity” knobs),
- **auditable** (each micro-edit is explainable),
- **robustly evaluated** against dataset confounds (topic/publisher leakage checks), and
- **analyzed linguistically** via a headline-specific taxonomy of sarcasm mechanisms (hyperbole, rhetorical framing, ironic praise, incongruity triggers). citeturn2search8turn4search8turn2search2

**Why this fits a CS4248-style rubric better than “train a bigger model”**  
A bigger generator can produce fluent text, but your current constraints (“~1 month,” limited compute, short report) favor a project whose *story* is about **NLP design choices**: (i) how to preserve meaning without references, (ii) how to operationalize style success, and (iii) how to test for leakage and spurious cues. The style-transfer literature explicitly treats evaluation as multi-dimensional—**style strength, content preservation, fluency**—and warns that inconsistent or unvalidated metrics can mislead conclusions. citeturn4search2turn4search18turn3search3  
A controlled pipeline lets you defend every component (“why, what, how”), run clean ablations, and present interpretable error slices—exactly the kind of analysis-heavy project that scores well even without SOTA generation.

## Task and corpus framing

**Problem framing**  
You have a labeled headline dataset curated from two sources: one intended to be sarcastic/satirical and one intended to be non-sarcastic/real news. citeturn1view1turn1view2turn9view0 Each record explicitly contains `headline` text and a sarcasm label (`is_sarcastic`), and includes an `article_link` field. citeturn1view1turn1view2 The dataset creators describe it as roughly **28K** headlines with about **13K** sarcastic instances. citeturn1view1turn1view2

Concretely define two directional tasks:

- **De-sarcasm (S→N)**: Input is sarcastic headline *h*; output is headline *h′* that reads non-sarcastic while preserving core content anchors (entities, key noun phrases, numbers) and remaining grammatical and headline-like.
- **Sarcasm transfer (N→S)**: Input is non-sarcastic headline *h*; output is sarcastic headline *h′* that expresses ironic stance, typically via rhetorical devices (hyperbole, rhetorical questions, evaluative reversal), while preserving the same anchors.

Because there are no gold pairs, your project must define what “preserve meaning” means **operationally** (entity retention, semantic similarity threshold, minimal edits), and what “style success” means **operationally** (detector score + human judgement). The text style transfer evaluation literature treats this as standard: evaluation should separately quantify **style**, **content**, and **fluency**. citeturn4search2turn4search18turn2search9

**Why sarcasm transfer in headlines is an NLP problem (not generic text generation)**  
Sarcasm is not merely “add negative words.” It frequently relies on:
- **Valence reversal**: surface praise while intending criticism (or vice versa). citeturn4search8turn4search17  
- **Semantic/pragmatic incongruity**: an evaluative stance that conflicts with world knowledge or expectation. citeturn2search8turn4search8  
- **Rhetorical devices** such as hyperbole, rhetorical questions, deliberate falsehood/overstatement, and targeted mockery. citeturn2search8turn2search2turn2search14  

Headlines make this harder: they are short, elliptical, and often lack explicit context—forcing the system to encode sarcasm using **lexico-syntactic choices** (intensifiers, framing, quotations, adverbials, presupposition triggers) rather than longer discourse cues.

**Major dataset risks and confounds you should explicitly address**

1) **Publisher/site leakage (label ≈ source)**  
The dataset is constructed from two distinct sites: sarcastic headlines come from entity["organization","The Onion","satirical news site"] and non-sarcastic headlines come from entity["organization","HuffPost","news site, us"]. citeturn1view1turn1view2turn9view0  
This creates a risk that models—and even your evaluation classifier—learn “publisher style” rather than sarcasm mechanisms. In generation, this can manifest as producing “Onion-like absurdity” rather than sarcasm that preserves the original event proposition.

2) **Topic leakage (satire topics differ from real news topics)**  
Even if you remove explicit domains, topics and named entities may differ substantially between the two sources. That means a strong detector score may reflect topic artifacts rather than sarcasm style. Style transfer evaluation work repeatedly emphasizes that classifiers can overestimate “style success” when domain shifts and spurious cues exist. citeturn4search2turn4search18turn0search6

3) **Weak pairing problem (no aligned references)**  
Your dataset is *not* parallel: there is no (headline, rewrite) supervision. citeturn1view1turn9view0 Approaches that fabricate pseudo-pairs can introduce noise and encourage meaning drift, a known challenge in unsupervised style transfer and synthetic-pair methods. citeturn0search14turn0search17turn3search16

4) **Meaning preservation is intrinsically tricky for satire**  
Some satirical headlines describe implausible events. In S→N, “preserving meaning” can conflict with producing plausible non-sarcastic tone. You must decide: is the goal to preserve *semantic content* even if implausible, or to “normalize” into plausible news? Choosing the former makes evaluation more principled; choosing the latter may be entertaining but harder to justify scientifically.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["The Onion headline screenshot","HuffPost headline screenshot","satirical news headline example","real news headline example"],"num_per_query":1}

## Ranked directions and final method

**Three ranked project directions**

**Direction A (rank 1): Minimal-edit, strategy-controlled rewriting with detector-guided reranking**  
- **Core idea**: Generate multiple candidates using explicit rewrite strategies (hyperbole, rhetorical tag, ironic praise); enforce semantic constraints (entity retention + embedding similarity); pick the best candidate via a weighted reranker using a sarcasm detector + semantic similarity + fluency heuristics.  
- **Novelty**: High *within CS4248 scope* because the novelty is in **analysis + controllability**: you will define and measure “minimal edit sarcasm transfer” and show trade-offs across strategies and topic-hard splits. This aligns with style transfer evaluation best practice: report style/content/fluency separately. citeturn4search2turn2search9  
- **Feasibility in 1 month**: Very feasible; most components are standard NLP building blocks (tokenization, NER, embedding similarity, classifier training, heuristic generation).  
- **Compute cost**: Low-to-mid (one small classifier + embeddings; optional small LM for fluency).  
- **Expected difficulty**: Moderate engineering + careful evaluation design.  
- **Likely failure points**: (i) detector overestimates sarcasm due to leakage; (ii) candidates become templated and repetitive; (iii) S→N may still sound “odd” even if non-sarcastic.  
- **Report story enabled**: A clean pipeline story with interpretable ablations: “which sarcasm mechanisms work,” “how minimal edits trade off with style strength,” and “what leakage-aware evaluation changes.”

**Direction B (rank 2): Retrieve-and-edit style transfer with prototype headlines**  
- **Core idea**: Retrieve semantically similar headlines from the target style as prototypes; edit them to insert source entities/slots, then rerank. This matches classic retrieve–edit–rerank paradigms. citeturn3search0turn3search1  
- **Novelty**: Medium; retrieve–edit is known, but applying it to sarcasm transfer with explicit meaning constraints and leakage-hard splits can be novel.  
- **Feasibility**: Feasible, but retrieval quality heavily determines outcome.  
- **Compute**: Low-to-mid (embedding index + lightweight editing).  
- **Difficulty**: Moderate; you need a robust “slotting” method for headlines.  
- **Failure points**: Prototype mismatch → meaning drift; outputs may copy training headlines too closely (memorization concerns).  
- **Report story**: Strong for analysis: you can show how prototype choice affects sarcasm strength and meaning retention, mirroring retrieve-edit-rerank analysis setups. citeturn3search0turn3search1

**Direction C (rank 3): Pseudo-parallel seq2seq transfer (T5/BART) trained on synthetic pairs**  
- **Core idea**: Construct pseudo-pairs via semantic retrieval + filtering; fine-tune a small seq2seq model on “rewrite into target style.”  
- **Novelty**: Lower (common approach in non-parallel style transfer), but still reportable if you emphasize noise analysis and robustness. Unsupervised/synthetic style transfer is a major research line. citeturn0search14turn0search17turn3search16  
- **Feasibility**: Riskier in 1 month because data synthesis + training + debugging can consume time.  
- **Compute**: Mid (fine-tuning GPU helpful).  
- **Difficulty**: Higher; hardest to keep controllable and analyzable.  
- **Failure points**: Noisy pseudo-pairs can teach the model to drift topics; training may “work” but the evaluation becomes muddy without careful controls.  
- **Report story**: Good for “learning from noisy supervision,” but weaker for interpretability unless you invest in extensive diagnostics.

**Final recommended method (what to implement)**  
Implement **Direction A** as the main system, with a small slice of Direction B as an optional module (prototype-based candidate generation). This yields a crisp, achievable centerpiece while keeping an “advanced” extension available if time remains.

## Method design for a six-person team

**Preprocessing (must-have)**  
- Treat the dataset as headline-only during generation; strip or ignore `article_link` to prevent accidental leakage in generation inputs. (Use it only for optional analyses such as date/topic extraction.) citeturn1view1turn1view2  
- Normalize whitespace; preserve case for named entities (headline case matters).  
- Build an NLP annotation pipeline that extracts:
  - Named entities (NER),
  - Numbers/percentages/money,
  - High-IDF content words (TF–IDF keywords),
  - Punctuation features (“!”, quotes, parentheses), which are often sarcasm-relevant cues in short text. citeturn4search8turn2search8  

**Data splitting strategy (must-have)**  
You want **two complementary evaluation settings**:

- **Standard split** (for quick iteration): random train/dev/test with fixed seed.  
- **Topic-hard split** (for your originality + leakage story): cluster headlines by semantic embeddings (or TF–IDF + k-means), then split by clusters so test topics are unseen during training. Use this split for both (i) sarcasm detector robustness checks and (ii) style transfer evaluation.

This directly addresses concerns raised in style transfer evaluation work: classifier-based style metrics can be misleading if domain/topic cues correlate with style labels. citeturn4search2turn4search18turn0search6

**Leakage checks (must-have, and very reportable)**  
Run these as a dedicated “Corpus Analysis” subsection:

- Train a sarcasm detector using only:
  - (a) bag-of-words / TF–IDF n-grams baseline,
  - (b) a lightweight transformer classifier (optional).  
- Compare performance between:
  - standard split vs topic-hard split.  
A large drop indicates topic leakage/spurious cues.

Also do a **feature audit**: identify top-weight n-grams in the linear model and manually categorize them (named entities, political terms, caricature vocabulary, etc.). This turns a common dataset weakness into an “original analysis” win.

**Baseline systems (must-have; keep them simple and clean)**  
You need at least two baselines plus your main system:

- **Baseline 0: Identity rewrite** (copy input).  
  Purpose: exposes how much your style classifier is fooled by content alone (should score “wrong style” most of the time; if not, your evaluation is broken).

- **Baseline 1: Template-only**  
  - N→S: append or prepend one sarcasm tag phrase (e.g., “in a stunning turn of events,” “because of course,” “what could possibly go wrong”). Keep to 8–15 curated templates.  
  - S→N: remove sarcasm markers (common intensifiers, rhetorical tags, “yeah right”-style cues), delete trailing punchline-like clauses if detected.  
  Purpose: establishes a controllable, interpretable baseline.

- **Baseline 2: Unconstrained prompt rewrite (optional if allowed)**  
  Use a fixed prompt recipe to rewrite sarcastically/non-sarcastically and generate N candidates. This gives an approximate “upper bound” for fluency/diversity, but must be logged carefully for reproducibility (prompt, parameters, outputs). Treat it as an *analysis baseline*, not your main contribution.

**Main system: Strategy-controlled candidate generation + multi-objective reranking (must-have)**

1) **Candidate generation (N→S)**  
Generate ~20–40 candidates per headline by combining:
- **Strategy choice** (one per candidate; controllable):
  - Hyperbole / overstatement,
  - Rhetorical question,
  - Ironic praise (“brilliant,” “heroic,” “groundbreaking”) applied to a typically negative/neutral situation,
  - Incongruity framing (“experts shocked that…”).  
These are motivated by rhetoric-centric views of sarcasm and its forms. citeturn2search8turn2search2turn4search8  
- **Minimal edits**: constrain to at most 1–2 insertions + ≤1 substitution (configurable “intensity” knob).
- **Anchor protection**: force-keep named entities and numbers (hard constraint).

2) **Candidate generation (S→N)**  
Generate candidates by:
- Removing rhetorical tails (“because why not,” similar),
- Downshifting hyperbole (“incredible”→neutral adjective deletion),
- Replacing evaluative adjectives with neutral ones,
- Removing scare quotes if present.  
Keep “headline grammar” by enforcing length and avoiding full sentences when source is fragmentary.

3) **Scoring and reranking**  
Compute three core scores per candidate (the style-transfer literature standardizes evaluation around these dimensions): citeturn4search2turn4search18turn2search9  
- **Style score**: probability of target style from a trained sarcasm detector.  
  Important: evaluate with **two detectors**:
  - Detector trained on standard split (in-domain),
  - Detector trained on topic-hard split (more robust).  
- **Content preservation score**:
  - Embedding cosine similarity (Sentence-BERT-style),
  - Plus **BERTScore** as a reference-free semantic similarity proxy (treat source as “reference” cautiously). citeturn3search3turn3search15  
- **Fluency/acceptability score**:
  - Perplexity under a small LM, or a lightweight acceptability classifier.  
  (You must interpret cautiously: headlines are not typical LM training format, so fluency metrics are imperfect—call this out in limitations.)

Add two **hard filters**:
- Named entity and number retention threshold (e.g., ≥80% entity overlap),
- Semantic similarity threshold (e.g., cosine ≥ τ) to prevent topic drift.

Then pick argmax under a weighted sum:
\[
\text{score} = w_s\cdot \text{Style} + w_c\cdot \text{Content} + w_f\cdot \text{Fluency} - w_e\cdot \text{EditCost}
\]
where EditCost is normalized Levenshtein distance (token-level) to explicitly operationalize “minimal edit.”

4) **Ablations (must-have)**  
Make ablations clean and hypothesis-driven:
- Remove reranking (pick random candidate) → tests reranker utility.
- Remove content filter → tests meaning drift control.
- Remove edit-cost penalty → tests minimal-edit vs free rewrite.
- Evaluate each sarcasm strategy alone → which rhetorical device yields best style/content trade-off.

**Training schedule (realistic in 1 month)**  
- Week 1: Implement preprocessing + detectors + baselines; establish topic clusters and splits.  
- Week 2: Implement generator strategies and reranker; tune thresholds on dev.  
- Week 3: Run full evaluations + ablations; start human evaluation.  
- Week 4: Error analysis + finalize report artifacts.

**If compute/time becomes tight (explicit fallback plan)**  
Drop anything involving seq2seq fine-tuning. Keep:
- TF–IDF detector + one robust detector,
- Strategy templates + reranker,
- Human eval on a smaller set (but better-designed rubric).  
This is still a complete scientific project.

## Originality and evaluation plan

**Originality and “unique spin” ideas, with triage**

1) **Leakage-aware evaluation via topic-hard splits** — **must-have**  
This is your strongest “original analysis” lever: quantify how much performance (detector and rewriting evaluation) changes when topic overlap is removed. Motivated by known pitfalls in style-transfer evaluation where classifiers can exploit spurious cues. citeturn4search2turn4search18turn0search6

2) **Minimal-edit sarcasm transfer with an explicit edit budget** — **must-have**  
Make edit distance a first-class variable (0–1 edits, 2 edits, 3+ edits) and show curves: style strength vs semantic preservation vs human preference. This is easy to report, easy to analyze, and aligns with edit-based style transfer research emphasizing operations like insert/replace/delete. citeturn3search16turn0search17

3) **Sarcasm-mechanism tagging for your generated outputs** — **must-have (lightweight version)**  
Define 4–6 mechanisms (hyperbole, rhetorical question, ironic praise, quotation/scare quotes, absurd consequence, understatement). Tag each candidate deterministically by pattern rules (or a small classifier if ambitious). Then do meso-level analysis: which mechanism yields best trade-offs and least meaning drift. Rhetorical-device-aware sarcasm work supports treating sarcasm through the lens of rhetorical devices. citeturn2search2turn2search8

4) **Controllable sarcasm intensity** — **nice-to-have**  
Operationalize intensity as edit budget + strength of intensifiers + rhetorical tail presence. Unsupervised style transfer work on controllable intensity exists, which gives you a good “related work” justification for intensity controls even if you implement a simpler version. citeturn0search2turn0search18

5) **Detector-as-critic disagreement study** — **nice-to-have**  
Analyze when the detector says “sarcastic” but humans disagree (and vice versa). This gives you a strong discussion question and naturally ties to limitations of automated metrics, emphasized by meta-analyses of style transfer evaluation. citeturn4search2turn2search9

6) **Prototype-based candidate generation (retrieve-and-slot)** — **nice-to-have**  
Adds diversity while keeping candidates headline-like. Justify with retrieve-edit-rerank paradigms. citeturn3search0turn3search1

7) **External parallel inspiration from humorous headline micro-edits** — **too risky unless tightly scoped**  
The Humicroedit dataset provides real minimal edits for humor in headlines, which is methodologically adjacent and could inspire your strategy templates. citeturn11search3turn11search1  
But integrating it as training data risks scope creep (humor ≠ sarcasm), so keep it as **Related Work / inspiration** unless you have extra time.

**Evaluation plan (rubric-satisfying, multi-layered)**

**Why BLEU/ROUGE alone are insufficient**  
They measure n-gram overlap with references, but you have **no gold references**, and you explicitly want *style change* (so overlap can be low even when output is good). Style transfer evaluation literature repeatedly emphasizes separate measurement of style, content, and fluency, and warns against relying exclusively on unvalidated automatic metrics. citeturn4search2turn2search9

**Automatic metrics (must-have)**

- **Style-transfer success**
  - Target-style accuracy under sarcasm detector(s): report both standard-split and topic-hard detector scores to expose leakage sensitivity. citeturn4search2turn4search18  
  - Also report mean target-style probability (not just accuracy) to see calibration shifts.

- **Semantic preservation**
  - **BERTScore** between source and output (treating source as reference): captures contextual semantic similarity beyond exact overlap. citeturn3search3turn3search15  
  - Embedding cosine similarity (Sentence-BERT style).
  - **Entity/number preservation**: F1 overlap of named entities + exact match rate for numeric tokens (highly interpretable).

- **Fluency/naturalness**
  - LM perplexity (interpret with caution for headlines); complement with human judgments (below).  
  - Simple surface sanity: length distribution match to target corpus (headline-like constraint).

- **Edit-based / controllability metrics**
  - Token-level Levenshtein distance; % tokens kept; # insertions/ deletions / substitutions (your intensity knob).  
  - Optional: **SARI-style decomposition** conceptually matches edit-based evaluation (add/keep/delete), but classic SARI needs references. If you mention SARI, do so as inspiration and cite its definition, not as your primary metric. citeturn3search6turn3search14

**Human evaluation (must-have, but small and well-designed)**  
Design a 3-axis Likert (1–5) with clear anchors:
- Style success (sarcastic/non-sarcastic as intended),
- Meaning preservation (entities + event claim preserved),
- Fluency/headline-likeness (naturalness, brevity).

Use:
- ~100 examples per direction (200 total),
- 3 raters per example (can be classmates; if team members rate, disclose this as limitation),
- Report inter-annotator agreement (Krippendorff’s alpha or at least pairwise agreement).

**Error analysis (must-have)**
Pick 30–50 representative failures and categorize:
- **Meaning drift** (new topic/event),
- **Over-template** (repetitive sarcasm tags, unnatural),
- **False style success** (detector fooled; humans disagree),
- **S→N awkwardness** (non-sarcastic but implausible, or still “snarky”).  
Link each category back to one pipeline component and propose mitigation.

**Meso-level analyses (must-have; easy to report)**
Slice results by:
- presence of named entities,
- presence of numbers,
- headline length bins,
- topic clusters,
- sarcasm mechanism tag (hyperbole vs rhetorical question, etc.).  
This is the easiest way to produce “original analysis” that looks like research rather than “we tried models.”

**What a convincing evaluation table/figure setup looks like**
- One main table with rows = systems (Identity, Template, Main pipeline, and optional Prompt baseline), columns = StyleAcc (two detectors), BERTScore, EntityF1, EditDist, HumanStyle/HumanMeaning/HumanFluency.  
- One scatter plot: each point is an example; x = semantic similarity, y = style probability; plot different systems to show the trade-off surface.  
- One bar chart: per-mechanism human style success and meaning preservation.

**Related work touchpoints you can cite and borrow from**
- Style transfer evaluation standardization and dimensions. citeturn4search2turn4search18turn2search9  
- Edit-based / synthetic style transfer methods (MASKER; Levenshtein editing) as conceptual justification for “minimal edits.” citeturn0search17turn3search16  
- Content preservation emphasis in style transfer. citeturn0search6  
- Retrieve–edit–rerank framework as a principled generation design. citeturn3search0turn3search1  
- The dataset’s construction, scale, and fields. citeturn1view1turn1view2turn9view0

## Discussion questions, team plan, and execution plan

**Discussion questions for the final report (research-question style)**

1) **Which linguistic transformations most reliably create sarcasm in headlines?**  
- *Why interesting*: connects directly to sarcasm as a rhetorical/pragmatic phenomenon (hyperbole, rhetorical questions, evaluative reversal). citeturn2search8turn4search8  
- *How to test*: mechanism-tag outputs; compare style success and human ratings by mechanism; include examples.  
- *Evidence needed*: per-mechanism metrics + a qualitative table of “good vs bad” rewrites.

2) **Does minimal editing preserve meaning better than freer rewriting, and at what cost to style success?**  
- *Why interesting*: tests the central hypothesis of your framing (controllability). Edit-based style transfer research motivates edit operations explicitly. citeturn3search16turn0search17  
- *How to test*: create intensity tiers (≤1 edit, ≤2 edits, ≤3 edits); compare curves across metrics.  
- *Evidence*: trade-off plots; human meaning ratings per tier.

3) **How much do topic/publisher confounds inflate automatic “style success” scores?**  
- *Why interesting*: addresses a real experimental validity issue highlighted in style transfer evaluation meta-analysis. citeturn4search2turn4search18  
- *How to test*: evaluate with standard-split detector vs topic-hard detector; measure divergence.  
- *Evidence*: table with both detectors; examples where the in-domain detector is fooled.

4) **When the detector-as-critic disagrees with humans, what linguistic patterns cause the mismatch?**  
- *Why interesting*: pushes beyond metrics into analysis of sarcasm markers vs genuine sarcastic intent. citeturn4search2turn4search8  
- *How to test*: collect disagreement set; annotate error types (templatic tags, entity mismatch, subtlety).  
- *Evidence*: confusion breakdown + representative examples.

5) **Is sarcasm transfer asymmetric (N→S easier than S→N), and why?**  
- *Why interesting*: tests a linguistically meaningful asymmetry: sarcasm often relies on stance/incongruity; removing it while preserving proposition in satire can be awkward. citeturn2search8turn4search8  
- *How to test*: compare human ratings and semantic similarity distributions in each direction.  
- *Evidence*: direction-wise tables, plus qualitative discussion.

6) **Do certain topics (clusters) favor particular sarcasm mechanisms?**  
- *Why interesting*: links topical framing and rhetorical strategy selection.  
- *How to test*: topic clusters × mechanism tags; analyze conditional success rates.  
- *Evidence*: heatmap-like summary (can be small in the report, larger in appendix).

**Work division for 6 people (parallelizable, deliverable-driven)**

- **Member A: Corpus & leakage analysis lead**  
  Deliverables: topic clusters + topic-hard split; leakage experiments with baseline detectors; “top n-grams” interpretability write-up.

- **Member B: Sarcasm detector & metrics lead**  
  Deliverables: two detectors (standard-split and topic-hard); calibration plots; wrapper for scoring candidates; metric computation scripts (BERTScore, entity F1, edit distance). citeturn3search3turn3search15

- **Member C: Template strategy engineer**  
  Deliverables: curated strategy library (hyperbole/rhetorical/ironic praise); S→N neutralization rules; unit tests and examples per strategy grounded in sarcasm forms. citeturn2search8turn4search8

- **Member D: Candidate generation + reranking engineer**  
  Deliverables: generator that outputs N candidates per input; hard filters; weighted reranker; ablation switches.

- **Member E: Human evaluation & error analysis lead**  
  Deliverables: annotation rubric; rating UI or spreadsheet protocol; rater instructions; agreement stats; curated failure taxonomy and example set.

- **Member F: Report owner & reproducibility lead**  
  Deliverables: experiment log template; seed/config management; cleaned result tables/figures; final report integration and audit trail for any AI-tool usage.

Dependencies: detectors must exist before reranking tuning; topic clusters/splits should be ready early to avoid re-running everything late.

**Four-week execution plan (fast → medium → full cadence)**

**Week 1 (foundation + validity)**
- Must-do: load dataset, finalize task definition and meaning constraints; create standard split + topic-hard split; train baseline detectors; implement identity + template baselines.  
- Checkpoint: can you generate outputs for 50 examples end-to-end and compute style + similarity metrics?

**Week 2 (main system build)**
- Must-do: implement strategy-controlled candidate generation; implement reranker with style/content/edit scores; tune thresholds on dev.  
- Optional: prototype retrieval-based candidate generation (Direction B module).  
- Checkpoint: do you see a measurable increase in target-style detector score *without* a catastrophic drop in semantic similarity?

**Week 3 (experiments + ablations + human eval launch)**
- Must-do: run full eval on both splits; run ablations; start human evaluation protocol.  
- Optional: add intensity tiers + mechanism tagging analysis.  
- Checkpoint: you have the main results table and at least one convincing plot.

**Week 4 (analysis + writing)**
- Must-do: error analysis + disagreement study; finalize figures; write report; document reproducibility + limitations.  
- Cut first if behind: retrieval module, intensity extension, any seq2seq attempts. Keep the core pipeline + leakage-aware analysis + human eval.

## Report blueprint, risks, and final deliverables

**Final report blueprint (mapped to your required structure, staying within 8 pages)**  
Because your page limit is tight, treat the report as: one clean pipeline + the most insightful analyses.

- **Introduction**: define bidirectional rewriting; motivate sarcasm as rhetorical/pragmatic; state your precise operational definitions (style success + meaning constraints). Cite sarcasm forms and why headlines are hard. citeturn2search8turn4search8  
  Strong artifact: one running example showing edits + scores.

- **Related Work**: (i) sarcasm detection survey anchors; (ii) text style transfer + evaluation dimensions; (iii) retrieve/edit and edit-based style transfer. citeturn4search8turn4search2turn3search0turn0search17turn3search16  
  Strong artifact: short paragraph emphasizing evaluation pitfalls from meta-analysis. citeturn4search2

- **Corpus Analysis & Method**: dataset facts + leakage analysis + your strategy taxonomy; describe splits (standard + topic-hard). Cite dataset construction and fields. citeturn1view1turn1view2turn9view0  
  Strong artifact: table of corpus stats + plot of topic clusters (appendix if too big).

- **Experiments**: list systems (Identity, Template, Main); metrics (style/content/fluency/edit); human eval protocol. Cite BERTScore and evaluation framing. citeturn3search3turn4search2turn4search18  
  Strong artifact: one main results table + one trade-off plot.

- **Discussion**: answer 3–4 of the proposed research questions, with at least 2 being language-centered (mechanisms, minimal edits). Support claims with meso-level slices and error examples.

- **Conclusion**: summarize takeaways as “what worked and why,” include limitations (dataset confounds, weak pairing, imperfect fluency metrics, rater biases), and future work.

Common mistakes to avoid in 8 pages:
- spending too many lines on generic transformer descriptions,
- using only BLEU/ROUGE,
- presenting only aggregate numbers without micro/meso analysis,
- not clearly defining meaning preservation.

**Risk management (top risks + mitigations)**

- **Noisy pseudo-pairs / meaning drift**: avoid relying on pseudo-parallel training; instead enforce semantic thresholds and entity constraints in reranking; show drift cases explicitly. citeturn0search14turn0search6  
- **Overfitting to source style (publisher leakage)**: adopt topic-hard split and dual-detector evaluation; include interpretability of detector n-grams. citeturn4search2turn4search18  
- **Weak evaluation**: commit early to human eval + disagreement study; ground your metrics in style-transfer evaluation literature. citeturn4search2turn2search9  
- **Too much scope**: keep seq2seq as a “maybe”; the core project is a controllable pipeline + analysis.  
- **Human evaluation overhead**: small but well-designed sample; reuse the same examples across systems; record guidelines and agreement.  
- **Poor reproducibility**: single config system; fixed seeds; log every run; if any external AI tool is used, store prompt + parameters + outputs and disclose limitations.

**Deliverable-oriented answer**

- **Single best recommended project title**:  
  **Minimal-Edit Sarcasm Style Transfer for News Headlines Under Leakage-Aware Evaluation**

- **One-sentence problem statement**:  
  Given a news headline labeled sarcastic or non-sarcastic, generate a headline in the opposite style that preserves key semantic anchors (entities, predicate, numbers) and remains headline-like.

- **One-sentence contribution statement**:  
  We propose and analyze a controllable, multi-objective rewriting pipeline—with explicit sarcasm mechanisms and detector-guided reranking—and evaluate it under topic-hard splits to expose and mitigate dataset leakage.

- **Minimum viable project (MVP)**  
  Identity baseline + template baseline + main strategy-controlled reranking system; standard split + topic-hard split; automatic style/content/edit metrics; 200-example human evaluation; ablations for reranker and semantic constraint.

- **Best ambitious extension (only if time remains)**  
  Add prototype-based retrieve-and-slot candidate generation (Direction B module) and an explicit “sarcasm intensity” control curve with mechanism-tagged meso-analysis, keeping evaluation leakage-aware. citeturn3search0turn3search1turn0search18
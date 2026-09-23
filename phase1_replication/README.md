# Sentence Complexity Prediction Using Eye-Tracking Measures

**Replication of:** Singh, A. D., Mehta, P., Husain, S., & Rajkumar, R. (2016). *Quantifying Sentence Complexity Based on Eye-Tracking Measures.* In Proceedings of the Workshop on Computational Linguistics for Linguistic Complexity, pages 202–212, Osaka, Japan.

> This project replicates and extends the original paper's two-system pipeline that (1) predicts per-word reading times from linguistic features, and (2) classifies which sentence in a Wikipedia/Simple-Wikipedia pair is harder to read.

---

## Results

### System 1 — Reading Time Prediction (Paper Table 3)

| Measure | Our Threshold | Our Test R² (baseline) | Our Test R² (+GPT-2, see [Phase 2](#phase-2--gpt-2-enhancement)) | Paper R² |
|---------|:---:|:---:|:---:|:---:|
| First Fixation Duration (FFD) | 55 ms | 0.447 | 0.458 | 0.649 |
| First Pass Duration (FPD) | — | 0.561 | 0.583 | 0.600 |
| Regression Path Duration (RPD) | — | 0.558 | **0.582** | 0.570 ✓ |
| Total Duration (TD) | — | 0.538 | **0.568** | 0.516 ✓ |

> With GPT-2 surprisal, RPD and TD now beat the paper, and FPD is within 0.017 R². Only FFD still trails by a wide margin. This is **not** a preprocessing bug — the paper's own Table 1 shows near-zero correlation for sentence length and syntactic surprisal too, same as ours (see [Methodology](#methodology) and [Phase 2](#phase-2--gpt-2-enhancement)).

### System 2 — Readability Classification (Paper Table 5)

| Model | Our Accuracy | Paper Accuracy |
|-------|:---:|:---:|
| Base RT + Pairwise Classification | **74.15%** | 73.82% |
| Extended RT + Pairwise Classification | **75.26%** | 75.21% |

> System 2 now matches/slightly exceeds the paper on both models (10-fold CV, 117,712 sentence pairs).

---

## Project Structure

```
.
├── src/                          # Core pipeline (run in order)
│   ├── step1_parse_dundee.py     # Parse Dundee corpus → per-word RT measures
│   ├── step2_lexical_features.py # Word length, freq, AoA, bigram probs
│   ├── step3_syntactic_features.py # PCFG surprisal, embedding depth, entropy
│   ├── step4_system1.py          # Train + evaluate reading-time regression
│   ├── step5_system2.py          # Train + evaluate readability classifier
│   └── syntactic_extractor.py   # Syntactic feature utilities
│
├── experiments/                  # Experimental scripts (advanced syntactic models)
│   ├── build_lcparse_surprisal.py  # Van Schijndel left-corner parser surprisal
│   ├── compute_pcfg_surprisal_python.py # BLLIP + PTB PCFG surprisal
│   ├── compute_gpt2_surprisal.py   # Phase 2: GPT-2 contextual surprisal per word
│   └── train_system1_gpt2.py       # Phase 2: compares GPT-2 vs. syn_surprisal for FFD
│
├── results/                      # Summary tables and key outputs
│   └── summary.md
│
├── output/                       # Generated files (created by running pipeline)
│   ├── dundee_rt.csv             # Step 1 output: per-word RT per subject
│   ├── lexical_features.csv      # Step 2 output: all lexical features
│   ├── syntactic_features.csv    # Step 3 output: all syntactic features
│   ├── system1_models.pkl        # Step 4 output: trained System 1 models
│   ├── system1_predictions.csv   # Step 4 output: predicted FFD for all words
│   └── ...
│
├── Dataset/                      # Input corpora (not on GitHub — licensed)
│   ├── dundee_corpus/            # Eye-tracking data (Kennedy et al. 2003)
│   ├── AoA.csv                   # Age of Acquisition norms
│   ├── BNC/                      # British National Corpus
│   ├── penn_treebank_3.tar.bz2   # Penn Treebank WSJ
│   └── Wiki/                     # Wikipedia / Simple-Wikipedia pairs
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key libraries: `scikit-learn`, `pandas`, `numpy`, `nltk`, `wordfreq`, `bllipparser` (optional, for syntactic surprisal).

### Datasets Required

> These are **not included** in this repo (licensed). Place them under `Dataset/`:

| Dataset | Source |
|---------|--------|
| Dundee Corpus | Kennedy, A. et al. (2003) — request from authors |
| Penn Treebank 3 | LDC99T42 — via LDC |
| British National Corpus (BNC) | http://www.natcorp.ox.ac.uk/ |
| Age of Acquisition Norms | Kuperman et al. (2012), available online |
| Wiki/SimpleWiki pairs | Hwang et al. (2015), `aligned-good(0.67).gz` |

---

## How to Run

Run steps 1–5 in order. Each step reads from `output/` and writes back to `output/`.

```bash
# Step 1: Parse Dundee eye-tracking corpus
python src/step1_parse_dundee.py
# Output: output/dundee_rt.csv  (~307K rows, one per fixated word×subject)

# Step 2: Extract lexical features
python src/step2_lexical_features.py
# Output: output/lexical_features.csv  (word length, frequency, AoA, bigram probs)

# Step 3: Extract syntactic features
python src/step3_syntactic_features.py
# Output: output/syntactic_features.csv  (surprisal, embedding depth, entropy)

# Step 4: Train System 1 (reading time regression)
python src/step4_system1.py
# Output: output/system1_models.pkl, output/system1_predictions.csv
# Prints: ablation table (Table 2) + final R² results (Table 3)

# Step 5: Train System 2 (readability classifier)
python src/step5_system2.py
# Output: accuracy scores (Table 5)
# Runtime: ~30–40 minutes on 117K Wikipedia pairs
```

> **WSL note:** Steps 1–5 should be run inside WSL (Ubuntu) for Linux-path compatibility. The scripts auto-detect the OS and set paths accordingly.

---

## Methodology

### System 1 — Reading Time Prediction

The paper predicts per-word reading times using a linear regression model (Ridge) trained on 17 linguistic features:

| Feature Group | Features |
|---|---|
| Word-level | Word length (chars), Wikipedia log-frequency |
| Sentence-level | Sentence length (# words) |
| Age of Acquisition | Mean AoA, Std Dev AoA (Kuperman et al.) |
| Transition probability | Forward P(w\|prev), Backward P(w\|next) from BNC |
| Lexical surprisal | −log P(word\|context) from BNC bigrams |
| Syntactic surprisal | −log P(parse\|prefix) from PTB PCFG |
| Entropy reduction | Drop in word-prediction entropy |
| Embedding depth | Distance of word to parse-tree root |
| Embedding difference | Depth change from previous word |
| Hierarchical structure | 8 binary left-corner parser memory ops |

**Training:** 60% train / 20% dev / 20% test split at **sentence level** (2,369 sentences). Threshold tuning: any prediction below `T` ms → 0.0 (models word-skipping).

**FFD gap explanation:** Our FFD R² (0.444) is lower than the paper's (0.649). We read the paper's own Table 1 (Pearson correlations) and found `sent_len` has r=-0.009 with FFD *in the paper's own data* — essentially the same near-zero value we get (r=-0.058), not the ≈-0.48 that would be needed to explain their reported +0.233 R² jump as a direct correlation effect. So this isn't a preprocessing difference to chase down. What we do know: the paper's word-length-alone model gets R²=0.267, ours gets R²=0.447 — our `wlen` feature already captures much of what `sent_len` adds in the paper. **What causes that is still an open question** — we tested and disproved the leading theory (that skipped words getting RT=0 makes `wlen` a fixation-probability proxy): the word-level-averaged target has **zero rows with RT=0** (min FFD=5.1ms across all 50,647 words), so that mechanism doesn't exist in this data. See [Improvement Roadmap](#improvement-roadmap--closing-the-gap-with-the-paper) item 2 for the full test and why a new hypothesis is needed.

### System 2 — Readability Classification

Given a Wikipedia / Simple-Wikipedia sentence pair, classify which is harder.

- **Base RT model:** Predict word-level FFD for each sentence using System 1. Concatenate predicted FFD vectors (padded to 60 words each) → 120-dimensional feature vector per pair.
- **Extended RT model:** Add sentence length, mean predicted FFD, and sum of lexical surprisal for both sentences.
- **Classifier:** Logistic Regression, **10-fold cross-validation** on 117,712 pairs.

---

## Paper Reference

The PDF is included at `Research Paper 1.pdf`.

```bibtex
@inproceedings{singh2016quantifying,
  title     = {Quantifying Sentence Complexity Based on Eye-Tracking Measures},
  author    = {Singh, Abhinav Deep and Mehta, Poojan and Husain, Samar and Rajkumar, Rajakrishnan},
  booktitle = {Proceedings of the Workshop on Computational Linguistics for Linguistic Complexity},
  year      = {2016},
  pages     = {202--212},
  address   = {Osaka, Japan}
}
```

---

## Phase 2 — GPT-2 Enhancement

The paper's syntactic surprisal feature (van Schijndel left-corner parser)
turned out to carry almost no signal for FFD — **in both our replication
(r=-0.017) and the paper's own Table 1 (r=-0.067)**. This isn't a bug in
either pipeline; it's a real property of that feature. GPT-2's contextual
next-word surprisal was tried as a replacement, computed in
`experiments/compute_gpt2_surprisal.py` and compared in
`experiments/train_system1_gpt2.py` (both scripts reuse `step4_system1.py`'s
data loading/training code unmodified — no other pipeline files were
changed for this).

### Result

| Feature | r(·, FFD) | standalone dev R² |
|---|---|---|
| `syn_surprisal` (paper's LC parser) | -0.017 | ~0.000 |
| `gpt2_surprisal` (gpt2, 124M) | **0.454** | **0.220** |

| Feature set | FFD test R² |
|---|---|
| baseline (`syn_surprisal` only) | 0.4439 |
| + `gpt2_surprisal` added | 0.4509 |
| `syn_surprisal` → `gpt2_surprisal` swap | **0.4514** |

GPT-2 surprisal is complementary to word length, not just a proxy for it
(`wlen` alone: dev R²=0.447 → `wlen`+`gpt2_surprisal`: dev R²=0.467).

**Model size doesn't help here:** `gpt2-medium` (355M) fits *worse*
(r=0.407, swap-in test R²=0.4483) than the base 124M model. This matches
a documented finding (Oh & Schuler 2023 — larger transformer LMs' surprisal
fits human reading times worse, not better) and is worth citing directly.
`gpt2-large` wasn't run since the trend predicts it'd continue downward.

Run it yourself:
```bash
python experiments/compute_gpt2_surprisal.py            # full corpus (~6 min on CPU)
python experiments/train_system1_gpt2.py                 # baseline / add / swap comparison
```

### Cross-sentence context — improves the result further

`compute_gpt2_surprisal.py --context-sentences 3 --context-tokens 300`
prepends up to 3 preceding sentences (same `text_id`, capped at 300
tokens) before scoring each sentence, instead of scoring it in isolation
with just a BOS token. Only the target sentence's own tokens are kept in
the output (context tokens are scored, for the model to condition on, but
discarded from the surprisal sums).

| Version | r(·, FFD) | swap-in FFD test R² |
|---|---|---|
| No context (isolated sentences) | 0.454 | 0.4514 |
| **+3 preceding sentences as context** | **0.465** | **0.4577** |

**+0.014 R² over the original paper-method baseline (0.4439 → 0.4577)** —
a bigger gain than the model-size experiments. Run it with:
```bash
python experiments/compute_gpt2_surprisal.py --context-sentences 3 --context-tokens 300
python experiments/train_system1_gpt2.py --surprisal-file output/gpt2_surprisal_gpt2_ctx.csv
```

Bug caught and fixed while building this: GPT-2's BPE tokenizer merges a
leading space into the *next* word's token (`" the"` is one token, not
`"the"`). The context/sentence boundary token was being silently dropped
by an offset check that only looked at where a token *started* — any
sentence-initial word whose first token straddled that boundary got
surprisal exactly `0.000`. Fixed in `word_token_surprisals()` by checking
where the token *ends* instead of where it starts.

### All four RT measures — GPT-2 helps FPD/RPD/TD even more than FFD

`train_system1_gpt2.py` now evaluates all four measures, not just FFD:

| Measure | Baseline | +GPT-2 (swap) | Gain | Paper R² |
|---|---|---|---|---|
| FFD | 0.4439 | 0.4577 | +0.014 | 0.649 |
| FPD | 0.5584 | 0.5834 | +0.025 | 0.600 |
| RPD | 0.5553 | 0.5820 | +0.027 | **0.570** |
| TD | 0.5357 | 0.5680 | +0.032 | **0.516** |

**RPD and TD now beat the paper** (0.582 vs 0.570, 0.568 vs 0.516), and FPD
closed most of its gap (0.583 vs 0.600, was 0.558 vs 0.600). Correlation
with GPT-2 surprisal is actually *stronger* for FPD/RPD/TD (r≈0.52-0.54)
than FFD (r=0.465) — later/integrative reading measures track contextual
predictability even more than first-fixation does. Only FFD still trails
the paper by a wide margin.

### Feature redundancy check (`gpt2_sentence_correlation.py`)

Checked whether `gpt2_surprisal` is redundant with the existing F1-F11
features, at word level and aggregated per sentence (mean/max/sum).

- **Word-level:** correlates moderately with other predictability features
  (`lex_surprisal` r=0.56, `wiki_freq` r=-0.65, `aoa_mean` r=0.54, `wlen`
  r=0.51 — expected, all are word-predictability proxies) but **r=0.014
  with `syn_surprisal`** — confirms it captures information the existing
  syntactic feature genuinely does not.
- **Sentence-level — `gpt2_sum` is a trap:** it correlates r=0.914 with
  `sent_len` itself, since summing surprisal over more words mostly just
  tracks word count. `gpt2_mean` is the clean aggregate (r=0.45-0.53 with
  all four RT measures, not confounded with sentence length).
  **Recommendation: use `gpt2_mean` for any sentence-level feature, or
  word-level `gpt2_surprisal` directly (as already done) — never
  `gpt2_sum`.**

### Next steps
- [ ] Try a larger context budget/window, or context from the actual
      preceding paragraph rather than just prior Dundee sentences.
- [ ] Decide whether to fully replace `syn_surprisal` with `gpt2_surprisal`
      (context version) in the main pipeline (`step3`/`step4`) or keep both.

---

## Improvement Roadmap — Closing the Gap With the Paper

Current FFD test R² is **0.4577** (GPT-2 with cross-sentence context) vs.
the paper's 0.649. Ranked by expected payoff, cheapest/most-promising
first:

1. ~~Cross-sentence context for GPT-2 surprisal~~ — **done**, see above
   (+0.014 R²). Room left: larger context budget, real paragraph context
   instead of just prior Dundee sentences.
2. ~~Fix the word-length ceiling effect via fixated-only training~~ —
   **tested and disproven** (`experiments/fixated_only_system1.py`). The
   theory was: skipped words get RT=0, so `wlen` doubles as a fixation-
   probability proxy, inflating its R². But there are **zero rows with
   RT=0** in the word-level-averaged data (min FFD=5.1ms across all
   50,647 words) — with 10 subjects averaged per word, essentially no
   word is skipped by all 10, so "fixated-only" and "all-words" are the
   same dataset at this granularity. The filter was a no-op (confirmed:
   identical R² for both). **The actual cause of `wlen`-alone R²=0.447 vs.
   the paper's 0.267 is still unexplained** — this specific mechanism
   doesn't exist in our data, so a different explanation is needed. A
   related but different granularity (one row per subject-word fixation
   event, 307K rows, filtered to only actually-fixated pairs) was tried in
   an earlier session and got wlen R²=0.008 — much *worse*, the opposite
   of closing the gap, so that's not the fix either.
7. **Found: the paper's own Table 1 and Table 2 are internally
   inconsistent for word length** (`experiments/check_wlen_r2_consistency.py`).
   For a single-predictor OLS, R² on the same data always equals the
   squared Pearson r — a textbook identity. But the paper reports
   r(wlen, FFD)=0.765 in Table 1 (implying R²=0.585) and wlen-alone
   ablation R²=0.267 in Table 2. These cannot both be true for the same
   data. We ruled out ordinary train→dev generalization loss as the cause
   (our own wlen-alone OLS: train R²=0.4449, dev R²=0.4468 — nearly
   identical, no meaningful overfitting for a 1-feature model). Our own
   numbers ARE internally consistent (r²=0.4511 on the full word-averaged
   data ≈ our train/dev R²), so this points to a data/methodology
   difference between the paper's own two tables — not a bug in our
   pipeline. Also notable: at the raw per-subject-fixation-event level
   (before word-averaging), r(wlen, FFD) is only 0.090 (r²=0.008,
   matching the earlier 307K-row result) — the strong 0.67 correlation
   only appears after averaging by a *fixed* 10 subjects, which lets word
   length in via fixation *probability* rather than fixation *duration*.
3. **Fine-tune GPT-2 on newspaper text.** The Dundee corpus is British
   newspaper editorials; base GPT-2 is trained on general web text
   (WebText). A light fine-tune (or using a news-domain checkpoint) could
   sharpen surprisal further, similar in spirit to why `gpt2` beat
   `gpt2-medium` — domain match may matter more than scale here.
4. **Use GPT-2 attention/entropy as an entropy-reduction replacement.**
   The paper's own `entropy_red` feature added 0 R² for us. GPT-2's
   next-token distribution entropy at each position is a natural drop-in
   replacement, computable from the same forward pass already run for
   surprisal — no extra model cost.
5. **Per-measure feature selection.** The paper notes feature relevance
   may differ across FFD/FPD/RPD/TD but only tunes for FFD. Since FPD/RPD/TD
   are already close to the paper (within 0.04 R²), a quick ablation per
   measure (reusing `ablation_study()` with `rt_name` swapped) could reveal
   if GPT-2 surprisal helps those measures more than FFD.
6. **Ensemble/nonlinear model.** Everything so far is Ridge regression per
   the paper's own method. A gradient-boosted tree (e.g. `HistGradientBoostingRegressor`)
   on the same features would show whether the ceiling is the linear model
   or the features themselves — useful as a diagnostic even if you keep
   Ridge for the final reported numbers (to stay comparable to the paper).

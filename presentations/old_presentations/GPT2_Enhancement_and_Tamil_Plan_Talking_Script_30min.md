# The 30-Minute Talk
**Progress Meeting Script — Phase 2 (GPT-2 Enhancement) + Phase 3 (Tamil) Proposal**
Bhagyalaxmi Pandiyan · Deck: `GPT2_Enhancement_and_Tamil_Plan.pptx`

> Open this file in VS Code and press **Ctrl+Shift+V** (or click the preview icon, top-right) to render it nicely. Keep it open in preview mode on a second monitor/half-screen while you present from the other half.

---

## Why this version is longer than the 10-min script

The original script (`GPT2_Enhancement_and_Tamil_Plan_Talking_Script.txt`) ran ~8–9 min talking, 10–12 with demos, and deliberately *skipped* live code walkthroughs on three slides to stay tight. For 30 minutes:

1. **Part A is new** — a full feature-by-feature walkthrough of all 17 features using `feature_demo.py`, with real values.
2. **Part B is un-compressed** — the three previously-skipped code walkthroughs are back in full.
3. **A 12-question Q&A appendix** is added at the end.

**Budget:** Part A (~10 min) + Part B (~13 min) + Part C (~5 min) + closing/buffer (~2 min) ≈ 30 min.
**If short on time:** cut Part C first. **Never** rush Part A — that's what's actually being evaluated.

**Before you start, have open:**
- `src/feature_demo.py`
- `experiments/compute_gpt2_surprisal.py`
- `experiments/train_system1_gpt2.py`
- `experiments/fixated_only_system1.py`
- `experiments/gpt2_sentence_correlation.py`
- `output/lexical_features.csv` and `output/syntactic_features.csv` — visible the whole time
- A terminal, `cwd = phase1_replication/`
- `README.md` as fallback if a live run fails

---

## Slide 1: Title
**SAY:** *"Good [morning/afternoon]. I want to use today's time a bit differently than last time — I'll start by walking through exactly how each of the 17 features in my model is computed, with real numbers, since that's the part I want to make sure I can explain confidently. Then I'll cover what's new since we last spoke: the GPT-2 results, and a proposal for Phase 3 on Tamil."*
*(~35 sec, no code)*

---

# PART A — Feature-by-Feature Walkthrough *(new)*

**SAY:** *"There are 17 features feeding System 1 — 6 lexical, 11 syntactic/structural. Rather than describe them abstractly, I'll run the script that computes all of them from raw data live."*

**RUN:** `python src/feature_demo.py`

### A1 — Raw Dundee Corpus
**SAY:** *"Each row is one word a subject actually fixated on — WORD, TEXT, WNUM, FDUR (first-fixation duration, ms). A skipped word has no row for that subject — matters later for a bug I ruled out."*
📍 **Point at:** terminal table under `"First 8 rows (Text 1, Subject A)"` → [`feature_demo.py:42-44`](../src/feature_demo.py#L42-L44)
📍 Then: `"Key insight..."` line → [`feature_demo.py:46`](../src/feature_demo.py#L46)

### A2 — From fixations to 4 reading-time measures
**SAY:** *"1P gives FFD directly. 2P gives FPD (before any regression), RPD (until the eye passes the word), TD (every fixation). I sum all 10 subjects' values per word — a skipper contributes zero — and divide by 10."*
📍 **Point at:** `"DATASET 1P → FFD"` / `"DATASET 2P → ..."` block → [`feature_demo.py:55-73`](../src/feature_demo.py#L55-L73)
📍 Then: per-subject table → [`feature_demo.py:77`](../src/feature_demo.py#L77) header, values at [`line 115`](../src/feature_demo.py#L115)
📍 Then: `dundee_rt.csv` sample, `ffd/fpd/rpd/td` columns → [`feature_demo.py:118-120`](../src/feature_demo.py#L118-L120)

### A3 — 6 Lexical Features

| # | Feature | SAY | 📍 Point at |
|---|---|---|---|
| 1 | **wlen** | Character count from Dundee's WLEN column. `'tourists' → 8`. One of the most predictive features overall. | [`feature_demo.py:135-139`](../src/feature_demo.py#L135-L139) |
| 2 | **sent_len** | Word count of the sentence, from Dundee's own sentence files. | [`feature_demo.py:146-149`](../src/feature_demo.py#L146-L149) |
| 3 | **wiki_freq** | `log(freq_per_million + 1)` via the `wordfreq` library — standard log-frequency transform. | [`feature_demo.py:157-160`](../src/feature_demo.py#L157-L160) — contrast `'the'` vs `'enticed'` |
| 4–5 | **aoa_mean, aoa_std** | Kuperman et al. (2012) norms — average age a word is learned. Missing words → corpus mean. | [`feature_demo.py:171-177`](../src/feature_demo.py#L171-L177) |
| 6–7 | **fwd_prob, bwd_prob** | BNC (100M words) bigram transition probability, forward/backward, add-1 smoothing. | [`feature_demo.py:186-188`](../src/feature_demo.py#L186-L188) |

*(~2 min)*

### A4 — 11 Syntactic/Structural Features

| # | Feature | SAY | 📍 Point at |
|---|---|---|---|
| 8–9 | **lex_surprisal, syn_surprisal** | Lexical = `−log P(word\|prev)` from BNC. Syntactic = left-corner parser over 39,832 PTB trees; POS-tag bigram surprisal. `'tourists'` after `'Are'` → NNS after VBP, unusual → high. | Annotation table → [`feature_demo.py:214-215`](../src/feature_demo.py#L214-L215); explanation → [`lines 218-220`](../src/feature_demo.py#L218-L220); values → [`lines 223-224`](../src/feature_demo.py#L223-L224) |
| 10 | **entropy_red** | Drop in uncertainty about next word (after "United" → only "States"/"Nations" likely), clipped at 0. | Explanation → [`lines 230-231`](../src/feature_demo.py#L230-L231); values → [`line 233`](../src/feature_demo.py#L233) |
| 11–12 | **emb_depth, emb_diff** | Depth = hops from sentence root in the dependency parse. Diff = change from previous word. | [`feature_demo.py:260-263`](../src/feature_demo.py#L260-L263) — trace one word's `head` number up to the row above |
| 13–17 | **h1 … h8** | Yes/no flags on depth transitions — head changed, depth up/down, is-root, big jump, unchanged. Captures local *shape*. | Definitions → [`lines 272-278`](../src/feature_demo.py#L272-L278); values → [`line 280`](../src/feature_demo.py#L280) |

*(~2 min)*

### A5 — From features to the R² you already know
**SAY:** *"17 features + 4 RT measures merge into one ~50,647-row table. Split by sentence, 60/20/20. Scale → Ridge → tune one threshold on dev only, apply unchanged on test. That's FFD 0.445, FPD 0.559, RPD 0.556, TD 0.536 — vs. the paper's 0.649/0.600/0.570/0.516."*
📍 **Point at:** `"Merged shape:"` → [`feature_demo.py:308`](../src/feature_demo.py#L308)
📍 Then: **FINAL RESULTS** table → [`feature_demo.py:361-369`](../src/feature_demo.py#L361-L369) — finger down `Test R²` vs `Paper R²` columns

**Part A total: ~10 min**

---

# PART B — Phase 2: GPT-2 Enhancement *(un-compressed)*

### Slide 2: Recap *(~30 sec)*
**SAY:** *"System 1 predicts 4 RT measures per word from those 17 features. System 2 uses System 1's predictions to classify which of two parallel sentences — Wikipedia vs. Simple Wikipedia — is harder to read. Both replicated on Dundee: 50,647 words, 2,378 sentences, 10 subjects."*

### Slide 3: Why We Turned to GPT-2 *(~1 min)*
**SAY:** *"syn_surprisal carries almost no signal for FFD: r = −0.017. Cross-checked against the paper's own Table 1 — they report −0.067. Same sign, same magnitude — not a bug, a real property of the feature."*
**RUN:** `python experiments/train_system1_gpt2.py`
📍 **Point at:** [`train_system1_gpt2.py:79-80`](../experiments/train_system1_gpt2.py#L79-L80) — the `for col in ["syn_surprisal", "gpt2_surprisal"]` loop
📍 Then terminal: `r(syn_surprisal, ffd) = -0.017` when it prints

### Slide 4: Method *(~80 sec)*
**SAY:** *"Compute GPT-2 surprisal per word, up to 3 preceding sentences of context, swap for syn_surprisal in the same 19-feature model. Same split, same Ridge, nothing else touched."*

| Say | 📍 Point at |
|---|---|
| "sentence split into words, context prepended" | [`compute_gpt2_surprisal.py:74-75`](../experiments/compute_gpt2_surprisal.py#L74-L75) |
| "GPT-2 forward pass, negative log probability" | [`compute_gpt2_surprisal.py:84-91`](../experiments/compute_gpt2_surprisal.py#L84-L91) — line 91 is the literal formula |
| "sum subtoken surprisal using token END position — a real bug I caught" | [`compute_gpt2_surprisal.py:106-116`](../experiments/compute_gpt2_surprisal.py#L106-L116), specifically the comment at [`110-113`](../experiments/compute_gpt2_surprisal.py#L110-L113) |
| "add_gpt2_feature... the three variants" | [`train_system1_gpt2.py:27-31`](../experiments/train_system1_gpt2.py#L27-L31) then [`59-63`](../experiments/train_system1_gpt2.py#L59-L63) |

### Slide 5: Results (Table 3) — *centerpiece* *(~55 sec + live run)*
**SAY:** *"RPD and TD beat the paper outright. FPD within 0.017. FFD trails but improved: 0.444 → 0.458. Makes sense — FFD is one brief first glance; the others integrate more reading behavior."*
**RUN:** `python experiments/train_system1_gpt2.py --surprisal-file output/gpt2_surprisal_gpt2_ctx.csv`
📍 **Point at:** printed rows from [`train_system1_gpt2.py:65,72-73`](../experiments/train_system1_gpt2.py#L65) — FFD → FPD → RPD → TD, left to right across `baseline / gpt2_added / gpt2_swap`

### Slide 6: Feature Quality Checks *(~75 sec + live run)*
**SAY:** *"Correlation with syn_surprisal is 0.014 — genuinely new signal. Moderate correlation with lex_surprisal/frequency is expected. The trap: summing GPT-2 surprisal over a sentence correlates 0.914 with sentence length — just word-counting. Used the mean instead."*
**RUN:** `python experiments/gpt2_sentence_correlation.py`

| Say | 📍 Point at |
|---|---|
| "correlation with syn_surprisal is 0.014" | `syn_surprisal` row under `"Word-level: r(gpt2_surprisal, feature)"` → [`gpt2_sentence_correlation.py:40-44`](../experiments/gpt2_sentence_correlation.py#L40-L44) |
| "correlates 0.914 with sentence length" | `sent_len` row, `gpt2_sum` column under `"Sentence-level..."` → [`gpt2_sentence_correlation.py:58-68`](../experiments/gpt2_sentence_correlation.py#L58-L68) |
| "I used the mean instead" | Same row, `gpt2_mean` column right next to it |

### Slide 7: Ruling Out a Wrong Explanation *(~100 sec + live run)*
**SAY:** *"Theory: skipped words get RT=0, so word length might secretly proxy 'gets fixated at all.' Tested directly — filtered to fixated-only words, retrained. Zero difference, R² byte-identical. Minimum FFD across all 50,647 words is 5.1 ms — essentially no word is exactly zero. Theory disproven, not just unconfirmed."*
**RUN:** `python experiments/fixated_only_system1.py`
📍 **Point at:** the mask → [`fixated_only_system1.py:60`](../experiments/fixated_only_system1.py#L60)
📍 Then: printed comparison row → [`fixated_only_system1.py:65-66`](../experiments/fixated_only_system1.py#L65-L66)

### Slide 8: System 2 Reconfirmed *(~1 min)*
**SAY:** *"System 2 still beats the paper on all four numbers. The FFD gap shows up as only a 1.4–2.6 point accuracy gap downstream. I'd consider System 2 effectively closed."*
📍 **Point at:** System 2 table in `README.md` — don't re-run the full 117K-pair pipeline live. If asked for mechanism: `feature_demo.py` Part 6 ([`lines 375-487`](../src/feature_demo.py#L375-L487)).

**Part B total: ~13 min**

---

# PART C — Phase 3 Proposal: Extending to Tamil

### Slide 9: Transition *(~24 sec)*
**SAY:** *"That's Phase 2 complete. Now I want to propose Phase 3: taking this pipeline to Tamil — no code or results yet, this is a proposal."*

### Slide 10: Why Tamil? *(~60 sec)*
**SAY:** *"No Tamil eye-tracking-while-reading corpus exists — even MultiplEYE doesn't cover it. No Tamil readability dataset in DravidianLangTech either. Closest precedent: a 2023 ML readability model for Gujarati. This was my own targeted check, not a systematic review."*

### Slide 11: The New Idea *(~90 sec)*
**SAY:** *"Word length is already unstable even in English — the Table 1 vs Table 2 inconsistency. Tamil is agglutinative — one word carries what English spreads across a clause. Plan: use ThamizhiMorph to build a morpheme-count complexity metric, then drive/evaluate an LLM-based Tamil simplifier against human judgments."*
**RUN (optional):** `python experiments/check_wlen_r2_consistency.py`
📍 **Point at:** **"Comparison with the paper"** table → [`check_wlen_r2_consistency.py:77-84`](../experiments/check_wlen_r2_consistency.py#L77-L84) — `Paper Table 1 (r²=0.585)` vs `Paper Table 2 (R²=0.267)`, then your own consistent rows below
📍 Then: conclusion lines → [`check_wlen_r2_consistency.py:86-91`](../experiments/check_wlen_r2_consistency.py#L86-L91)

### Slide 12: Why This Direction *(~70 sec)*
**SAY:** *"Reuses the GPT-2 pipeline — Tamil GPT-2 models already exist. Every missing piece has a real open-source option. Two deliverables: a testable linguistic claim, and a working simplifier. Bigger than Tamil — applies to Turkish, Finnish, Korean too."*

**Part C total: ~5 min**

---

## Closing *(~24 sec)*
**SAY:** *"That's Phase 2 complete, explained down to the feature level, and Phase 3 proposed with a concrete first step. Happy to re-run any script, or go deeper into whatever's most useful with the remaining time."*

**Grand total: ~28-32 minutes** including live-run and interruption buffer.
- **Running short?** Cut Part C first.
- **Running long?** Skip re-running `gpt2_sentence_correlation.py` live — just state the two numbers.

---

## Appendix — Anticipated Q&A

| # | Question | Answer |
|---|---|---|
| 1 | Why Ridge, not a neural net / gradient boosting? | Stays comparable to the paper's model choice; more interpretable for 17-19 features on ~50K rows. |
| 2 | Could the GPT-2 gain be memorized training data? | Dundee is 1990s British newspaper text; surprisal is local-context, not retrieval. Sentence-level split prevents the split itself from leaking. |
| 3 | Why cap context at 3 sentences? | Practical tradeoff, not an exhaustive sweep — easy follow-up to show a sensitivity curve. |
| 4 | Combined both surprisal features instead of swapping? | Yes — the `gpt2_added` variant. *(Fill in actual numbers before the meeting.)* |
| 5 | Is threshold-tuning overfitting to dev? | Threshold chosen only on dev, applied unchanged to held-out test. |
| 6 | Where would Tamil ground-truth labels come from? | Human ratings on sentence pairs, or a Tamil parallel plain/simplified resource — open question for his input. |
| 7 | Isn't morpheme count just another proxy? | Fair — a hypothesis, tested the same way word length was tested here. |
| 8 | Biggest technical risk? | ThamizhiMorph's coverage/accuracy on real-world text — validate on a sample first. |
| 9 | Why not a lower-risk language first? | Turkish/Finnish have more tooling; chose Tamil for the literature gap + personal validation ability. |
| 10 | How long will Phase 3 take? | *(Fill in your real estimate before the meeting.)* |
| 11 | Nothing left to learn from FFD lagging? | Some headroom, but diminishing returns vs. time on the new Tamil territory. |
| 12 | What would make you abandon Tamil? | Unreliable analyzer output or too few human judgments — scope down to a descriptive study instead. |

---

*Fallback (tight 10-min version): `GPT2_Enhancement_and_Tamil_Plan_Talking_Script.txt`*

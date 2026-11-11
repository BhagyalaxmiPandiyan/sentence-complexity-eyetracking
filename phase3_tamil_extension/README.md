# Phase 3 — Tamil Morphology-Aware Complexity + LLM Simplification

Implements `Tamil_Readability_Thesis_Proposal.md`. Hypothesis: in an agglutinative
language like Tamil, word length is the wrong unit of difficulty and morphemes are the
right one.

## Status (2026-11-11)

| Phase | What | Status |
|---|---|---|
| A | Difficulty metric: word-level (A), morphology (B), syntax (C) features, tested on 2 graded datasets | ✅ done |
| B | LLM simplification of 60 complex sentences, scored by both metrics | ✅ 1 system (Claude Opus 5.5); a 2nd free web-chat system is optional |
| C | Native-speaker rating study, compared against both metrics | ✅ done (12 raters; results in `results/human_eval_summary.md`) |
| C2 | Webcam eye-tracking pilot (System 1 on Tamil reading times) | ✅ done (5 readers kept, updated 2026-11-11; `results/eyetracking_summary.md`) |
| D | Write-up | ✅ full draft incl. eye-tracking in [`Phase3_Research_Paper.md`](Phase3_Research_Paper.md) (+ PDF); author names still to fill in |

## Pipeline (`src/`)

Each script reads a CSV and writes a CSV to `output/` (same convention as `phase1_replication/src`).

| Script | Phase | What it does | Output |
|---|---|---|---|
| `step0_fetch_textbooks.py` | data | Downloads the TN state Tamil textbooks, Class 3–12 | `Dataset/graded_textbooks/` |
| `step1_extract_sentences.py` | data | Prose sentences only, font repair, filters | `output/graded_sentences.csv` (2,351) |
| `step2_build_pairs.py` | data | Pairs ≥3 grades apart, page-level split | `output/graded_pairs.csv` (8,004) |
| `step3_textbook_features.py` | A | 12 features per sentence (Stanza, IndicNLP Morfessor, Tamil GPT-2, wordfreq); ~40 min on CPU | `output/sentence_features.csv` |
| `step4_storyweaver_data.py` | A | 2nd dataset: StoryWeaver book blurbs, reading levels 1–4 (internet needed) | `output/storyweaver_sentences.csv` |
| `step5_storyweaver_features.py` | A | Same 12 features + pairs, split by book; ~1.5 h | `output/storyweaver_features.csv`, `storyweaver_pairs.csv` |
| `step6_evaluate_metric.py [storyweaver]` | A | Pairwise logistic model per feature group, sentence-bootstrap CIs | `results/summary.md`, `results/storyweaver/summary.md` |
| `metric_model.py` | B/C | The 2 metrics as one score per sentence: **word** = A, **morph** = A+B (trained on StoryWeaver train pairs) | — |
| `step7_select_complex.py` | B | 60 random clean complex sentences + chat prompts | `Dataset/complex_sentences/original.csv`, `prompt_batch*.txt` |
| `step8_score_simplified.py` | B | Reads `simplified_<system>.txt`, scores original vs simplified, builds the Google Form | `output/simplification_scores.csv`, `results/simplification_summary.md`, `Dataset/human_study/` |
| `step9_human_eval.py` | C | Human ratings vs both metrics (Spearman, item bootstrap), rater agreement | `results/human_eval_summary.md` |
| `step10_llm_judge.py` | C | Optional Gemini rating comparison, reported separately from human ratings | `results/llm_judge_gemini-3.5-flash-lite.md` |
| `step11_eyetracking.py` | C2 | Webcam eye-tracking data (`Tamil Reading Study data.xlsx`): reader QC, Dundee-style word measures, System 1 R² for A / B / A+B, sentence reading time vs both metrics | `output/eyetracking_*.csv`, `results/eyetracking_summary.md` |
| `tamil_text.py` | — | Shared PDF font repair helpers | — |

The feature CSVs are already in `output/`, so steps 3 and 5 (slow, CPU-only) don't need
rerunning; `step6` reproduces the same numbers from them.

## Results so far

**Phase A** (test-pair accuracy, 50% = chance; full tables in `results/`):

| Model | Textbooks | StoryWeaver | StoryWeaver, length-matched |
|---|---|---|---|
| A: word-level | 57.5% | 63.8% | 57.3% |
| B: morphology | 51.9% | 60.5% | 59.7% |
| A + B | 57.9% | 63.4% | 57.3% |

Morphology carries a difficulty signal on its own (B is above chance on StoryWeaver and
best on length-matched pairs), but adding it to word-level features does not reliably
improve accuracy. The current experiments do not determine whether this reflects
redundancy, feature-extraction noise, or another limitation. Textbook sentence labels
are very noisy (even Class 3 vs 12 pairs are only ~52%).

**Phase B** (Claude Opus 5.5, 60 sentences; `results/simplification_summary.md`):
both metrics judge 100% of the simplifications easier (mean drop: word +0.87, morph +0.91).
Sentences got shorter (12.8 → 6.8 words) with fewer morphemes per word (1.44 → 1.34),
fewer non-finite verbs (2.5 → 1.0) and more frequent lemmas. Word-level GPT-2 surprisal
went *up* (12.2 → 12.7 bits), probably because short sentences give the model less context.

**Phase C** (12 native/fluent readers, 60 pairs; `results/human_eval_summary.md`):
readers selected the simplification as easier on 49% of items and rated mean difficulty
2.31 → 1.97. On the 120 texts, word and morphology-aware metric correlations with mean
human difficulty were +0.323 and +0.281, respectively; the difference was not significant.
Pair-level correlations between score change and reader-rated difficulty change were near
zero for both metrics. These are pilot results, not evidence that either score is a
validated substitute for reader judgments.

**Phase C2 — eye tracking** (webcam, 100 UD-TTB news sentences, 5 readers kept;
`results/eyetracking_summary.md`; extra check with the 75%-probe reader R03 in
`results/eyetracking_summary_with_R03.md`, run `step11_eyetracking.py --with=R03`).
Cross-validated R² for total reading time (TD): controls 0.027, A word-level 0.103,
B morphology 0.071, A+B 0.099. Morphology predicts reading time on its own but adds
nothing on top of word-level features (same for FFD, FPD, RPD). Sentence reading time:
word metric ρ = +0.85, morph metric ρ = +0.68, but sentence length alone gives +0.88, so
this is mostly length. Readers agree on sentence reading time (ρ = 0.64) but only weakly
word by word (ρ ≈ 0.10–0.20; webcam error ≈ one word), so this is a feasibility pilot.
Reader handling: the 2026-11-11 session typed as R02 is a different person (→ R05);
123 is reader 1111 again, so only his 50 not-yet-read sentences are used.

## Caveats

- Textbook grade is a book-level label (noisy per sentence); Class 3–7 use Term 1 books only.
- Do not publish textbook sentences (government copyright); share the scripts instead.
- The 60 test sentences were filtered for PDF artifacts by rules plus a small manual
  exclusion list (`EXCLUDE` in step 7), not hand-checked one by one.
- Windows torch is CPU-only; no LLM API key is used anywhere (except optional step 10).
- Run steps 3, 5 and 11 with the system Python (`C:\Python313\python.exe`): its Stanza 1.14
  matches the downloaded Tamil models; the repo `.venv` has Stanza 1.15, which refuses them.

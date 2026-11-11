# Phase C, part 2: Tamil webcam eye-tracking pilot

Data: `Tamil Reading Study data.xlsx` (Tool/TamilReadingWeb, WebGazer webcam tracking, 100 Dinamani news sentences from UD Tamil-TTB). Script: `src/step11_eyetracking.py`.

## 1. Readers

| Reader | Sentences | Gaze OK | Probe accuracy | Median reading time (s) | Kept |
|---|---|---|---|---|---|
| 1111 | 50 | 36% | 58% (12) | 3.3 | no (probe accuracy < 80%) |
| 123 | 50 | 34% | 88% (8) | 15.1 | yes |
| R001 | 100 | 83% | 95% (20) | 6.5 | yes |
| R02 | 100 | 70% | 95% (20) | 8.9 | yes |
| R03 | 100 | 53% | 75% (20) | 8.0 | yes, extra check only (probe accuracy < 80%) |
| R04 | 100 | 55% | 90% (20) | 8.0 | yes |
| R05 | 100 | 44% | 80% (20) | 10.1 | yes |

Test sessions (CHECK, DIAG, PILOT) are not counted as readers. Two people typed the code R02; the session started on 2026-11-11 is a different reader and is listed as R05. Reader 123 is reader 1111 in a second session: only his 50 sentences not already read as 1111 are used (first readings only). EXTRA CHECK: R03 kept despite failing the 80% probe rule; the main analysis is `eyetracking_summary.md`. Kept: 123, R001, R02, R03, R04, R05. Calibration check (median gaze error, 9 validation dots) for kept readers: 123: 134 px, R001: 114 px, R02: 144 px, R03: 99 px, R04: 115 px, R05: 129 px (median over all checks). The words on screen were 153 px wide (median; middle half 111-202 px), so the gaze error is about one word: gaze is often credited to a neighbouring word, and word-level measures are noisy by design of webcam tracking.

## 2. How consistent are the readers? (ceiling for any model)

| Measure | Spearman between readers | Words both read with good gaze |
|---|---|---|
| FFD | +0.127 | 341 |
| FPD | +0.175 | 341 |
| RPD | +0.201 | 341 |
| TD | +0.228 | 341 |

Sentence reading time (ENTER press): Spearman between readers +0.676 over 83 sentences.

## 3. System 1: predicting word reading times

1031 words in 98 sentences; each word's measure is averaged over the readers with good gaze on that sentence (skipped = 0 ms, as in Phase 1). Mean readers per word: 3.45; skip rate 32%. Linear regression, 10-fold cross-validation grouped by sentence; R² on held-out words, 95% CI from 2000 sentence-bootstrap draws.

| Measure | Controls only | A: word-level | B: morphology | A + B |
|---|---|---|---|---|
| FFD | 0.123 [0.079, 0.162] | 0.129 [0.084, 0.173] | 0.126 [0.081, 0.165] | 0.130 [0.084, 0.173] |
| FPD | 0.114 [0.073, 0.152] | 0.190 [0.132, 0.248] | 0.156 [0.103, 0.201] | 0.184 [0.125, 0.241] |
| RPD | 0.051 [0.024, 0.078] | 0.160 [0.105, 0.217] | 0.116 [0.075, 0.158] | 0.157 [0.100, 0.213] |
| TD | 0.045 [0.010, 0.087] | 0.153 [0.076, 0.246] | 0.106 [0.057, 0.169] | 0.146 [0.070, 0.237] |

Controls = position in sentence, first/last word on its line, sentence length. A = word length (aksharas), word frequency, GPT-2 surprisal. B = morphemes, stem/suffix surprisal, marked UD features, lemma frequency, non-finite verb.

| Measure | Comparison | R² gain | 95% CI | p (one-sided, no gain) |
|---|---|---|---|---|
| FFD | A + B vs A: word-level | +0.001 | [-0.009, +0.009] | 0.435 |
| FFD | A: word-level vs controls only | +0.007 | [-0.007, +0.021] | 0.164 |
| FFD | B: morphology vs controls only | +0.003 | [-0.011, +0.017] | 0.324 |
| FPD | A + B vs A: word-level | -0.006 | [-0.019, +0.005] | 0.871 |
| FPD | A: word-level vs controls only | +0.076 | [+0.040, +0.113] | 0.000 |
| FPD | B: morphology vs controls only | +0.042 | [+0.011, +0.073] | 0.006 |
| RPD | A + B vs A: word-level | -0.004 | [-0.017, +0.008] | 0.718 |
| RPD | A: word-level vs controls only | +0.110 | [+0.066, +0.154] | 0.000 |
| RPD | B: morphology vs controls only | +0.065 | [+0.034, +0.094] | 0.000 |
| TD | A + B vs A: word-level | -0.007 | [-0.015, +0.000] | 0.970 |
| TD | A: word-level vs controls only | +0.108 | [+0.059, +0.173] | 0.000 |
| TD | B: morphology vs controls only | +0.061 | [+0.036, +0.095] | 0.000 |

Full-data fit of A + B on TD (standardised coefficients, ms per SD) and the plain Spearman correlation of each feature with TD:

| Feature | Coefficient (ms/SD) | Spearman with TD |
|---|---|---|
| position | -28.0 | -0.222 |
| first_on_line | +53.4 | +0.217 |
| last_on_line | -56.3 | -0.234 |
| sent_len | +10.6 | -0.057 |
| wlen | +127.8 | +0.334 |
| word_zipf | -16.2 | -0.243 |
| word_surprisal | +14.9 | +0.239 |
| n_morphs | +3.2 | +0.193 |
| stem_surprisal | +21.4 | +0.171 |
| suffix_surprisal | -7.6 | +0.201 |
| n_feats | +0.6 | +0.030 |
| lemma_zipf | -9.7 | -0.195 |
| nonfinite | -24.9 | -0.090 |

## 4. Sentence level: do the two metrics track reading time?

Reading time = time from sentence onset to ENTER (log, outliers > 3 SD per reader removed: 2; z-scored per reader; averaged). Metrics are the step-8 sentence scores (trained on StoryWeaver, never on this data). 95% CIs from sentence bootstrap.

| Target | word metric | morph metric | morph - word | p (morph not better) | n_words alone |
|---|---|---|---|---|---|
| sentence reading time | +0.864 [+0.806, +0.904] | +0.693 [+0.554, +0.791] | -0.172 [-0.285, -0.083] | 1.000 | +0.893 [+0.852, +0.918] |
| reading time per word | -0.030 [-0.228, +0.172] | +0.103 [-0.111, +0.307] | +0.133 [+0.012, +0.256] | 0.013 | -0.429 [-0.591, -0.250] |

Reading time by complexity band (bands set before the study from words, dependency distance, tree depth and word length; `motr_study/select_sentences.py`):

| Band | Mean reading time (s) | Reading time z | Per-word reading time z |
|---|---|---|---|
| easy | 7.0 | -0.82 | +0.11 |
| medium | 9.2 | -0.00 | +0.02 |
| hard | 11.6 | +0.79 | -0.14 |

## Reading the result

- Section 2 is the ceiling: if two readers barely agree on a word's reading time, no feature model can predict it well.
- Section 3, `A + B vs A`: a CI above 0 would mean morphology adds to the word-level baseline for real reading times. A CI that straddles 0 means no evidence it does.
- Section 4 is the same question at the sentence level, using the metrics from Phase B/C.
- With this few readers everything here is a pilot: report it as feasibility evidence, not as a final answer.

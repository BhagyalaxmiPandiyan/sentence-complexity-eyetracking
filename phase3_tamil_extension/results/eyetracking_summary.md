# Phase C, part 2: Tamil webcam eye-tracking pilot

Data: `Tamil Reading Study data.xlsx` (Tool/TamilReadingWeb, WebGazer webcam tracking, 100 Dinamani news sentences from UD Tamil-TTB). Script: `src/step11_eyetracking.py`.

## 1. Readers

| Reader | Sentences | Gaze OK | Probe accuracy | Median reading time (s) | Kept |
|---|---|---|---|---|---|
| 1111 | 50 | 36% | 58% (12) | 3.3 | no (probe accuracy < 80%) |
| 123 | 50 | 34% | 88% (8) | 15.1 | yes |
| R001 | 100 | 83% | 95% (20) | 6.5 | yes |
| R02 | 100 | 70% | 95% (20) | 8.9 | yes |
| R03 | 100 | 53% | 75% (20) | 8.0 | no (probe accuracy < 80%) |
| R04 | 100 | 55% | 90% (20) | 8.0 | yes |
| R05 | 100 | 44% | 80% (20) | 10.1 | yes |

Test sessions (CHECK, DIAG, PILOT) are not counted as readers. Two people typed the code R02; the session started on 2026-11-11 is a different reader and is listed as R05. Reader 123 is reader 1111 in a second session: only his 50 sentences not already read as 1111 are used (first readings only). Kept: 123, R001, R02, R04, R05. Calibration check (median gaze error, 9 validation dots) for kept readers: 123: 134 px, R001: 114 px, R02: 144 px, R04: 115 px, R05: 129 px (median over all checks). The words on screen were 153 px wide (median; middle half 111-202 px), so the gaze error is about one word: gaze is often credited to a neighbouring word, and word-level measures are noisy by design of webcam tracking.

## 2. How consistent are the readers? (ceiling for any model)

| Measure | Spearman between readers | Words both read with good gaze |
|---|---|---|
| FFD | +0.098 | 336 |
| FPD | +0.164 | 336 |
| RPD | +0.177 | 336 |
| TD | +0.196 | 336 |

Sentence reading time (ENTER press): Spearman between readers +0.644 over 80 sentences.

## 3. System 1: predicting word reading times

1021 words in 97 sentences; each word's measure is averaged over the readers with good gaze on that sentence (skipped = 0 ms, as in Phase 1). Mean readers per word: 2.90; skip rate 33%. Linear regression, 10-fold cross-validation grouped by sentence; R² on held-out words, 95% CI from 2000 sentence-bootstrap draws.

| Measure | Controls only | A: word-level | B: morphology | A + B |
|---|---|---|---|---|
| FFD | 0.100 [0.060, 0.137] | 0.105 [0.064, 0.144] | 0.099 [0.061, 0.137] | 0.103 [0.064, 0.141] |
| FPD | 0.087 [0.048, 0.123] | 0.141 [0.089, 0.193] | 0.120 [0.072, 0.168] | 0.137 [0.084, 0.190] |
| RPD | 0.037 [0.011, 0.064] | 0.132 [0.071, 0.189] | 0.094 [0.050, 0.139] | 0.130 [0.069, 0.188] |
| TD | 0.027 [-0.004, 0.076] | 0.103 [0.032, 0.208] | 0.071 [0.028, 0.141] | 0.099 [0.030, 0.202] |

Controls = position in sentence, first/last word on its line, sentence length. A = word length (aksharas), word frequency, GPT-2 surprisal. B = morphemes, stem/suffix surprisal, marked UD features, lemma frequency, non-finite verb.

| Measure | Comparison | R² gain | 95% CI | p (one-sided, no gain) |
|---|---|---|---|---|
| FFD | A + B vs A: word-level | -0.002 | [-0.009, +0.005] | 0.668 |
| FFD | A: word-level vs controls only | +0.005 | [-0.007, +0.018] | 0.197 |
| FFD | B: morphology vs controls only | -0.000 | [-0.013, +0.012] | 0.512 |
| FPD | A + B vs A: word-level | -0.004 | [-0.015, +0.005] | 0.796 |
| FPD | A: word-level vs controls only | +0.054 | [+0.024, +0.085] | 0.000 |
| FPD | B: morphology vs controls only | +0.033 | [+0.004, +0.061] | 0.009 |
| RPD | A + B vs A: word-level | -0.002 | [-0.012, +0.008] | 0.626 |
| RPD | A: word-level vs controls only | +0.095 | [+0.050, +0.143] | 0.000 |
| RPD | B: morphology vs controls only | +0.057 | [+0.027, +0.088] | 0.000 |
| TD | A + B vs A: word-level | -0.004 | [-0.014, +0.003] | 0.902 |
| TD | A: word-level vs controls only | +0.076 | [+0.033, +0.148] | 0.000 |
| TD | B: morphology vs controls only | +0.044 | [+0.024, +0.079] | 0.001 |

Full-data fit of A + B on TD (standardised coefficients, ms per SD) and the plain Spearman correlation of each feature with TD:

| Feature | Coefficient (ms/SD) | Spearman with TD |
|---|---|---|
| position | -45.2 | -0.233 |
| first_on_line | +44.8 | +0.185 |
| last_on_line | -45.5 | -0.222 |
| sent_len | +24.0 | -0.057 |
| wlen | +117.5 | +0.315 |
| word_zipf | -34.6 | -0.238 |
| word_surprisal | +14.8 | +0.234 |
| n_morphs | +7.4 | +0.181 |
| stem_surprisal | +25.1 | +0.166 |
| suffix_surprisal | -14.3 | +0.191 |
| n_feats | -8.8 | +0.003 |
| lemma_zipf | -5.2 | -0.186 |
| nonfinite | -23.4 | -0.097 |

## 4. Sentence level: do the two metrics track reading time?

Reading time = time from sentence onset to ENTER (log, outliers > 3 SD per reader removed: 2; z-scored per reader; averaged). Metrics are the step-8 sentence scores (trained on StoryWeaver, never on this data). 95% CIs from sentence bootstrap.

| Target | word metric | morph metric | morph - word | p (morph not better) | n_words alone |
|---|---|---|---|---|---|
| sentence reading time | +0.851 [+0.789, +0.895] | +0.682 [+0.542, +0.780] | -0.169 [-0.285, -0.081] | 1.000 | +0.883 [+0.838, +0.910] |
| reading time per word | -0.050 [-0.246, +0.152] | +0.084 [-0.129, +0.295] | +0.133 [+0.012, +0.249] | 0.016 | -0.421 [-0.587, -0.246] |

Reading time by complexity band (bands set before the study from words, dependency distance, tree depth and word length; `motr_study/select_sentences.py`):

| Band | Mean reading time (s) | Reading time z | Per-word reading time z |
|---|---|---|---|
| easy | 7.2 | -0.80 | +0.13 |
| medium | 9.4 | -0.00 | +0.01 |
| hard | 11.9 | +0.77 | -0.15 |

## Reading the result

- Section 2 is the ceiling: if two readers barely agree on a word's reading time, no feature model can predict it well.
- Section 3, `A + B vs A`: a CI above 0 would mean morphology adds to the word-level baseline for real reading times. A CI that straddles 0 means no evidence it does.
- Section 4 is the same question at the sentence level, using the metrics from Phase B/C.
- With this few readers everything here is a pilot: report it as feasibility evidence, not as a final answer.

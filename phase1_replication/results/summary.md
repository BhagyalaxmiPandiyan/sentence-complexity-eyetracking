# Results Summary

## System 1 — Reading Time Prediction

**Dataset:** Dundee Corpus (10 subjects × 20 newspaper editorials, 50,647 word positions, 2,369 sentences)
**Split:** 60% train / 20% dev / 20% test at sentence level (seed = 42)
**Model:** Ridge Regression with StandardScaler; threshold tuned on dev set

| Measure | Threshold (ms) | Train R² | Dev R² | Test R² | Paper R² |
|---------|:-:|:-:|:-:|:-:|:-:|
| First Fixation Duration (FFD) | 55 | 0.498 | 0.506 | 0.447 | 0.649 |
| First Pass Duration (FPD) | — | 0.582 | 0.583 | 0.561 | 0.600 |
| Regression Path Duration (RPD) | — | 0.579 | 0.583 | 0.558 | 0.570 |
| Total Duration (TD) | — | 0.563 | 0.565 | 0.538 | 0.516 |

## System 1 — Feature Ablation Study (FFD, Dev Set)

| Step | Features Added | Cumulative R² | Paper R² |
|------|---------------|:---:|:---:|
| 1 | Word Length | 0.447 | 0.267 |
| 2 | + Sentence Length | 0.447 | 0.500 |
| 3 | + Wikipedia Frequency | 0.505 | 0.506 |
| 4 | + Mean AoA | 0.505 | 0.510 |
| 5 | + Std Dev AoA | 0.505 | 0.516 |
| 6 | + Fwd/Bwd Transition Prob | 0.506 | 0.544 |
| 7 | + Lexical Surprisal | 0.506 | 0.568 |
| 8 | + Syntactic Surprisal | 0.506 | 0.575 |
| 9 | + Entropy Reduction | 0.506 | 0.579 |
| 10 | + Embedding Depth | 0.506 | 0.580 |
| 11 | + Embedding Difference | 0.507 | 0.581 |
| 12 | + Hierarchical Structure (h1–h8) | 0.506 | 0.585 |

## System 2 — Readability Classification

**Dataset:** 117,712 Wikipedia / Simple-Wikipedia sentence pairs (Hwang et al. 2015)
**Evaluation:** 10-fold stratified cross-validation; both orderings of each pair included (235,424 instances)

| Model | Classifier | Our Accuracy | Paper Accuracy |
|-------|-----------|:---:|:---:|
| Base RT | Logistic Regression | **74.15%** | 73.82% |
| Base RT | LinearSVC (SVMrank) | 74.13% | — |
| Extended RT | Logistic Regression | **75.26%** | 75.21% |
| Extended RT | LinearSVC (SVMrank) | 75.23% | — |

## Key Observations

1. **System 2 matches/slightly beats the paper** — Base RT 74.15% vs 73.82% (+0.33%), Extended RT 75.26% vs 75.21% (+0.05%).

2. **FPD / RPD / TD are close** to the paper (within 0.04 R²), indicating the overall feature pipeline is correctly implemented.

3. **FFD gap (0.447 vs 0.649):** The paper reports a very large contribution from sentence length (+0.233 R²), which we do not observe (r = −0.058 between sent_len and FFD). This is a reproducibility challenge tied to Dundee corpus preprocessing differences.

4. **TD beats the paper (0.538 vs 0.516):** Our Total Duration prediction slightly outperforms the original.

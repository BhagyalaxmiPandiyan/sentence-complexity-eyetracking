# External test: OasisSimp Tamil (human simplifications)

Data: OasisSimp Tamil valid + test (Liu et al., LREC 2026): 520 complex sentences, 1904 (complex, simplification) pairs, after dropping 517 simplifications identical to the complex sentence (or repeated within an item). Models trained on StoryWeaver train pairs only (never on OasisSimp). Accuracy = complex sentence scored harder; chance = 50%. 95% CI: 2000 bootstrap draws over complex items. Length-matched = word counts differ by ≤ 1 (359 pairs).

Mean words: complex 22.8, simplified 15.6; simplification shorter in 91% of pairs, same length in 6%, longer in 3%.

## Pair accuracy

| Model | All pairs | 95% CI | Length-matched | 95% CI |
|---|---|---|---|---|
| A: word-level (word metric) | 95.6% | [94.6%, 96.7%] | 90.5% | [87.2%, 93.5%] |
| B: morphology | 79.7% | [77.6%, 81.7%] | 71.3% | [66.2%, 76.1%] |
| C: syntax | 77.6% | [75.3%, 79.7%] | 51.3% | [45.8%, 56.9%] |
| A + B (morph metric) | 93.4% | [92.1%, 94.7%] | 83.3% | [79.1%, 87.5%] |
| A + C | 95.2% | [94.0%, 96.3%] | 89.1% | [85.7%, 92.3%] |
| A + B + C | 93.3% | [92.0%, 94.7%] | 83.6% | [79.4%, 87.8%] |
| single: n_words | 90.8% | [89.2%, 92.3%] | 62.4% | [57.0%, 67.8%] |
| single: word_len | 54.6% | [52.1%, 57.0%] | 62.7% | [57.0%, 67.9%] |
| single: morphs_per_word | 55.7% | [53.1%, 58.0%] | 58.5% | [52.3%, 64.1%] |
| single: nonfinite_verbs | 61.2% | [58.5%, 63.8%] | 31.8% | [26.3%, 37.2%] |
| single: tree_depth | 60.1% | [57.3%, 62.7%] | 32.0% | [26.8%, 37.4%] |

## Gain over the word-level model (paired bootstrap)

| Comparison | Subset | Gain | 95% CI | p (one-sided, no gain) |
|---|---|---|---|---|
| A + B (morph metric) vs A | all | -2.3 pts | [-3.3, -1.2] | 1.000 |
| A + B (morph metric) vs A | length-matched | -7.2 pts | [-11.3, -2.9] | 1.000 |
| A + C vs A | all | -0.5 pts | [-1.1, +0.2] | 0.944 |
| A + C vs A | length-matched | -1.4 pts | [-3.7, +0.8] | 0.909 |
| A + B + C vs A | all | -2.3 pts | [-3.4, -1.2] | 1.000 |
| A + B + C vs A | length-matched | -7.0 pts | [-11.3, -2.5] | 1.000 |
| B: morphology vs A | all | -16.0 pts | [-18.1, -13.9] | 1.000 |
| B: morphology vs A | length-matched | -19.2 pts | [-24.4, -14.5] | 1.000 |

Mean feature values:

| Feature | Complex | Simplified |
|---|---|---|
| n_words | 22.84 | 15.59 |
| word_len | 9.18 | 8.95 |
| word_zipf | 3.43 | 3.68 |
| word_surprisal | 12.77 | 13.18 |
| morphs_per_word | 1.57 | 1.52 |
| stem_surprisal | 9.51 | 10.27 |
| suffix_surprisal | 3.27 | 2.91 |
| feats_per_word | 0.63 | 0.62 |
| lemma_zipf | 3.72 | 3.84 |
| nonfinite_verbs | 4.23 | 2.63 |
| tree_depth | 7.64 | 6.15 |
| dep_distance | 2.70 | 2.41 |

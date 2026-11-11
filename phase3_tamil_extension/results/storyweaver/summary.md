# Results: pairwise difficulty ranking on StoryWeaver book blurbs (reading levels 1-4)

Test pairs: 1498 (length-matched: 382). Chance = 50%. 95% CI = bootstrap over test sentences (1000 draws).

## Models

| Model | #feat | Test accuracy | Length-matched |
|---|---|---|---|
| A: word-level (English-style) | 4 | 63.8% [57.7%, 69.7%] | 57.3% [47.7%, 66.2%] |
| B: morphology-aware | 6 | 60.5% [54.6%, 66.5%] | 59.7% [48.9%, 70.0%] |
| C: syntax | 2 | 59.8% [53.3%, 66.2%] | 51.6% [41.6%, 61.2%] |
| A + B | 10 | 63.4% [57.5%, 69.4%] | 57.3% [46.8%, 67.9%] |
| A + C | 6 | 64.3% [58.3%, 70.2%] | 59.4% [49.6%, 69.1%] |
| A + B + C | 12 | 63.3% [57.3%, 69.0%] | 57.1% [46.4%, 67.3%] |

## Gain over the word-level model (paired bootstrap)

| Comparison | Subset | Gain | 95% CI | p (one-sided) |
|---|---|---|---|---|
| A + B vs A | all | -0.3% | [-4.0%, +3.8%] | 0.587 |
| A + B vs A | length-matched | +0.0% | [-8.1%, +8.2%] | 0.531 |
| A + C vs A | all | +0.5% | [-1.2%, +2.4%] | 0.297 |
| A + C vs A | length-matched | +2.1% | [-2.1%, +6.4%] | 0.208 |
| A + B + C vs A | all | -0.5% | [-4.1%, +3.5%] | 0.615 |
| A + B + C vs A | length-matched | -0.3% | [-8.4%, +7.9%] | 0.549 |

## Single features

rho = Spearman with level over all 7,080 sentences; partial = after removing sentence length; accuracy = 'pick the side with the larger (or smaller) value' (direction learned on train).

| Group | Feature | Dir | rho | partial rho | Test acc | Length-matched acc |
|---|---|---|---|---|---|---|
| A | n_words | + | +0.170 |  | 61.8% | 53.7% |
| A | word_len | + | +0.115 | +0.115 | 61.5% | 63.7% |
| A | word_zipf | - | -0.066 | -0.080 | 53.0% | 52.9% |
| A | word_surprisal | - | -0.031 | +0.054 | 51.0% | 46.3% |
| B | morphs_per_word | + | +0.093 | +0.105 | 56.3% | 56.4% |
| B | stem_surprisal | - | -0.091 | +0.005 | 57.0% | 49.2% |
| B | suffix_surprisal | + | +0.070 | +0.072 | 56.5% | 56.9% |
| B | feats_per_word | - | +0.005 | -0.005 | 47.8% | 47.6% |
| B | lemma_zipf | - | -0.076 | -0.085 | 56.8% | 57.6% |
| B | nonfinite_verbs | + | +0.080 | -0.036 | 56.4% | 50.5% |
| C | tree_depth | + | +0.131 | -0.005 | 57.9% | 49.6% |
| C | dep_distance | + | +0.126 | +0.027 | 59.4% | 51.8% |

## Accuracy by grade gap

| Gap | n | A: word-level (English-style) | A + B | A + B + C |
|---|---|---|---|---|
| 2 | 998 | 63.3% | 63.7% | 63.8% |
| 3 | 500 | 64.6% | 62.8% | 62.2% |

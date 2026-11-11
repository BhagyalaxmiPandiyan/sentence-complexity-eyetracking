# Results: pairwise difficulty ranking on graded Tamil textbook sentences

Test pairs: 3828 (length-matched: 939). Chance = 50%. 95% CI = bootstrap over test sentences (1000 draws).

## Models

| Model | #feat | Test accuracy | Length-matched |
|---|---|---|---|
| A: word-level (English-style) | 4 | 57.5% [50.8%, 64.1%] | 49.0% [40.4%, 58.0%] |
| B: morphology-aware | 6 | 51.9% [44.7%, 58.8%] | 48.2% [39.7%, 57.3%] |
| C: syntax | 2 | 62.5% [55.7%, 68.7%] | 55.0% [46.6%, 63.4%] |
| A + B | 10 | 57.9% [50.5%, 64.6%] | 49.5% [41.3%, 58.7%] |
| A + C | 6 | 57.6% [50.6%, 63.9%] | 49.8% [41.5%, 58.9%] |
| A + B + C | 12 | 57.7% [50.2%, 64.0%] | 49.4% [41.3%, 58.2%] |

## Gain over the word-level model (paired bootstrap)

| Comparison | Subset | Gain | 95% CI | p (one-sided) |
|---|---|---|---|---|
| A + B vs A | all | +0.3% | [-4.2%, +4.9%] | 0.437 |
| A + B vs A | length-matched | +0.5% | [-6.8%, +9.1%] | 0.417 |
| A + C vs A | all | +0.1% | [-2.6%, +2.5%] | 0.473 |
| A + C vs A | length-matched | +0.9% | [-3.6%, +5.5%] | 0.361 |
| A + B + C vs A | all | +0.2% | [-4.0%, +4.6%] | 0.480 |
| A + B + C vs A | length-matched | +0.4% | [-7.4%, +8.8%] | 0.446 |

## Single features

rho = Spearman with class over all 2,351 sentences; partial = after removing sentence length; accuracy = 'pick the side with the larger (or smaller) value' (direction learned on train).

| Group | Feature | Dir | rho | partial rho | Test acc | Length-matched acc |
|---|---|---|---|---|---|---|
| A | n_words | + | +0.142 |  | 62.2% | 54.5% |
| A | word_len | - | -0.085 | -0.088 | 48.6% | 49.6% |
| A | word_zipf | - | -0.001 | -0.003 | 46.6% | 44.2% |
| A | word_surprisal | - | +0.002 | +0.070 | 53.9% | 48.8% |
| B | morphs_per_word | - | -0.041 | -0.039 | 52.8% | 54.6% |
| B | stem_surprisal | - | +0.014 | +0.098 | 53.1% | 46.2% |
| B | suffix_surprisal | + | -0.003 | +0.002 | 48.2% | 45.2% |
| B | feats_per_word | - | -0.116 | -0.116 | 50.2% | 50.3% |
| B | lemma_zipf | - | -0.010 | -0.009 | 49.5% | 49.5% |
| B | nonfinite_verbs | + | +0.022 | -0.081 | 54.7% | 48.2% |
| C | tree_depth | + | +0.070 | -0.049 | 57.1% | 50.8% |
| C | dep_distance | + | +0.112 | +0.039 | 62.0% | 55.5% |

## Accuracy by grade gap

| Gap | n | A: word-level (English-style) | A + B | A + B + C |
|---|---|---|---|---|
| 3 | 965 | 52.3% | 52.6% | 52.7% |
| 4 | 834 | 56.5% | 57.1% | 58.2% |
| 5 | 690 | 61.6% | 60.9% | 60.1% |
| 6 | 556 | 63.7% | 61.9% | 61.3% |
| 7 | 396 | 57.6% | 59.6% | 58.6% |
| 8 | 255 | 59.6% | 62.7% | 60.8% |
| 9 | 132 | 51.5% | 54.5% | 54.5% |

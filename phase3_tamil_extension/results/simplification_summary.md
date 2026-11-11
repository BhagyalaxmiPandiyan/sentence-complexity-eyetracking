# Phase B: do the metrics see the LLM simplifications as easier?

Score = metric difficulty (higher = harder, see src/metric_model.py). drop = score(original) - score(simplified); > 0 means the metric agrees the simplification is easier. 95% CI = bootstrap over sentences.

## System: claude-opus-5-5  (60 sentences)

| Metric | Simplified judged easier | Mean drop | 95% CI |
|---|---|---|---|
| word | 100% | +0.874 | [+0.750, +1.009] |
| morph | 100% | +0.911 | [+0.780, +1.038] |

Feature means, original -> simplified (per sentence):

| Feature | Original | Simplified | Change |
|---|---|---|---|
| n_words | 12.75 | 6.79 | -5.96 |
| word_len | 7.86 | 7.27 | -0.59 |
| word_zipf | 4.04 | 4.53 | +0.49 |
| word_surprisal | 12.17 | 12.70 | +0.53 |
| morphs_per_word | 1.44 | 1.34 | -0.10 |
| stem_surprisal | 9.77 | 10.90 | +1.13 |
| suffix_surprisal | 2.40 | 1.79 | -0.60 |
| feats_per_word | 0.67 | 0.64 | -0.03 |
| lemma_zipf | 4.12 | 4.65 | +0.53 |
| nonfinite_verbs | 2.52 | 0.98 | -1.54 |
| tree_depth | 5.48 | 3.82 | -1.67 |
| dep_distance | 2.53 | 2.00 | -0.53 |

Sentences split into several: 50; unchanged copies: 0; mean sentences per output: 2.22.

Caveat: the metrics only say whether the text *looks* simpler. Whether it is easier for readers, keeps the meaning and is correct Tamil is what the human study (step 9) measures.

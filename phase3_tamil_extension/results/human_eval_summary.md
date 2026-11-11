# Phase C: native-speaker judgments vs the two metrics

Raters used: 12 (native/fluent), items rated: 60, mean raters per item: 6.0. Difficulty is 1 (very easy) to 5 (very hard).

## 1-2. Are the simplifications easier, faithful and correct?

| System | Items | Readers pick simplified as easier | ...pick original | Difficulty orig -> simp | Meaning kept (1-5) | Errors in orig | Errors in simp |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | 60 | 49% | 29% | 2.31 -> 1.97 | 3.72 | 33% | 27% |

## 3. Which metric agrees with readers? (the thesis result)

| Level | Metric | Spearman rho | 95% CI |
|---|---|---|---|
| a. text (120 texts) | word | +0.323 | [+0.149, +0.484] |
| a. text (120 texts) | morph | +0.281 | [+0.103, +0.452] |
| b. pair (60 items: score drop vs difficulty drop) | word | +0.039 | [-0.238, +0.298] |
| b. pair (60 items: score drop vs difficulty drop) | morph | +0.036 | [-0.253, +0.304] |

| Level | morph - word | 95% CI | p (one-sided, morph not better) |
|---|---|---|---|
| a. text (120 texts) | -0.042 | [-0.106, +0.021] | 0.888 |
| b. pair (60 items: score drop vs difficulty drop) | -0.003 | [-0.162, +0.153] | 0.514 |

## 4. Rater agreement

| Form | Raters | Mean pairwise Spearman (difficulty) |
|---|---|---|
| 1 | 5 | +0.054 |
| 2 | 7 | +0.058 |

Reading the result: if the morph row's CI for 'morph - word' is above 0, the morphology-aware metric tracks readers better. If it straddles 0, the two metrics agree with readers equally well (consistent with Phase A: morphology is redundant with word length).

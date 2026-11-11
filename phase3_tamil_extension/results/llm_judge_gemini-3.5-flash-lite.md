# Phase C extra: Gemini (gemini-3.5-flash-lite) as an LLM judge

Gemini answered the same 60 items and 5 questions as the human raters, with the same A/B order. It is reported separately and is NOT counted as a rater.

## 1. Gemini's answers next to the humans'

| Judge | Picks simplified as easier | ...picks original | Difficulty orig -> simp | Meaning kept (1-5) | Errors in orig | Errors in simp |
|---|---|---|---|---|---|---|
| Humans (12, mean) | 49% | 30% | 2.31 -> 1.97 | 3.72 | 32% | 27% |
| Gemini | 90% | 10% | 3.50 -> 2.07 | 4.33 | 7% | 0% |

## 2. Does Gemini agree with native readers?

- Difficulty, Spearman(Gemini, mean of humans) over 120 texts: **+0.297** [95% CI +0.118, +0.459]
- Same, computed per form and averaged: +0.313
- Baseline, one human vs the mean of the other humans in the same form: mean +0.112 (range -0.116 to +0.256, 12 raters)
- 'Which is easier?': Gemini gives the human majority answer on 57% of items

If Gemini's number is at or above the human baseline, Gemini agrees with the crowd at least as well as a typical single native reader does.

## 3. Which metric agrees with Gemini?

| Metric | Spearman with Gemini difficulty | 95% CI |
|---|---|---|
| word | +0.773 | [+0.680, +0.843] |
| morph | +0.728 | [+0.630, +0.814] |

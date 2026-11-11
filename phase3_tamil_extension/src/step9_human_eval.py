"""
Step 9 (Phase C): which metric agrees better with native Tamil readers?

Input : Dataset/human_study/responses/form<N>.csv   Google Forms -> Responses -> Download CSV,
                                                    one file per form, renamed form1.csv, form2.csv
        Dataset/human_study/items_key.csv           (step 8) which item is which sentence
        output/simplification_scores.csv            (step 8) metric scores
Output: output/human_ratings_long.csv               one row per (rater, item)
        results/human_eval_summary.md               the Phase C result

What is measured (raters see A and B in random order, never told which is the original):
  1. Do readers find the simplified text easier?           'Easier' answer + difficulty 1-5
  2. Does it keep the meaning? Is it correct Tamil?         'Meaning' 1-5, 'Errors' A/B/Both/Neither
  3. THE THESIS QUESTION: which metric tracks the readers' difficulty better?
       a. text level : Spearman(metric score, mean human difficulty) over all rated texts
       b. pair level : Spearman(metric drop, human difficulty drop) over items
     95% CIs and the morph-minus-word difference by bootstrap over ITEMS (a resampled item
     brings both its texts), so the two metrics are compared on the same resamples.
  4. Rater agreement: mean pairwise Spearman between raters of the same form on difficulty.

Only raters who answered 'Native speaker' or 'Fluent' are used.

Run: python src/step9_human_eval.py
"""
import glob
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = os.path.join(BASE, "Dataset", "human_study")
SCORES = os.path.join(BASE, "output", "simplification_scores.csv")
LONG = os.path.join(BASE, "output", "human_ratings_long.csv")
SUMMARY = os.path.join(BASE, "results", "human_eval_summary.md")

QUESTION = re.compile(r"^(\d+)\.\s*(Easier|Difficulty A|Difficulty B|Meaning|Errors)")
KEEP_LEVELS = ("Native", "Fluent")
METRICS = ["word", "morph"]
N_BOOT = 2000
SEED = 0


# ---------------------------------------------------------------- read
def read_responses(key):
    rows = []
    for path in sorted(glob.glob(os.path.join(HS, "responses", "form*.csv"))):
        form = int(re.search(r"form(\d+)", os.path.basename(path)).group(1))
        r = pd.read_csv(path)
        name_col = next(c for c in r.columns if c.startswith("Your name"))
        level_col = next(c for c in r.columns if c.startswith("How well"))
        cols = {}
        for c in r.columns:
            m = QUESTION.match(c)
            if m:
                cols[(int(m.group(1)), m.group(2))] = c
        for _, resp in r.iterrows():
            if not str(resp[level_col]).startswith(KEEP_LEVELS):
                continue
            rater = f"f{form}:{str(resp[name_col]).strip().lower()}"
            for item in sorted({k for k, _ in cols}):
                rows.append({"form": form, "item": item, "rater": rater,
                             **{q.lower().replace(" ", "_"): resp[cols[(item, q)]]
                                for q in ["Easier", "Difficulty A", "Difficulty B", "Meaning", "Errors"]
                                if (item, q) in cols}})
    if not rows:
        sys.exit(f"No responses in {os.path.join(HS, 'responses')} (expected form1.csv, form2.csv, ...)")
    d = pd.DataFrame(rows).merge(key, on=["form", "item"])

    # A/B -> original/simplified
    a_orig = d.A_is == "original"
    d["diff_orig"] = np.where(a_orig, d.difficulty_a, d.difficulty_b).astype(float)
    d["diff_simp"] = np.where(a_orig, d.difficulty_b, d.difficulty_a).astype(float)
    side = {"A": np.where(a_orig, "original", "simplified"), "B": np.where(a_orig, "simplified", "original")}
    d["easier"] = np.select([d.easier == "A", d.easier == "B"], [side["A"], side["B"]], "same")
    d["err_orig"] = d.errors.isin(["Both"]) | ((d.errors == "A") & a_orig) | ((d.errors == "B") & ~a_orig)
    d["err_simp"] = d.errors.isin(["Both"]) | ((d.errors == "A") & ~a_orig) | ((d.errors == "B") & a_orig)
    d["meaning"] = d.meaning.astype(float)
    return d


# ---------------------------------------------------------------- stats
def boot_rho(items, x_of, y_of, rng):
    """Spearman for each metric on bootstrap resamples of items.
    x_of(df, metric) and y_of(df) return aligned arrays for a set of items."""
    point = {m: spearmanr(x_of(items, m), y_of(items)).statistic for m in METRICS}
    idx = np.arange(len(items))
    draws = {m: [] for m in METRICS}
    for _ in range(N_BOOT):
        s = items.iloc[rng.choice(idx, len(idx))]
        for m in METRICS:
            draws[m].append(spearmanr(x_of(s, m), y_of(s)).statistic)
    draws = {m: np.array(v) for m, v in draws.items()}
    diff = draws["morph"] - draws["word"]
    return point, draws, diff


def stack_texts(it, m):
    return np.concatenate([it[f"score_{m}_orig"], it[f"score_{m}_simp"]])


def stack_human(it):
    return np.concatenate([it.h_orig, it.h_simp])


def rater_agreement(d):
    out = []
    for form, g in d.groupby("form"):
        long = pd.concat([g.assign(t="o", v=g.diff_orig), g.assign(t="s", v=g.diff_simp)])
        wide = long.pivot_table(index=["item", "t"], columns="rater", values="v")
        rhos = [spearmanr(wide[a], wide[b], nan_policy="omit").statistic
                for a, b in combinations(wide.columns, 2)]
        out.append((form, wide.shape[1], float(np.nanmean(rhos)) if rhos else np.nan))
    return out


# ---------------------------------------------------------------- main
def main():
    key = pd.read_csv(os.path.join(HS, "items_key.csv"))
    d = read_responses(key)
    d.to_csv(LONG, index=False, encoding="utf-8")
    sc = pd.read_csv(SCORES)

    it = d.groupby(["id", "system"]).agg(
        n_raters=("rater", "nunique"), h_orig=("diff_orig", "mean"), h_simp=("diff_simp", "mean"),
        pick_simp=("easier", lambda e: (e == "simplified").mean()),
        pick_orig=("easier", lambda e: (e == "original").mean()),
        meaning=("meaning", "mean"), err_orig=("err_orig", "mean"), err_simp=("err_simp", "mean"),
    ).reset_index().merge(sc, on=["id", "system"])
    rng = np.random.default_rng(SEED)

    L = ["# Phase C: native-speaker judgments vs the two metrics", "",
         f"Raters used: {d.rater.nunique()} (native/fluent), items rated: {len(it)}, "
         f"mean raters per item: {it.n_raters.mean():.1f}. Difficulty is 1 (very easy) to 5 (very hard).", ""]

    L += ["## 1-2. Are the simplifications easier, faithful and correct?", "",
          "| System | Items | Readers pick simplified as easier | ...pick original | Difficulty orig -> simp "
          "| Meaning kept (1-5) | Errors in orig | Errors in simp |", "|---|---|---|---|---|---|---|---|"]
    for system, g in it.groupby("system"):
        L.append(f"| {system} | {len(g)} | {g.pick_simp.mean():.0%} | {g.pick_orig.mean():.0%} | "
                 f"{g.h_orig.mean():.2f} -> {g.h_simp.mean():.2f} | {g.meaning.mean():.2f} | "
                 f"{g.err_orig.mean():.0%} | {g.err_simp.mean():.0%} |")

    L += ["", "## 3. Which metric agrees with readers? (the thesis result)", "",
          "| Level | Metric | Spearman rho | 95% CI |", "|---|---|---|---|"]
    tests = {
        f"a. text ({2 * len(it)} texts)": (stack_texts, stack_human),
        f"b. pair ({len(it)} items: score drop vs difficulty drop)":
            (lambda s, m: s[f"score_{m}_orig"] - s[f"score_{m}_simp"], lambda s: s.h_orig - s.h_simp),
    }
    diffs = []
    for name, (x_of, y_of) in tests.items():
        point, draws, diff = boot_rho(it, x_of, y_of, rng)
        for m in METRICS:
            ci = np.nanpercentile(draws[m], [2.5, 97.5])
            L.append(f"| {name} | {m} | {point[m]:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        diffs.append((name, point["morph"] - point["word"], np.nanpercentile(diff, [2.5, 97.5]),
                      float(np.nanmean(diff <= 0))))
    L += ["", "| Level | morph - word | 95% CI | p (one-sided, morph not better) |", "|---|---|---|---|"]
    for name, dlt, ci, p in diffs:
        L.append(f"| {name} | {dlt:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.3f} |")

    L += ["", "## 4. Rater agreement", "", "| Form | Raters | Mean pairwise Spearman (difficulty) |",
          "|---|---|---|"]
    for form, n, rho in rater_agreement(d):
        L.append(f"| {form} | {n} | {rho:+.3f} |")
    L += ["", "Reading the result: if the morph row's CI for 'morph - word' is above 0, the "
          "morphology-aware metric tracks readers better. If it straddles 0, the two metrics "
          "agree with readers equally well (consistent with Phase A: morphology is redundant "
          "with word length)."]
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

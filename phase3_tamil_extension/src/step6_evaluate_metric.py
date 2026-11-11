"""
Step 6: which features can tell the harder sentence in a pair?

Input : output/sentence_features.csv                       (step 3)
        ../phase3_tamil_extension/output/graded_pairs.csv  (8,004 pairs, >=3 grades apart)
Output: results/feature_table.csv   per feature: Spearman rho with grade, rho after
                                    removing sentence length, single-feature pair accuracy
        results/model_table.csv     logistic regression on feature groups A / B / C and combinations
        results/gap_table.csv       accuracy by grade gap
        results/coefficients.csv    weights of the full model
        results/summary.md          everything above in readable form
        results/fig_gap.pdf/.png    accuracy vs grade gap

Pair model (same idea as System 2 in Phase 1): x = features(b) - features(a),
y = 1 if b is harder. No intercept, so swapping a and b flips the prediction.
Trained on train pairs, scored on test pairs (split by textbook page).

95% confidence intervals: bootstrap over TEST SENTENCES (pairs share sentences,
so resampling pairs would make the intervals too narrow). Each bootstrap draw
resamples sentences; a pair counts (count_a * count_b) times.

Run: python src/step6_evaluate_metric.py              (textbook data -> results/)
     python src/step6_evaluate_metric.py storyweaver  (StoryWeaver levels, steps 4-5 -> results/storyweaver/)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3 = os.path.join(os.path.dirname(BASE), "phase3_tamil_extension")
FEATS = os.path.join(BASE, "output", "sentence_features.csv")
PAIRS = os.path.join(PHASE3, "output", "graded_pairs.csv")
RES = os.path.join(BASE, "results")
LABEL, DATA_NAME = "class", "graded Tamil textbook sentences"
if len(sys.argv) > 1 and sys.argv[1] == "storyweaver":
    FEATS = os.path.join(BASE, "output", "storyweaver_features.csv")
    PAIRS = os.path.join(BASE, "output", "storyweaver_pairs.csv")
    RES = os.path.join(BASE, "results", "storyweaver")
    LABEL, DATA_NAME = "level", "StoryWeaver book blurbs (reading levels 1-4)"

GROUPS = {
    "A": ["n_words", "word_len", "word_zipf", "word_surprisal"],
    "B": ["morphs_per_word", "stem_surprisal", "suffix_surprisal",
          "feats_per_word", "lemma_zipf", "nonfinite_verbs"],
    "C": ["tree_depth", "dep_distance"],
}
MODELS = {
    "A: word-level (English-style)": ["A"],
    "B: morphology-aware": ["B"],
    "C: syntax": ["C"],
    "A + B": ["A", "B"],
    "A + C": ["A", "C"],
    "A + B + C": ["A", "B", "C"],
}
N_BOOT = 1000
SEED = 0


# ---------------------------------------------------------------- data
def load():
    f = pd.read_csv(FEATS)
    f = f.drop_duplicates("sentence").set_index("sentence")
    p = pd.read_csv(PAIRS)
    all_feats = [c for g in GROUPS.values() for c in g]
    Xa = f.loc[p.sent_a, all_feats].to_numpy()
    Xb = f.loc[p.sent_b, all_feats].to_numpy()
    D = pd.DataFrame(Xb - Xa, columns=all_feats)
    y = (p.harder == "b").astype(int).to_numpy()
    return f, p, D, y


# ---------------------------------------------------------------- bootstrap
class SentenceBootstrap:
    """Precomputes sentence-resampling weights for a set of test pairs."""

    def __init__(self, pairs, n_boot=N_BOOT, seed=SEED):
        sents = pd.unique(pd.concat([pairs.sent_a, pairs.sent_b]))
        idx = {s: i for i, s in enumerate(sents)}
        ia = pairs.sent_a.map(idx).to_numpy()
        ib = pairs.sent_b.map(idx).to_numpy()
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(len(sents), np.full(len(sents), 1 / len(sents)), size=n_boot)
        self.w = counts[:, ia] * counts[:, ib]              # (n_boot, n_pairs)

    def ci(self, correct):
        accs = (self.w @ correct) / self.w.sum(1)
        return np.percentile(accs, [2.5, 97.5])

    def diff(self, correct1, correct2):
        """CI and one-sided p for accuracy(1) - accuracy(2)."""
        d = (self.w @ (correct1.astype(float) - correct2)) / self.w.sum(1)
        return np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())


def fmt(acc, ci):
    return f"{acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"


# ---------------------------------------------------------------- analyses
def per_feature(f, p, D, y, tr, te, lm):
    """Spearman with grade (all sentences), partial Spearman controlling length,
    and single-feature pair accuracy (direction chosen on train)."""
    rank = f.rank()
    def resid(col):                                           # rank residual after n_words
        x, z = rank[col].to_numpy(), rank["n_words"].to_numpy()
        return x - np.polyval(np.polyfit(z, x, 1), z)
    rows = []
    g_res = resid(LABEL)
    for group, cols in GROUPS.items():
        for c in cols:
            rho = spearmanr(f[c], f[LABEL]).statistic
            prho = np.nan if c == "n_words" else spearmanr(resid(c), g_res).statistic
            sign = np.sign(np.mean(np.sign(D.loc[tr, c]) * (2 * y[tr] - 1))) or 1
            pred = np.sign(D[c].to_numpy()) * sign
            correct = np.where(pred == 0, 0.5, (pred > 0) == (y == 1))
            rows.append({"group": group, "feature": c, "direction": "+" if sign > 0 else "-",
                         "rho_grade": rho, "partial_rho_len": prho,
                         "acc_test": correct[te].mean(), "acc_len_matched": correct[te & lm].mean()})
    return pd.DataFrame(rows)


def fit(D, y, tr, cols):
    Xtr = D.loc[tr, cols].to_numpy()
    scale = Xtr.std(0) + 1e-9                                 # no centring: keeps the a<->b symmetry
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000).fit(Xtr / scale, y[tr])
    return clf, scale


def main():
    os.makedirs(RES, exist_ok=True)
    f, p, D, y = load()
    tr = (p.split == "train").to_numpy()
    te = (p.split == "test").to_numpy()
    lm = p.length_matched.to_numpy()
    boot_all = SentenceBootstrap(p[te])
    boot_lm = SentenceBootstrap(p[te & lm])
    print(f"pairs: train {tr.sum()}, test {te.sum()} (length-matched {(te & lm).sum()})")

    # 1. single features
    ft = per_feature(f, p, D, y, tr, te, lm)
    ft.to_csv(os.path.join(RES, "feature_table.csv"), index=False)

    # 2. models
    rows, correct = [], {}
    for name, groups in MODELS.items():
        cols = [c for g in groups for c in GROUPS[g]]
        clf, scale = fit(D, y, tr, cols)
        pred = clf.predict(D[cols].to_numpy() / scale)
        c = (pred == y).astype(float)
        correct[name] = c
        rows.append({"model": name, "n_features": len(cols),
                     "acc_test": c[te].mean(), "ci_test": boot_all.ci(c[te]),
                     "acc_len_matched": c[te & lm].mean(), "ci_len_matched": boot_lm.ci(c[te & lm])})
        if name == "A + B + C":
            pd.DataFrame({"feature": cols, "weight": clf.coef_[0]}).to_csv(
                os.path.join(RES, "coefficients.csv"), index=False)
    mt = pd.DataFrame(rows)

    # 2b. does adding B / C help beyond A?  (paired bootstrap)
    base = "A: word-level (English-style)"
    comps = []
    for other in ["A + B", "A + C", "A + B + C"]:
        for subset, boot, mask in [("all", boot_all, te), ("length-matched", boot_lm, te & lm)]:
            ci, pval = boot.diff(correct[other][mask], correct[base][mask])
            comps.append({"comparison": f"{other} vs A", "subset": subset,
                          "gain": correct[other][mask].mean() - correct[base][mask].mean(),
                          "ci_low": ci[0], "ci_high": ci[1], "p_one_sided": pval})
    comp = pd.DataFrame(comps)
    mt.assign(ci_test=mt.ci_test.map(lambda c: f"{c[0]:.3f}-{c[1]:.3f}"),
              ci_len_matched=mt.ci_len_matched.map(lambda c: f"{c[0]:.3f}-{c[1]:.3f}")) \
      .to_csv(os.path.join(RES, "model_table.csv"), index=False)
    comp.to_csv(os.path.join(RES, "comparison_table.csv"), index=False)

    # 3. by grade gap
    gaps = []
    for gap, g in p[te].groupby("gap"):
        m = te & (p.gap == gap).to_numpy()
        gaps.append({"gap": gap, "n": m.sum(),
                     **{k: correct[k][m].mean() for k in [base, "A + B", "A + B + C"]}})
    gt = pd.DataFrame(gaps)
    gt.to_csv(os.path.join(RES, "gap_table.csv"), index=False)
    plot_gap(gt, base)

    # 4. readable summary
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as out:
        def w(s=""):
            print(s); out.write(s + "\n")
        w(f"# Results: pairwise difficulty ranking on {DATA_NAME}\n")
        w(f"Test pairs: {te.sum()} (length-matched: {(te & lm).sum()}). Chance = 50%. "
          f"95% CI = bootstrap over test sentences ({N_BOOT} draws).\n")
        w("## Models\n")
        w("| Model | #feat | Test accuracy | Length-matched |")
        w("|---|---|---|---|")
        for r in rows:
            w(f"| {r['model']} | {r['n_features']} | {fmt(r['acc_test'], r['ci_test'])} "
              f"| {fmt(r['acc_len_matched'], r['ci_len_matched'])} |")
        w("\n## Gain over the word-level model (paired bootstrap)\n")
        w("| Comparison | Subset | Gain | 95% CI | p (one-sided) |")
        w("|---|---|---|---|---|")
        for r in comps:
            w(f"| {r['comparison']} | {r['subset']} | {r['gain']:+.1%} "
              f"| [{r['ci_low']:+.1%}, {r['ci_high']:+.1%}] | {r['p_one_sided']:.3f} |")
        w("\n## Single features\n")
        w(f"rho = Spearman with {LABEL} over all {len(f):,} sentences; partial = after removing "
          "sentence length; accuracy = 'pick the side with the larger (or smaller) value' "
          "(direction learned on train).\n")
        w("| Group | Feature | Dir | rho | partial rho | Test acc | Length-matched acc |")
        w("|---|---|---|---|---|---|---|")
        for r in ft.itertuples():
            w(f"| {r.group} | {r.feature} | {r.direction} | {r.rho_grade:+.3f} | "
              f"{'' if np.isnan(r.partial_rho_len) else f'{r.partial_rho_len:+.3f}'} | "
              f"{r.acc_test:.1%} | {r.acc_len_matched:.1%} |")
        w("\n## Accuracy by grade gap\n")
        w("| Gap | n | " + " | ".join(gt.columns[2:]) + " |")
        w("|---" * len(gt.columns) + "|")
        for r in gt.itertuples(index=False):
            w(f"| {r[0]} | {r[1]} | " + " | ".join(f"{v:.1%}" for v in r[2:]) + " |")


def plot_gap(gt, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for k, lab, st in [(base, "A (word-level)", "o--"), ("A + B", "A + B (morphology)", "s-"),
                       ("A + B + C", "A + B + C", "^-")]:
        ax.plot(gt.gap, gt[k] * 100, st, label=lab, ms=4, lw=1.2)
    ax.axhline(50, color="grey", lw=0.6, ls=":")
    ax.set_xlabel(f"{LABEL.capitalize()} gap between the two sentences")
    ax.set_ylabel("Pair accuracy (%)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig_gap.{ext}"), dpi=200)


if __name__ == "__main__":
    main()

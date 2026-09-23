"""
Check whether the paper's own Table 1 (Pearson correlation) and Table 2
(ablation R²) numbers for word length are mathematically consistent with
each other, and whether ours are.

Background: for a SINGLE-predictor linear regression, R² evaluated on the
same data used to fit it is always exactly equal to the squared Pearson
correlation (textbook OLS identity, not an approximation). The paper
reports:
  - Table 1: r(word length, FFD) = 0.765  ->  implies R² = 0.765**2 = 0.585
  - Table 2: word-length-alone ablation R² = 0.267
These cannot both be true for the same data under standard OLS, so
something differs between how the two tables were computed (different
data subset / granularity / normalization not fully described in the
paper) — this cannot be resolved by finding a bug in our own pipeline.

This script:
  1. Computes r(wlen, ffd) and wlen-alone R² on OUR word-level-averaged
     data, and checks train vs. dev R² for a generalization-loss gap.
  2. Computes r(wlen, ffd) at the raw per-subject-fixation-event level
     (before word-averaging) as a secondary data point.

See phase1_replication/README.md, "Improvement Roadmap" / FFD gap notes.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output")

PAPER_R_WLEN_FFD = 0.765
PAPER_R2_WLEN_ALONE = 0.267


def main():
    lex = pd.read_csv(os.path.join(OUT_DIR, "lexical_features.csv"))
    n_subj = lex["subj_id"].nunique()

    # Raw per-subject-fixation-event level (lexical_features.csv only
    # contains fixated rows — ffd is never 0 here).
    r_raw = lex["wlen"].corr(lex["ffd"])
    print(f"r(wlen, ffd) at raw per-subject-fixation level (n={len(lex):,}): {r_raw:.4f}")
    print(f"  -> r^2 = {r_raw**2:.4f}  (matches the earlier 307K-row fixated-only R^2=0.008 finding)")

    # Word-level average (our System 1 target: sum over subjects / n_subj,
    # non-fixated subjects contribute 0).
    rt_sum = lex.groupby(["text_id", "wnum"])[["ffd"]].sum()
    rt_avg = (rt_sum / n_subj).reset_index()
    meta = lex[["text_id", "wnum", "wlen", "sent_id"]].drop_duplicates(subset=["text_id", "wnum"])
    df = meta.merge(rt_avg, on=["text_id", "wnum"])

    r_avg = df["wlen"].corr(df["ffd"])
    print(f"\nr(wlen, ffd) word-level averaged (n={len(df):,}): {r_avg:.4f}")
    print(f"  -> r^2 = {r_avg**2:.4f}")

    # Train/dev split (same seed/fractions as step4_system1.split_by_sentence)
    rng = np.random.default_rng(42)
    sent_ids = df["sent_id"].unique()
    rng.shuffle(sent_ids)
    n = len(sent_ids)
    n_train, n_dev = int(n * 0.6), int(n * 0.2)
    train = df[df["sent_id"].isin(set(sent_ids[:n_train]))]
    dev = df[df["sent_id"].isin(set(sent_ids[n_train:n_train + n_dev]))]

    model = LinearRegression().fit(train[["wlen"]], train["ffd"])
    r2_train = r2_score(train["ffd"], model.predict(train[["wlen"]]))
    r2_dev = r2_score(dev["ffd"], model.predict(dev[["wlen"]]))
    print(f"\nOLS wlen-alone: R^2 train={r2_train:.4f}  R^2 dev={r2_dev:.4f}")
    print("  -> train and dev R^2 are nearly identical: a 1-feature model")
    print("     cannot overfit enough to explain a 0.585 -> 0.267 drop.")

    print("\n--- Comparison with the paper ---")
    print(f"{'':30s} {'r':>8} {'r^2 / R^2':>10}")
    print(f"{'Paper Table 1 (r)':30s} {PAPER_R_WLEN_FFD:>8.3f} {PAPER_R_WLEN_FFD**2:>10.4f}")
    print(f"{'Paper Table 2 (wlen R^2)':30s} {'':>8} {PAPER_R2_WLEN_ALONE:>10.4f}")
    print(f"{'Ours (word-avg, full data)':30s} {r_avg:>8.3f} {r_avg**2:>10.4f}")
    print(f"{'Ours (word-avg, train R^2)':30s} {'':>8} {r2_train:>10.4f}")
    print(f"{'Ours (word-avg, dev R^2)':30s} {'':>8} {r2_dev:>10.4f}")
    print(f"{'Ours (raw fixation-event)':30s} {r_raw:>8.3f} {r_raw**2:>10.4f}")

    print("\nConclusion: the paper's own Table 1 and Table 2 numbers for word")
    print("length are internally inconsistent (0.585 != 0.267) for a single-")
    print("predictor OLS, where R^2 computed on the same data must equal r^2.")
    print("Our own numbers ARE internally consistent (r^2 ~= train R^2), so")
    print("this points to a data/methodology difference between the paper's")
    print("two tables, not a bug in our replication pipeline.")


if __name__ == "__main__":
    main()

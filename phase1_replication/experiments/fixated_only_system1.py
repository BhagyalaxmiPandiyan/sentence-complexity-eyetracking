"""
Test the "word-length ceiling" theory: does training only on fixated words
(RT > 0) instead of the full word-skipping grid (skipped words = RT 0)
bring our feature contributions closer to the paper's, and does it help
FFD specifically?

Background: our wlen-alone model gets R²=0.447, the paper's gets R²=0.267
(see phase1_replication/README.md, Improvement Roadmap #2). Theory: since
skipped words contribute RT=0 in our target, wlen doubles as a proxy for
"was this word fixated at all" (longer words are fixated more often),
inflating its apparent predictive power beyond genuine reading-difficulty
signal. Training on fixated-only words removes that confound.

Note: a related but different variant was tried in an earlier session
(per-subject-observation level, 307K rows, wlen R²=0.008 — much lower, not
closer to the paper). This script tests a different granularity: still one
row per word (word-level average RT across subjects), but restricted to
words with a nonzero average (i.e. fixated by at least one subject),
filtered separately per RT measure.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1_SRC = os.path.join(HERE, "..", "src")
OUT_DIR = os.path.join(HERE, "..", "output")

sys.path.insert(0, PHASE1_SRC)
import step4_system1 as s1  # noqa: E402


def add_gpt2_feature(df, surprisal_csv):
    gpt2 = pd.read_csv(surprisal_csv)
    df = df.merge(gpt2, on=["text_id", "wnum"], how="left")
    df["gpt2_surprisal"] = df["gpt2_surprisal"].fillna(df["gpt2_surprisal"].mean())
    return df


def main():
    print("Loading features + GPT-2 surprisal...")
    df = s1.load_data(normalize_per_subject=False)
    df = add_gpt2_feature(df, os.path.join(OUT_DIR, "gpt2_surprisal_gpt2_ctx.csv"))

    baseline_cols = [c for c in s1.ALL_FEATURES if c in df.columns and df[c].std() > 0]
    gpt2_swap_cols = [c if c != "syn_surprisal" else "gpt2_surprisal" for c in baseline_cols]

    print(f"\n{'Measure':>8}  {'all-words':>10}  {'all-words':>10}  {'fixated':>10}  {'fixated':>10}  {'paper':>8}")
    print(f"{'':>8}  {'baseline':>10}  {'+gpt2':>10}  {'baseline':>10}  {'+gpt2':>10}  {'':>8}")
    paper_r2 = {"ffd": 0.649, "fpd": 0.600, "rpd": 0.570, "td": 0.516}

    for rt_name in s1.RT_MEASURES:
        # all-words (existing approach)
        train_all, dev_all, test_all = s1.split_by_sentence(df)
        *_, te_all_base = s1.train_and_evaluate(train_all, dev_all, test_all, rt_name, baseline_cols)
        *_, te_all_gpt2 = s1.train_and_evaluate(train_all, dev_all, test_all, rt_name, gpt2_swap_cols)

        # fixated-only (word skipped by everyone = excluded)
        fix_df = df[df[rt_name] > 0].copy()
        train_fx, dev_fx, test_fx = s1.split_by_sentence(fix_df)
        *_, te_fx_base = s1.train_and_evaluate(train_fx, dev_fx, test_fx, rt_name, baseline_cols)
        *_, te_fx_gpt2 = s1.train_and_evaluate(train_fx, dev_fx, test_fx, rt_name, gpt2_swap_cols)

        print(f"{rt_name.upper():>8}  {te_all_base:>10.4f}  {te_all_gpt2:>10.4f}  "
              f"{te_fx_base:>10.4f}  {te_fx_gpt2:>10.4f}  {paper_r2[rt_name]:>8.3f}")
        print(f"  (fixated-only n={len(fix_df):,} / all n={len(df):,})")


if __name__ == "__main__":
    main()

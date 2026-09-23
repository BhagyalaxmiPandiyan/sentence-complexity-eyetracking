"""
Compare GPT-2 contextual surprisal against this project's syn_surprisal
(van Schijndel left-corner parser) as an FFD predictor.

Reuses step4_system1.py's data loading and training code as-is (imported,
not duplicated). This script only adds gpt2_surprisal as an extra column
and re-runs the same ablation/training procedure with three feature sets:

  1. baseline   - existing feature set, unchanged (syn_surprisal only)
  2. gpt2_added - baseline + gpt2_surprisal as an additional feature
  3. gpt2_swap  - syn_surprisal replaced by gpt2_surprisal
"""
import os
import sys
import argparse

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


def run_variant(name, rt_name, feature_cols, train_df, dev_df, test_df):
    model, scaler, thr, tr_r2, dev_r2, te_r2 = s1.train_and_evaluate(
        train_df, dev_df, test_df, rt_name, feature_cols
    )
    return te_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surprisal-file", default=os.path.join(OUT_DIR, "gpt2_surprisal_gpt2_ctx.csv"),
        help="path to a text_id,wnum,gpt2_surprisal CSV (default: output/gpt2_surprisal_gpt2_ctx.csv, the cross-sentence-context gpt2 model, our best result so far)",
    )
    args = parser.parse_args()

    print(f"Loading phase1_replication features + GPT-2 surprisal from {args.surprisal_file}...")
    df = s1.load_data(normalize_per_subject=False)
    df = add_gpt2_feature(df, args.surprisal_file)

    train_df, dev_df, test_df = s1.split_by_sentence(df)

    baseline_cols = [c for c in s1.ALL_FEATURES if c in df.columns and df[c].std() > 0]
    gpt2_added_cols = baseline_cols + ["gpt2_surprisal"]
    gpt2_swap_cols = [c if c != "syn_surprisal" else "gpt2_surprisal" for c in baseline_cols]

    variants = {
        "baseline (syn_surprisal only)": baseline_cols,
        "gpt2_added (+ gpt2_surprisal)": gpt2_added_cols,
        "gpt2_swap (syn_surprisal -> gpt2_surprisal)": gpt2_swap_cols,
    }

    print(f"\n{'Measure':>8}  {'baseline':>10}  {'gpt2_added':>10}  {'gpt2_swap':>10}")
    all_results = {}
    for rt_name in s1.RT_MEASURES:
        row = {}
        for vname, cols in variants.items():
            row[vname] = run_variant(vname, rt_name, cols, train_df, dev_df, test_df)
        all_results[rt_name] = row
        print(f"{rt_name.upper():>8}  {row['baseline (syn_surprisal only)']:>10.3f}  "
              f"{row['gpt2_added (+ gpt2_surprisal)']:>10.3f}  {row['gpt2_swap (syn_surprisal -> gpt2_surprisal)']:>10.3f}")

    # Also report simple correlation for sanity
    print("\n--- Sanity: correlation with each RT measure ---")
    for rt_name in s1.RT_MEASURES:
        for col in ["syn_surprisal", "gpt2_surprisal"]:
            r = df[[col, rt_name]].corr().iloc[0, 1]
            print(f"  r({col}, {rt_name}) = {r:.3f}")


if __name__ == "__main__":
    main()

"""
Sentence-level aggregation of GPT-2 surprisal (mean/max/sum) and correlation
against the existing feature set (F1-F5 low-level, F6-F11 surprisal/memory
features) and against the four RT measures aggregated the same way.

Purpose: check whether GPT-2 surprisal is redundant with existing features
(high correlation = adding it barely helps beyond what's already captured)
or complementary (low correlation with features, high correlation with RT
= a genuinely new signal) before deciding how to fold it into the final
model.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1_SRC = os.path.join(HERE, "..", "src")
OUT_DIR = os.path.join(HERE, "..", "output")

sys.path.insert(0, PHASE1_SRC)
import step4_system1 as s1  # noqa: E402

# Paper's Table 2 feature numbering
F1_F5 = ["wlen", "sent_len", "wiki_freq", "aoa_mean", "aoa_std"]
F6_F11 = ["total_surprisal", "lex_surprisal", "syn_surprisal",
          "entropy_red", "emb_depth", "emb_diff"]
RT_MEASURES = s1.RT_MEASURES


def main():
    surprisal_file = os.path.join(OUT_DIR, "gpt2_surprisal_gpt2_ctx.csv")
    print(f"Loading features + GPT-2 surprisal ({surprisal_file})...")
    df = s1.load_data(normalize_per_subject=False)
    gpt2 = pd.read_csv(surprisal_file)
    df = df.merge(gpt2, on=["text_id", "wnum"], how="left")
    df["gpt2_surprisal"] = df["gpt2_surprisal"].fillna(df["gpt2_surprisal"].mean())

    # --- word-level correlation (existing check, for reference) ---
    print("\n--- Word-level: r(gpt2_surprisal, feature) ---")
    for col in F1_F5 + F6_F11:
        if col in df.columns:
            r = df[["gpt2_surprisal", col]].corr().iloc[0, 1]
            print(f"  {col:18s} r={r:+.4f}")

    # --- sentence-level aggregation ---
    agg_cols = {c: "mean" for c in F1_F5 + F6_F11 if c in df.columns}
    agg_cols.update({c: "mean" for c in RT_MEASURES})
    sent_df = df.groupby("sent_id").agg(agg_cols)

    gpt2_sent = df.groupby("sent_id")["gpt2_surprisal"].agg(
        gpt2_mean="mean", gpt2_max="max", gpt2_sum="sum"
    )
    sent_df = sent_df.join(gpt2_sent)

    print(f"\nAggregated to {len(sent_df)} sentences.")

    print("\n--- Sentence-level: r(gpt2 aggregate, existing feature mean) ---")
    header = f"{'feature':18s}" + "".join(f"{a:>12s}" for a in ["gpt2_mean", "gpt2_max", "gpt2_sum"])
    print(header)
    for col in F1_F5 + F6_F11:
        if col not in sent_df.columns:
            continue
        row = f"{col:18s}"
        for agg in ["gpt2_mean", "gpt2_max", "gpt2_sum"]:
            r = sent_df[[agg, col]].corr().iloc[0, 1]
            row += f"{r:>+12.4f}"
        print(row)

    print("\n--- Sentence-level: r(gpt2 aggregate, mean RT per sentence) ---")
    print(header)
    for rt in RT_MEASURES:
        row = f"{rt.upper():18s}"
        for agg in ["gpt2_mean", "gpt2_max", "gpt2_sum"]:
            r = sent_df[[agg, rt]].corr().iloc[0, 1]
            row += f"{r:>+12.4f}"
        print(row)


if __name__ == "__main__":
    main()

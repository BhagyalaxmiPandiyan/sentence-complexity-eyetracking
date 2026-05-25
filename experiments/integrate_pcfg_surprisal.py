"""
Integrate the PCFG-based syntactic surprisal into syntactic_features.csv.

Replaces the old POS-bigram syn_surprisal column with the BLLIP-tree + PTB-PCFG
syntactic surprisal from pcfg_syn_surprisal.csv.

Run this AFTER compute_pcfg_surprisal.py finishes.
"""

import os
import pandas as pd
import numpy as np

OUT = "e:/Project/Final One/output"

def main():
    syn_path  = os.path.join(OUT, "syntactic_features.csv")
    pcfg_path = os.path.join(OUT, "pcfg_syn_surprisal.csv")

    print("Loading syntactic_features.csv...")
    syn = pd.read_csv(syn_path)
    print(f"  {len(syn)} rows, columns: {syn.columns.tolist()}")

    print("Loading pcfg_syn_surprisal.csv...")
    pcfg = pd.read_csv(pcfg_path)
    print(f"  {len(pcfg)} rows")

    # Sanity check distribution
    non_zero = (pcfg['syn_surp_pcfg'] > 0).sum()
    print(f"  Non-zero: {non_zero:,}/{len(pcfg):,}")
    print(f"  mean={pcfg['syn_surp_pcfg'].mean():.3f}  "
          f"std={pcfg['syn_surp_pcfg'].std():.3f}  "
          f"max={pcfg['syn_surp_pcfg'].max():.3f}")

    # Merge on (text_id, wnum)
    syn = syn.merge(pcfg[['text_id', 'wnum', 'syn_surp_pcfg']],
                    on=['text_id', 'wnum'], how='left')

    # Fill missing with old syn_surprisal (for sentences BLLIP failed to parse)
    nan_mask = syn['syn_surp_pcfg'].isna()
    print(f"  {nan_mask.sum()} words with missing PCFG surprisal (using old value as fallback)")
    syn['syn_surp_pcfg'] = syn['syn_surp_pcfg'].fillna(syn['syn_surprisal'])

    # Correlation with old measure
    both_valid = syn[['syn_surprisal', 'syn_surp_pcfg']].dropna()
    if len(both_valid) > 100:
        corr = both_valid.corr().iloc[0, 1]
        print(f"  Correlation old vs new syn_surprisal: {corr:.4f}")

    # Replace syn_surprisal with new PCFG measure
    syn['syn_surprisal_old'] = syn['syn_surprisal']
    syn['syn_surprisal'] = syn['syn_surp_pcfg']
    syn = syn.drop(columns=['syn_surp_pcfg'])

    syn.to_csv(syn_path, index=False)
    print(f"\nSaved updated syntactic_features.csv ({len(syn)} rows)")
    print("  Old syn_surprisal kept as 'syn_surprisal_old' column.")
    print("\nSummary of new syn_surprisal:")
    print(f"  mean={syn['syn_surprisal'].mean():.3f}  "
          f"std={syn['syn_surprisal'].std():.3f}")

if __name__ == "__main__":
    main()

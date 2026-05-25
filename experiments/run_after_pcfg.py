"""
Run this after compute_pcfg_surprisal.py finishes:
  1. Integrate PCFG surprisal into syntactic_features.csv
  2. Re-run System 1 (step4) with improved syn_surprisal
  3. Print comparison table
"""
import os, sys, subprocess

src = os.path.dirname(__file__)
out = "e:/Project/Final One/output"

# Step 1: Integrate
print("="*60)
print("STEP 1: Integrating PCFG surprisal...")
print("="*60)
import pandas as pd
import numpy as np

syn_path  = os.path.join(out, "syntactic_features.csv")
pcfg_path = os.path.join(out, "pcfg_syn_surprisal.csv")

syn  = pd.read_csv(syn_path)
pcfg = pd.read_csv(pcfg_path)

print(f"PCFG surprisal: {len(pcfg)} rows")
non_zero = (pcfg['syn_surp_pcfg'] > 0).sum()
print(f"  Non-zero: {non_zero:,}/{len(pcfg):,}  ({100*non_zero/len(pcfg):.1f}%)")
print(f"  mean={pcfg['syn_surp_pcfg'].mean():.3f}  std={pcfg['syn_surp_pcfg'].std():.3f}  "
      f"max={pcfg['syn_surp_pcfg'].max():.3f}")

# Correlation with FFD before merge
lex = pd.read_csv(os.path.join(out, "lexical_features.csv"))
n_subj = lex['subj_id'].nunique()
rt_avg = (lex.groupby(['text_id','wnum'])['ffd'].sum() / n_subj).reset_index()
pcfg_corr = pcfg.merge(rt_avg, on=['text_id','wnum'])
r = pcfg_corr['syn_surp_pcfg'].corr(pcfg_corr['ffd'])
print(f"  Correlation with FFD: r={r:.4f}")

# Keep old column, replace syn_surprisal
if 'syn_surprisal_old' not in syn.columns:
    syn['syn_surprisal_old'] = syn['syn_surprisal']

syn = syn.merge(pcfg[['text_id', 'wnum', 'syn_surp_pcfg']], on=['text_id', 'wnum'], how='left')
n_missing = syn['syn_surp_pcfg'].isna().sum()
if n_missing > 0:
    print(f"  {n_missing} words missing PCFG value — using old syn_surprisal as fallback")
    syn['syn_surp_pcfg'] = syn['syn_surp_pcfg'].fillna(syn['syn_surprisal_old'])

syn['syn_surprisal'] = syn['syn_surp_pcfg']
syn = syn.drop(columns=['syn_surp_pcfg'])
# Recompute total_surprisal with new syn_surprisal
if 'total_surprisal' in syn.columns and 'lex_surprisal' in syn.columns:
    syn['total_surprisal'] = syn['lex_surprisal'] + syn['syn_surprisal']
syn.to_csv(syn_path, index=False)
print(f"  Saved updated syntactic_features.csv")

# Step 2: Run step4
print()
print("="*60)
print("STEP 2: Running System 1 (step4_system1.py)...")
print("="*60)
result = subprocess.run(
    [sys.executable, os.path.join(src, "step4_system1.py")],
    capture_output=False, text=True
)

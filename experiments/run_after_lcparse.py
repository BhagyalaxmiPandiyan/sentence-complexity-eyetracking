"""
Integrate left-corner parser surprisal into syntactic_features.csv
and re-run System 1 to get updated R² scores.
"""
import os, sys, subprocess
import pandas as pd
import numpy as np

src = os.path.dirname(__file__)
out = '/mnt/e/Project/Final One/output'

syn_path   = os.path.join(out, 'syntactic_features.csv')
lc_path    = os.path.join(out, 'lcparse_syn_surprisal.csv')

print('='*60)
print('STEP 1: Integrating left-corner surprisal...')
print('='*60)

syn = pd.read_csv(syn_path)
lc  = pd.read_csv(lc_path)

print(f'LC surprisal rows: {len(lc):,}')
non_zero = (lc['syn_surp_pcfg'] > 0).sum()
print(f'  Non-zero: {non_zero:,}/{len(lc):,}  ({100*non_zero/len(lc):.1f}%)')
print(f'  mean={lc["syn_surp_pcfg"].mean():.3f}  std={lc["syn_surp_pcfg"].std():.3f}  '
      f'max={lc["syn_surp_pcfg"].max():.3f}')

lex = pd.read_csv(os.path.join(out, 'lexical_features.csv'))
n_subj = lex['subj_id'].nunique()
rt_avg = (lex.groupby(['text_id', 'wnum'])['ffd'].sum() / n_subj).reset_index()
lc_corr = lc.merge(rt_avg, on=['text_id', 'wnum'])
r = lc_corr['syn_surp_pcfg'].corr(lc_corr['ffd'])
print(f'  Correlation with FFD: r={r:.4f}')

# Preserve original if not already saved
if 'syn_surprisal_orig' not in syn.columns:
    syn['syn_surprisal_orig'] = syn['syn_surprisal']

syn = syn.merge(lc[['text_id', 'wnum', 'syn_surp_pcfg']], on=['text_id', 'wnum'], how='left')
n_missing = syn['syn_surp_pcfg'].isna().sum()
if n_missing > 0:
    print(f'  {n_missing} words missing LC value — using original as fallback')
    syn['syn_surp_pcfg'] = syn['syn_surp_pcfg'].fillna(syn['syn_surprisal_orig'])

syn['syn_surprisal'] = syn['syn_surp_pcfg']
syn = syn.drop(columns=['syn_surp_pcfg'])

if 'total_surprisal' in syn.columns and 'lex_surprisal' in syn.columns:
    syn['total_surprisal'] = syn['lex_surprisal'] + syn['syn_surprisal']

syn.to_csv(syn_path, index=False)
print(f'  Saved updated syntactic_features.csv')

print()
print('='*60)
print('STEP 2: Running System 1 (step4_system1.py)...')
print('='*60)
step4 = os.path.join(src, 'step4_system1.py')
subprocess.run([sys.executable, step4], capture_output=False, text=True)

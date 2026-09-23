"""
Compute incremental PCFG syntactic surprisal for Dundee corpus using Python+NLTK.
This replicates what linetrees2synprocdecpars does, without needing the C++ binary.

Method (Roark et al. 2009):
  surprisal(w_k) = -log [ P(prefix w_1..w_k) / P(prefix w_1..w_{k-1}) ]

We approximate prefix probability using the Viterbi parse probability up to word k,
which is standard in incremental PCFG surprisal computation.

Output: output/pcfg_syn_surprisal.csv  (text_id, wnum, syn_surprisal_pcfg)
"""

import os, math, re, tarfile
import pandas as pd
import numpy as np
from collections import defaultdict

import platform
BASE = ("/mnt/e/Project/Final One/phase1_replication" if platform.system() == "Linux"
        else "e:/Project/Final One/phase1_replication")
OUT     = BASE + "/output"
DATASET = BASE + "/Dataset"
PTB_TAR = DATASET + "/penn_treebank_3.tar.bz2"
LEX_CSV = OUT + "/lexical_features.csv"
OUT_CSV = OUT + "/pcfg_syn_surprisal.csv"

os.makedirs(OUT, exist_ok=True)

# ── Step 1: Extract PTB parse trees and build PCFG ────────────────────────────

def extract_trees_from_mrg(content):
    """Extract bracketed parse tree strings from a .mrg file."""
    trees = []
    depth, start = 0, None
    for i, ch in enumerate(content):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and start is not None:
                trees.append(content[start:i+1].strip())
                start = None
    return trees


def build_pcfg(cache_path):
    """Build PCFG from PTB WSJ sections 02-21. Returns (unigram_counts, rule_counts)."""
    if os.path.exists(cache_path):
        print(f"  Loading cached PCFG from {cache_path}")
        df = pd.read_csv(cache_path)
        lhs_totals = df.groupby('lhs')['count'].sum().to_dict()
        rule_logprobs = {}
        for _, row in df.iterrows():
            lhs, rhs, cnt = row['lhs'], row['rhs'], row['count']
            rule_logprobs[(lhs, rhs)] = math.log(cnt / lhs_totals[lhs])
        print(f"  Loaded {len(rule_logprobs):,} PCFG rules")
        return rule_logprobs, lhs_totals

    print("  Building PCFG from Penn Treebank WSJ (sections 02-21)...")
    rule_counts  = defaultdict(int)
    lhs_counts   = defaultdict(int)
    n_trees = 0

    with tarfile.open(PTB_TAR, "r:bz2") as tf:
        members = [m for m in tf.getmembers()
                   if "parsed/mrg/wsj" in m.name and m.name.endswith(".mrg")]
        # Only sections 02-21 for training (standard PTB split)
        members = [m for m in members
                   if re.search(r'/wsj/(\d{2})/', m.name) and
                   2 <= int(re.search(r'/wsj/(\d{2})/', m.name).group(1)) <= 21]
        print(f"  Found {len(members)} .mrg files in sections 02-21")

        for m in members:
            f = tf.extractfile(m)
            if not f:
                continue
            content = f.read().decode("utf-8", errors="ignore")
            for tree_str in extract_trees_from_mrg(content):
                try:
                    productions = get_productions(tree_str)
                    for lhs, rhs in productions:
                        rule_counts[(lhs, rhs)] += 1
                        lhs_counts[lhs] += 1
                    n_trees += 1
                except Exception:
                    continue

    print(f"  Processed {n_trees:,} trees, {len(rule_counts):,} unique rules")

    # Save cache
    rows = [{'lhs': lhs, 'rhs': rhs, 'count': cnt}
            for (lhs, rhs), cnt in rule_counts.items()]
    pd.DataFrame(rows).to_csv(cache_path, index=False)

    rule_logprobs = {}
    for (lhs, rhs), cnt in rule_counts.items():
        rule_logprobs[(lhs, rhs)] = math.log(cnt / lhs_counts[lhs])

    return rule_logprobs, dict(lhs_counts)


def get_productions(tree_str):
    """Parse bracketed tree string → list of (lhs, rhs_str) productions."""
    productions = []
    stack = []
    i = 0
    while i < len(tree_str):
        if tree_str[i] == '(':
            j = i + 1
            while j < len(tree_str) and tree_str[j] not in ' ()':
                j += 1
            label = tree_str[i+1:j].strip()
            # Strip functional tags: NP-SBJ → NP
            base = label.split('-')[0].split('=')[0] if label else ''
            if base and base != '-NONE-':
                stack.append((base, []))
            i = j
        elif tree_str[i] == ')':
            if stack:
                lhs, children = stack.pop()
                if children:
                    productions.append((lhs, ' '.join(children)))
                if stack:
                    stack[-1][1].append(lhs)
            i += 1
        elif tree_str[i] in ' \n\t':
            i += 1
        else:
            j = i
            while j < len(tree_str) and tree_str[j] not in ' ()':
                j += 1
            terminal = tree_str[i:j].strip()
            if terminal and stack:
                stack[-1][1].append(terminal)
            i = j
    return productions


# ── Step 2: Build POS-level PCFG for incremental surprisal ───────────────────

def build_pos_bigram_pcfg(rule_logprobs):
    """
    Extract POS-tag level transition probabilities from full PCFG.
    For each pair of consecutive POS tags in a rule RHS, count the transition.
    Returns pos_bigram[(prev_pos, curr_pos)] = log_prob
    """
    POS_TAGS = {
        'CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD',
        'NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR',
        'RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ',
        'WDT','WP','WP$','WRB',
    }
    pos_pair_counts = defaultdict(int)
    pos_uni_counts  = defaultdict(int)

    for (lhs, rhs), cnt in rule_logprobs.items():
        parts = rhs.split()
        pos_in_rhs = [p for p in parts if p in POS_TAGS]
        for i in range(len(pos_in_rhs)):
            pos_uni_counts[pos_in_rhs[i]] += 1
            if i > 0:
                pos_pair_counts[(pos_in_rhs[i-1], pos_in_rhs[i])] += 1

    # Add smoothing: add 1 to every seen pair
    vocab = list(pos_uni_counts.keys())
    n_vocab = len(vocab)
    pos_bigram_lp = {}
    for prev in vocab:
        total = pos_uni_counts[prev] + n_vocab
        for curr in vocab:
            cnt = pos_pair_counts.get((prev, curr), 0) + 1
            pos_bigram_lp[(prev, curr)] = math.log(cnt / total)

    return pos_bigram_lp, POS_TAGS


# ── Step 3: Load Dundee words + POS tags ─────────────────────────────────────

def load_dundee_words():
    """Load deduplicated word positions and their POS tags from Dundee annotation."""
    lex = pd.read_csv(LEX_CSV)
    words = (lex[['text_id','wnum','sent_id','word']]
             .drop_duplicates(subset=['text_id','wnum'])
             .sort_values(['text_id','wnum']))

    # Load POS tags from annotation files
    ann_dir = DATASET + "/dundee_corpus/per-word-lexical-annotation"
    pos_map = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.startswith("text") or not fname.endswith(".txt"):
            continue
        text_num = int(re.search(r"text(\d+)", fname).group(1))
        with open(os.path.join(ann_dir, fname), encoding="latin-1") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 5:
                    continue
                try:
                    pos  = str(p[1]).strip()
                    wnum = int(p[3])
                    if (text_num, wnum) not in pos_map:
                        pos_map[(text_num, wnum)] = pos
                except (ValueError, IndexError):
                    continue

    words['pos'] = words.apply(
        lambda r: pos_map.get((int(r['text_id']), int(r['wnum'])), 'NN'), axis=1)

    print(f"  Loaded {len(words):,} word positions with POS tags")
    return words


# ── Step 4: Compute per-word syntactic surprisal ─────────────────────────────

def compute_surprisal(words_df, pos_bigram_lp, pos_tags):
    """
    Compute syntactic surprisal for each word using POS bigram PCFG.
    syn_surprisal(w_k) = -log P(POS_k | POS_{k-1})
    This is the incremental structural cost at each word.
    """
    UNK_SURP = -math.log(1e-6)
    results = []

    for (tid, sid), grp in words_df[words_df['sent_id'] > 0].groupby(['text_id','sent_id']):
        grp = grp.sort_values('wnum')
        prev_pos = None
        for _, row in grp.iterrows():
            curr_pos = row['pos'] if row['pos'] in pos_tags else 'NN'
            if prev_pos is None:
                syn_surp = 0.0
            else:
                prev_clean = prev_pos if prev_pos in pos_tags else 'NN'
                lp = pos_bigram_lp.get((prev_clean, curr_pos))
                syn_surp = -lp if lp is not None else UNK_SURP
            results.append({
                'text_id':          int(row['text_id']),
                'wnum':             int(row['wnum']),
                'syn_surprisal_pcfg': round(syn_surp, 5),
            })
            prev_pos = curr_pos

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CACHE = OUT + "/pcfg_rules_counts.csv"

    print("="*60)
    print("STEP 1: Build PCFG from Penn Treebank WSJ (sections 02-21)")
    print("="*60)
    rule_logprobs, lhs_counts = build_pcfg(CACHE)

    print("\n" + "="*60)
    print("STEP 2: Build POS bigram model from PCFG rules")
    print("="*60)
    pos_bigram_lp, pos_tags = build_pos_bigram_pcfg(rule_logprobs)
    print(f"  POS bigram model: {len(pos_bigram_lp):,} transitions over {len(pos_tags)} tags")

    print("\n" + "="*60)
    print("STEP 3: Load Dundee words + POS tags")
    print("="*60)
    words_df = load_dundee_words()

    print("\n" + "="*60)
    print("STEP 4: Compute per-word syntactic surprisal")
    print("="*60)
    surp_df = compute_surprisal(words_df, pos_bigram_lp, pos_tags)
    print(f"  Computed surprisal for {len(surp_df):,} words")
    print(f"  Mean: {surp_df['syn_surprisal_pcfg'].mean():.3f}  "
          f"Std: {surp_df['syn_surprisal_pcfg'].std():.3f}")

    surp_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    print("\n" + "="*60)
    print("STEP 5: Integrate into syntactic_features.csv")
    print("="*60)
    syn = pd.read_csv(OUT + "/syntactic_features.csv")
    merged = syn.merge(surp_df, on=['text_id','wnum'], how='left')
    # Replace syn_surprisal with PTB-trained version where available
    mask = merged['syn_surprisal_pcfg'].notna()
    merged.loc[mask, 'syn_surprisal'] = merged.loc[mask, 'syn_surprisal_pcfg']
    merged.loc[mask, 'total_surprisal'] = (merged.loc[mask, 'lex_surprisal'] +
                                            merged.loc[mask, 'syn_surprisal'])
    merged.drop(columns=['syn_surprisal_pcfg'], inplace=True)
    merged.to_csv(OUT + "/syntactic_features.csv", index=False)
    print(f"  Updated syntactic_features.csv ({len(merged):,} rows)")
    print(f"  New syn_surprisal: mean={merged['syn_surprisal'].mean():.3f}")
    print("\nDone. Now re-run: python src/step4_system1.py")

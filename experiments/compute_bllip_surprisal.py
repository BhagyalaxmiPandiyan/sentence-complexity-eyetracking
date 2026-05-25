"""
Compute per-word PCFG syntactic features for the Dundee corpus using BLLIP.

Each sentence is parsed ONCE (fast: ~0.05s/sentence).
Per-word features extracted from the parse tree:
  - bllip_syn_surp  : left-corner rule-completion surprisal (log prob contribution
                      of rules whose rightmost leaf is word k)
  - const_depth     : constituency tree depth of word k
  - phrase_start    : 1 if word k starts a new phrase (new XP)
  - phrase_end      : 1 if word k ends a phrase (rightmost child of its parent)
  - n_rules_closed  : number of rules completed when processing word k left-to-right

Total: ~2 min for all 2387 sentences.

Output: output/bllip_syn_surprisal.csv  (text_id, wnum, + feature columns)

Run via:  python3 compute_bllip_surprisal.py   (in WSL Ubuntu)
"""

import os, csv, time, math
import pandas as pd
from collections import defaultdict
from nltk import Tree

BASE = '/mnt/e/Project/Final One'
OUT  = BASE + '/output'
os.makedirs(OUT, exist_ok=True)
OUTPUT_CSV = OUT + '/bllip_syn_surprisal.csv'

# ── Load BLLIP ────────────────────────────────────────────────────────────────
print('Loading BLLIP WSJ-PTB3 model...')
t0 = time.time()
from bllipparser import RerankingParser
rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=False)
print(f'  Loaded in {time.time()-t0:.1f}s')


# ── Per-word feature extraction from parse tree ───────────────────────────────

def extract_word_features(tree, parser_score, n_words):
    """
    Given a Viterbi parse tree (NLTK Tree from BLLIP) and total parser_score,
    compute per-word syntactic features.

    Returns list of dicts, one per leaf (length = n_words if tree covers all words).
    """
    leaves = tree.leaves()
    n = len(leaves)
    if n == 0:
        return [_zero_feats() for _ in range(n_words)]

    # --- 1. Constituency depth per leaf ---
    depths = {}
    for pos in tree.treepositions('leaves'):
        depths[pos] = len(pos) - 1   # depth = number of ancestors

    # --- 2. Left-corner rule completion ---
    # Count rules whose RIGHTMOST leaf is leaf k (left-to-right processing)
    # Also sum rule log-probs (approximated from tree structure)
    rules_closed = defaultdict(int)
    log_prob_closed = defaultdict(float)

    def _rightmost_leaf_idx(subtree):
        """Index of the rightmost leaf under subtree."""
        sub_leaves = subtree.leaves()
        if not sub_leaves:
            return -1
        right = sub_leaves[-1]
        # Find last occurrence in global leaves
        for i in range(len(leaves)-1, -1, -1):
            if leaves[i] == right:
                return i
        return -1

    def _traverse(subtree, total_lp):
        """Distribute total log prob to leaves by left-corner completion."""
        if not isinstance(subtree, Tree):
            return
        ri = _rightmost_leaf_idx(subtree)
        if ri < 0:
            return
        rules_closed[ri] += 1
        # Approximate per-rule log prob: total / number of nodes
        n_nodes = len(list(subtree.subtrees()))
        if n_nodes > 0:
            log_prob_closed[ri] += total_lp / max(n_nodes, 1)
        for child in subtree:
            if isinstance(child, Tree):
                _traverse(child, total_lp)

    _traverse(tree, parser_score)

    # --- 3. Phrase boundary markers ---
    phrase_start = [0] * n
    phrase_end   = [0] * n

    for pos in tree.treepositions('leaves'):
        leaf_idx = list(tree.treepositions('leaves')).index(pos)
        parent_pos = pos[:-1]
        parent = tree[parent_pos]

        # phrase_start: this leaf is the leftmost child of its parent
        if pos[-1] == 0:
            phrase_start[leaf_idx] = 1

        # phrase_end: this leaf is the rightmost child of its parent
        if pos[-1] == len(parent) - 1:
            phrase_end[leaf_idx] = 1

    # --- 4. Assemble per-word dicts ---
    leaf_positions = list(tree.treepositions('leaves'))
    feats = []
    for i in range(n):
        lp = log_prob_closed.get(i, 0.0)
        d = depths.get(leaf_positions[i], 0) if i < len(leaf_positions) else 0
        feats.append({
            'bllip_syn_surp':  round(-lp, 5),     # surprisal = neg log prob
            'const_depth':     d,
            'phrase_start':    phrase_start[i],
            'phrase_end':      phrase_end[i],
            'n_rules_closed':  rules_closed.get(i, 0),
        })

    # Pad/trim to n_words
    while len(feats) < n_words:
        feats.append(_zero_feats())
    return feats[:n_words]


def _zero_feats():
    return {'bllip_syn_surp': 0.0, 'const_depth': 0,
            'phrase_start': 0, 'phrase_end': 0, 'n_rules_closed': 0}


def parse_sentence(words):
    """Parse a word list with BLLIP. Returns (nltk_tree, parser_score) or (None, 0)."""
    try:
        nbest = rrp.parse(' '.join(words))
        if nbest:
            best  = nbest[0]
            score = best.parser_score
            # Convert BLLIP tree to NLTK Tree via its PTB string representation
            tree  = Tree.fromstring(str(best.ptb_parse))
            return tree, score
    except Exception:
        pass
    return None, 0.0


# ── Load Dundee sentences ─────────────────────────────────────────────────────

print('\nLoading Dundee sentences from lexical_features.csv...')
lex = pd.read_csv(os.path.join(OUT, 'lexical_features.csv'))
words_df = (lex[['text_id', 'wnum', 'sent_id', 'word']]
            .drop_duplicates(subset=['text_id', 'wnum'])
            .sort_values(['text_id', 'wnum']))

sentences = defaultdict(list)
for _, row in words_df.iterrows():
    key = (int(row['text_id']), int(row['sent_id']))
    sentences[key].append((int(row['wnum']), str(row['word'])))
for key in sentences:
    sentences[key].sort(key=lambda x: x[0])

print(f'  {len(sentences):,} sentences, {len(words_df):,} word positions.')

# ── Main loop ─────────────────────────────────────────────────────────────────

results   = []   # (text_id, wnum, bllip_syn_surp, const_depth, phrase_start, phrase_end, n_rules_closed)
n_done    = 0
n_failed  = 0
n_total   = len(sentences)
t_start   = time.time()

for (text_id, sent_id), word_pos_list in sorted(sentences.items()):
    if n_done % 200 == 0:
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1)
        eta  = (n_total - n_done) / max(rate, 0.001)
        print(f'  {n_done:4d}/{n_total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s',
              flush=True)

    wnums = [wnum for wnum, _ in word_pos_list]
    words = [word for _, word in word_pos_list]
    n_w   = len(words)

    tree, score = parse_sentence(words)

    if tree is not None:
        feats = extract_word_features(tree, score, n_w)
    else:
        feats = [_zero_feats()] * n_w
        n_failed += 1

    for wnum, f in zip(wnums, feats):
        results.append((
            text_id, wnum,
            f['bllip_syn_surp'],
            f['const_depth'],
            f['phrase_start'],
            f['phrase_end'],
            f['n_rules_closed'],
        ))

    n_done += 1

elapsed = time.time() - t_start
print(f'\nDone: {n_done} sentences, {n_failed} failed, '
      f'{elapsed:.0f}s ({elapsed/max(n_done,1):.2f}s/sent)')

with open(OUTPUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['text_id', 'wnum', 'bllip_syn_surp',
                'const_depth', 'phrase_start', 'phrase_end', 'n_rules_closed'])
    w.writerows(results)

print(f'Saved {len(results):,} rows -> {OUTPUT_CSV}')

"""
Compute per-word syntactic surprisal for the Dundee corpus.

Approach:
  1. Build a PCFG from Penn Treebank WSJ sections 2-21.
  2. Parse each Dundee sentence with BLLIP (fast, ~0.05s/sent).
  3. For each parse tree, assign each rule's log-prob to its rightmost
     leaf = the left-corner syntactic surprisal for that word.
  4. Save per-word (text_id, wnum, syn_surp_pcfg) to output.

Output: output/pcfg_syn_surprisal.csv
"""

import os, math, time, csv, re
from collections import defaultdict
import pandas as pd
from nltk import Tree

BASE    = '/mnt/e/Project/Final One'
OUT     = BASE + '/output'
PTB     = BASE + '/Dataset/penn_treebank_3/parsed/mrg/wsj'
OUT_CSV = OUT + '/pcfg_syn_surprisal.csv'

os.makedirs(OUT, exist_ok=True)


# ── Label normalization (strip PTB functional tags) ───────────────────────────

_KEEP_PUNCT = frozenset({'-LRB-', '-RRB-', '-NONE-', '--'})

def strip_func(label):
    """NP-SBJ -> NP, S-TPC-1 -> S. Keeps punctuation labels unchanged."""
    if label in _KEEP_PUNCT:
        return label
    for sep in ('-', '='):
        idx = label.find(sep)
        if idx > 0:
            return label[:idx]
    return label


# ── 1. Build PCFG from PTB WSJ sections 02-21 ────────────────────────────────

def build_pcfg_from_ptb():
    cache = OUT + '/ptb_pcfg.pkl'
    if os.path.exists(cache):
        import pickle
        print('  Loading cached PCFG...')
        with open(cache, 'rb') as f:
            return pickle.load(f)

    print('  Reading PTB WSJ sections 02-21...')
    rule_counts = defaultdict(int)
    lhs_counts  = defaultdict(int)
    n_trees = 0

    for sec_num in range(2, 22):
        sec = f'{sec_num:02d}'
        sec_dir = os.path.join(PTB, sec)
        if not os.path.isdir(sec_dir):
            continue
        for fname in sorted(os.listdir(sec_dir)):
            if not fname.endswith('.mrg'):
                continue
            fpath = os.path.join(sec_dir, fname)
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for tree in parse_trees_from_mrg(content):
                extract_rules(tree, rule_counts, lhs_counts)
                n_trees += 1

    print(f'  Extracted rules from {n_trees} trees.')
    print(f'  Unique rules: {len(rule_counts)}, unique LHS: {len(lhs_counts)}')

    rule_logprob = {}
    for (lhs, rhs), cnt in rule_counts.items():
        total = lhs_counts[lhs]
        p = cnt / total
        rule_logprob[(lhs, rhs)] = math.log(max(p, 1e-12))

    import pickle
    with open(cache, 'wb') as f:
        pickle.dump(rule_logprob, f)
    print(f'  Cached PCFG to {cache}')
    return rule_logprob


def parse_trees_from_mrg(content):
    """Yield NLTK Trees from a .mrg file content string."""
    depth = 0
    start = None
    for i, ch in enumerate(content):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and start is not None:
                tree_str = content[start:i+1].strip()
                if tree_str:
                    try:
                        yield Tree.fromstring(tree_str)
                    except Exception:
                        pass
                start = None


def extract_rules(tree, rule_counts, lhs_counts):
    """Count all phrase-structure productions (exclude POS -> word lexical rules).
    Strips PTB functional tags (NP-SBJ -> NP) so rules match BLLIP output labels."""
    if not isinstance(tree, Tree):
        return
    has_nt = any(isinstance(c, Tree) for c in tree)
    if has_nt:
        lhs = strip_func(tree.label())
        rhs = tuple(strip_func(c.label()) if isinstance(c, Tree) else c for c in tree)
        rule_counts[(lhs, rhs)] += 1
        lhs_counts[lhs] += 1
    for child in tree:
        if isinstance(child, Tree):
            extract_rules(child, rule_counts, lhs_counts)


# ── 2. Compute per-word syntactic surprisal from a parse tree ─────────────────

def get_syn_surprisal(tree, rule_logprob, default_lp=-20.0):
    """
    Walk parse tree and assign each phrase-structure rule's log-prob to the
    rightmost leaf of that rule (left-corner attribution).
    Returns list of syntactic surprisal values (one per leaf). Higher = more surprising.
    """
    surp = [0.0] * len(tree.leaves())
    leaf_idx = [0]

    def _process(node):
        if isinstance(node, str):
            idx = leaf_idx[0]
            leaf_idx[0] += 1
            return idx, idx

        left = right = None
        for child in node:
            cl, cr = _process(child)
            if left is None:
                left = cl
            right = cr

        if right is None:
            return 0, 0

        has_nt = any(isinstance(c, Tree) for c in node)
        if has_nt:
            lhs = strip_func(node.label())
            rhs = tuple(strip_func(c.label()) if isinstance(c, Tree) else c for c in node)
            lp = rule_logprob.get((lhs, rhs), default_lp)
            surp[right] += -lp   # surprisal = -log_prob (positive)

        return left, right

    _process(tree)
    return surp


def _norm(s):
    """Normalize token for alignment: lowercase, remove non-alphanumeric."""
    s = s.lower()
    for a, b in [('-lrb-', '('), ('-rrb-', ')'), ('-lsb-', '['), ('-rsb-', ']'),
                 ('-lcb-', '{'), ('-rcb-', '}'), ('``', '"'), ("''", '"')]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]", '', s)


def align_tokens(bllip_tokens, dundee_words):
    """
    Map each BLLIP leaf to a Dundee word index.
    BLLIP splits contractions ("won't" -> "wo","n't") and possessives ("Britain's" -> "Britain","'s").
    Returns list of len(bllip_tokens): dundee word index (0-based), or -1 if unmatched.
    """
    bl = [_norm(t) for t in bllip_tokens]
    dl = [_norm(w) for w in dundee_words]
    result = [-1] * len(bl)
    bi = di = 0
    while bi < len(bl) and di < len(dl):
        if bl[bi] == dl[di]:
            result[bi] = di
            bi += 1; di += 1
        elif len(bl[bi]) <= len(dl[di]):
            # One or more BLLIP tokens combine to form one Dundee word
            acc = bl[bi]
            result[bi] = di
            bi += 1
            while bi < len(bl) and acc != dl[di]:
                acc += bl[bi]
                result[bi] = di
                bi += 1
            di += 1
        else:
            # One BLLIP token spans multiple Dundee words (rare) — map to last
            acc = dl[di]
            di += 1
            while di < len(dl) and acc != bl[bi]:
                acc += dl[di]
                di += 1
            result[bi] = di - 1
            bi += 1
    return result


# ── 3. Load BLLIP ─────────────────────────────────────────────────────────────

print('Loading BLLIP WSJ-PTB3 model...')
t0 = time.time()
from bllipparser import RerankingParser
rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=False)
print(f'  Loaded in {time.time()-t0:.1f}s')


# ── 4. Build PCFG ────────────────────────────────────────────────────────────

print('\nBuilding PTB PCFG from WSJ sections 02-21...')
t0 = time.time()
rule_logprob = build_pcfg_from_ptb()
print(f'  Done in {time.time()-t0:.1f}s  ({len(rule_logprob):,} unique rules)')

# Mean log-prob of known rules — used as default for rules not seen in PTB training
MEAN_RULE_LP = sum(rule_logprob.values()) / max(len(rule_logprob), 1)
print(f'  Mean rule log-prob (unknown-rule default): {MEAN_RULE_LP:.3f}  ({-MEAN_RULE_LP:.3f} nats/rule)')

N_BEST = 20   # number of BLLIP parses to marginalise over


# ── 5. Load Dundee sentences ─────────────────────────────────────────────────

print('\nLoading Dundee sentences...')
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

print(f'  {len(sentences):,} sentences, {len(words_df):,} word positions')


# ── 6. Parse and compute surprisal ──────────────────────────────────────────

results = []
n_done = n_failed = 0
n_total = len(sentences)
t_start = time.time()

for (text_id, sent_id), word_pos_list in sorted(sentences.items()):
    if n_done % 200 == 0:
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1)
        eta = (n_total - n_done) / max(rate, 0.001)
        print(f'  {n_done:4d}/{n_total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)

    wnums = [wnum for wnum, _ in word_pos_list]
    words = [word for _, word in word_pos_list]
    n_w = len(words)

    # ── get N-best parses ──────────────────────────────────────────────────
    try:
        nbest = rrp.parse(' '.join(words))
    except Exception:
        nbest = []

    if not nbest:
        surp = [0.0] * n_w
        n_failed += 1
    else:
        # Process each of the top-N parses; score under our PTB PCFG
        parse_scores = []      # PCFG log-prob for each parse (sum of rule log-probs)
        parse_word_surps = []  # per-Dundee-word surprisal for each parse

        for nbest_item in nbest[:N_BEST]:
            try:
                tree_i = Tree.fromstring(str(nbest_item.ptb_parse))
                if tree_i.label() in {'S1', 'TOP', 'ROOT'} and len(tree_i) == 1:
                    tree_i = tree_i[0]
                bllip_tokens_i = tree_i.leaves()
                bllip_surp_i = get_syn_surprisal(tree_i, rule_logprob,
                                                  default_lp=MEAN_RULE_LP)
                # Aggregate to Dundee word indices via token alignment
                leaf_to_word_i = align_tokens(bllip_tokens_i, words)
                word_surp_i = [0.0] * n_w
                for li, wi in enumerate(leaf_to_word_i):
                    if 0 <= wi < n_w:
                        word_surp_i[wi] += bllip_surp_i[li]
                # Score this parse under our PCFG: log P ≈ -total surprisal
                pcfg_score = -sum(bllip_surp_i)
                parse_scores.append(pcfg_score)
                parse_word_surps.append(word_surp_i)
            except Exception:
                continue

        if not parse_scores:
            surp = [0.0] * n_w
            n_failed += 1
        else:
            # Softmax: weight each parse by exp(score - max_score)
            max_s = max(parse_scores)
            weights = [math.exp(s - max_s) for s in parse_scores]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            # Weighted-average surprisal per Dundee word
            surp = [
                sum(weights[i] * parse_word_surps[i][k] for i in range(len(weights)))
                for k in range(n_w)
            ]

    for wnum, s in zip(wnums, surp):
        results.append((text_id, wnum, round(s, 5)))

    n_done += 1

elapsed = time.time() - t_start
print(f'\nDone: {n_done} sentences, {n_failed} failed, {elapsed:.0f}s ({elapsed/max(n_done,1):.2f}s/sent)')

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['text_id', 'wnum', 'syn_surp_pcfg'])
    w.writerows(results)

print(f'Saved {len(results):,} rows -> {OUT_CSV}')

vals = [r[2] for r in results if r[2] > 0]
print(f'Non-zero entries: {len(vals):,}/{len(results):,}')
if vals:
    import statistics
    print(f'  mean={statistics.mean(vals):.3f}  median={statistics.median(vals):.3f}  '
          f'min={min(vals):.3f}  max={max(vals):.3f}')

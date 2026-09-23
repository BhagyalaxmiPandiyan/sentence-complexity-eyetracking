"""
Build van Schijndel-style left-corner syntactic surprisal for Dundee corpus.

Pipeline:
  1. Binarize PTB WSJ 2-21 trees → linetrees format
  2. Run linetrees2synprocdecpars (training mode) → decision sequences
  3. Build CPT model via relfreq.py
  4. Parse Dundee sentences with Berkeley parser → binary linetrees
  5. Run linetrees2synprocdecpars (test mode, same binary) → Dundee decisions
  6. Compute per-word surprisal from decisions + model
  7. Save to output/lcparse_syn_surprisal.csv
"""

import os, sys, re, math, csv, subprocess, tempfile
from collections import defaultdict
from nltk import Tree

BASE    = '/mnt/e/Project/Final One'
PTB_DIR = BASE + '/Dataset/penn_treebank_3/parsed/mrg/wsj'
OUT_DIR = BASE + '/output'
LEX_CSV = OUT_DIR + '/lexical_features.csv'
OUT_CSV = OUT_DIR + '/lcparse_syn_surprisal.csv'

BINARY     = '/tmp/linetrees2synprocdecpars'
RELFREQ    = '/home/bhagya/modelblocks-release/resource-incrsem/scripts/relfreq.py'
BERK_JAR   = '/home/berkeleyparser/BerkeleyParser-1.7.jar'
BERK_GR    = '/home/berkeleyparser/eng_sm6.gr'
MODEL_FILE = OUT_DIR + '/lcparse_model.cpt'


# ── 1. Label normalization and binarization ───────────────────────────────────

_KEEP_PUNCT = frozenset({'-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-',
                          '-NONE-', '--', 'T'})

def strip_func(label):
    """Strip PTB functional tags: NP-SBJ → NP, S-TPC-1 → S. Keeps punct labels."""
    if label in _KEEP_PUNCT:
        return label
    for sep in ('-', '='):
        idx = label.find(sep)
        if idx > 0:
            return label[:idx]
    return label


def normalize_labels(tree):
    """Recursively strip functional tags from all non-leaf nodes."""
    if not isinstance(tree, Tree):
        return
    tree.set_label(strip_func(tree.label()))
    for child in tree:
        normalize_labels(child)


def binarize(tree):
    """Recursively right-binarize an NLTK Tree in-place."""
    if not isinstance(tree, Tree):
        return
    for child in tree:
        binarize(child)
    while len(tree) > 2:
        last   = tree.pop()
        second = tree.pop()
        new_node = Tree(tree.label(), [second, last])
        tree.append(new_node)


def tree_to_oneline(tree):
    """Convert NLTK Tree to one-line string."""
    return ' '.join(str(tree).split())


# ── 2. Read PTB trees (sections 02-21) → binarize → write linetrees ──────────

def build_ptb_linetrees(out_path):
    if os.path.exists(out_path):
        print(f'  Using cached PTB linetrees: {out_path}')
        return
    print('  Reading PTB WSJ sections 02-21...')
    n = 0
    with open(out_path, 'w') as fout:
        for sec in range(2, 22):
            sec_dir = os.path.join(PTB_DIR, f'{sec:02d}')
            if not os.path.isdir(sec_dir):
                continue
            for fname in sorted(os.listdir(sec_dir)):
                if not fname.endswith('.mrg'):
                    continue
                with open(os.path.join(sec_dir, fname), encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                depth, start = 0, None
                for i, ch in enumerate(content):
                    if ch == '(':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0 and start is not None:
                            ts = content[start:i+1].strip()
                            if ts:
                                try:
                                    t = Tree.fromstring(ts)
                                    normalize_labels(t)
                                    binarize(t)
                                    fout.write(tree_to_oneline(t) + '\n')
                                    n += 1
                                except Exception:
                                    pass
                            start = None
    print(f'  Wrote {n:,} binary trees to {out_path}')


# ── 3. Run binary in training mode → decision sequences ────────────────────────

def run_decpars(linetrees_path, decpars_path):
    if os.path.exists(decpars_path):
        print(f'  Using cached decisions: {decpars_path}')
        return
    print(f'  Running linetrees2synprocdecpars on {linetrees_path}...')
    with open(linetrees_path) as fin, open(decpars_path, 'w') as fout:
        result = subprocess.run(
            [BINARY],
            stdin=fin, stdout=fout, stderr=subprocess.PIPE, text=True
        )
    if result.returncode != 0:
        print(f'  STDERR: {result.stderr[:500]}')
    print(f'  Done → {decpars_path}')


# ── 4. Build CPT model from decision sequences ────────────────────────────────

def build_model(decpars_path, model_path):
    if os.path.exists(model_path):
        print(f'  Using cached model: {model_path}')
        return
    print('  Building CPT model from decisions...')
    counts = defaultdict(lambda: defaultdict(float))
    with open(decpars_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('TREE:'):
                continue
            # Format: "TYPE condition : response"
            # But type is first token; condition is rest before " : "
            parts = line.split(' : ', 1)
            if len(parts) != 2:
                continue
            predictor = parts[0].strip()
            response  = parts[1].strip()
            if predictor and response:
                counts[predictor][response] += 1.0

    with open(model_path, 'w') as fout:
        for predictor in sorted(counts):
            total = sum(counts[predictor].values())
            for response in sorted(counts[predictor]):
                prob = counts[predictor][response] / total
                fout.write(f'{predictor} : {response} = {prob}\n')

    n_entries = sum(len(v) for v in counts.values())
    print(f'  Model: {len(counts):,} predictors, {n_entries:,} entries → {model_path}')


# ── 5. Load model into memory ─────────────────────────────────────────────────

def load_model(model_path):
    model = {}
    with open(model_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "predictor : response = prob"
            m = re.match(r'^(.*) : (\S+) = (.+)$', line)
            if m:
                predictor = m.group(1).strip()
                response  = m.group(2).strip()
                prob      = float(m.group(3))
                if predictor not in model:
                    model[predictor] = {}
                model[predictor][response] = prob
    print(f'  Loaded model: {len(model):,} predictors')
    return model


# ── 6. Parse Dundee sentences with Berkeley parser ────────────────────────────

import pandas as pd

def parse_dundee_sentences(linetrees_path):
    if os.path.exists(linetrees_path):
        print(f'  Using cached Dundee linetrees: {linetrees_path}')
        return

    print('  Loading Dundee sentences from lexical_features.csv...')
    lex = pd.read_csv(LEX_CSV)
    words_df = (lex[['text_id', 'wnum', 'sent_id', 'word']]
                .drop_duplicates(subset=['text_id', 'wnum'])
                .sort_values(['text_id', 'wnum']))

    sentences = defaultdict(list)
    for _, row in words_df.iterrows():
        key = (int(row['text_id']), int(row['sent_id']))
        sentences[key].append((int(row['wnum']), str(row['word'])))
    for key in sentences:
        sentences[key].sort(key=lambda x: x[0])

    # Filter out sent_id <= 0 (unknown sentence boundaries)
    valid = {k: v for k, v in sentences.items() if k[1] > 0}
    print(f'  {len(valid):,} valid sentences to parse')

    # Write sentence texts for Berkeley parser
    sent_keys = sorted(valid.keys())
    sent_file = OUT_DIR + '/dundee_sentences_for_berk.txt'
    with open(sent_file, 'w') as f:
        for key in sent_keys:
            words = [w for _, w in valid[key]]
            f.write(' '.join(words) + '\n')

    print(f'  Running Berkeley parser ({len(sent_keys):,} sentences)...')
    with open(sent_file) as fin, open(linetrees_path + '.raw', 'w') as fout:
        result = subprocess.run(
            ['java', '-jar', BERK_JAR, '-gr', BERK_GR, '-maxLength', '200'],
            stdin=fin, stdout=fout, stderr=subprocess.PIPE, text=True
        )
    if result.returncode != 0:
        print(f'  Berkeley STDERR: {result.stderr[:300]}')

    # Read Berkeley output, binarize, write linetrees with metadata
    print('  Binarizing Berkeley parse output...')
    with open(linetrees_path + '.raw') as fin, open(linetrees_path, 'w') as fout:
        i = 0
        for line in fin:
            line = line.strip()
            if not line or line == '()':
                # Failed parse — write a dummy single-word tree
                if i < len(sent_keys):
                    key = sent_keys[i]
                    words = [w for _, w in valid[key]]
                    dummy = '(S ' + ' '.join(f'(X {w})' for w in words) + ')'
                    try:
                        t = Tree.fromstring(dummy)
                        binarize(t)
                        fout.write(tree_to_oneline(t) + '\t' + f'{key[0]},{key[1]}' + '\n')
                    except Exception:
                        fout.write(f'(S (X ?))\t{key[0]},{key[1]}\n')
                    i += 1
                continue
            if i < len(sent_keys):
                key = sent_keys[i]
                try:
                    t = Tree.fromstring(line)
                    normalize_labels(t)
                    binarize(t)
                    fout.write(tree_to_oneline(t) + '\t' + f'{key[0]},{key[1]}' + '\n')
                except Exception:
                    fout.write(f'(S (X ?))\t{key[0]},{key[1]}\n')
                i += 1

    print(f'  Wrote {i:,} binarized Dundee trees → {linetrees_path}')
    return sent_keys, valid


MAX_SENT_LEN = 50   # skip sentences longer than this (avoid binary hang)

# ── 7. Compute word-level surprisal from decisions + model ───────────────────

DEFAULT_PROB = 1e-6   # for unseen condition/response pairs

def _parse_decisions(output, wnums, model):
    """Parse one block of decision output into per-word surprisal."""
    results_word = {}
    lines = output.split('\n')
    current_surp = 0.0
    word_idx = 0
    i = 0
    while i < len(lines):
        dline = lines[i].strip()
        if not dline or dline.startswith('TREE:'):
            i += 1; continue
        parts = dline.split(' : ', 1)
        if len(parts) != 2:
            i += 1; continue
        dtype     = parts[0].split()[0]
        predictor = parts[0].strip()
        response  = parts[1].strip()

        if dtype in ('F', 'P'):
            prob = model.get(predictor, {}).get(response, DEFAULT_PROB)
            current_surp += -math.log(max(prob, DEFAULT_PROB))
            i += 1
        elif dtype == 'W':
            i += 1
            while i < len(lines):
                nline = lines[i].strip()
                ntype = nline.split()[0] if nline.split() else ''
                if ntype in ('J', 'A', 'B'):
                    nparts = nline.split(' : ', 1)
                    if len(nparts) == 2:
                        prob = model.get(nparts[0].strip(), {}).get(
                            nparts[1].strip(), DEFAULT_PROB)
                        current_surp += -math.log(max(prob, DEFAULT_PROB))
                    i += 1
                else:
                    break
            if word_idx < len(wnums):
                results_word[wnums[word_idx]] = round(current_surp, 5)
            word_idx += 1
            current_surp = 0.0
        else:
            i += 1
    return results_word


def compute_surprisal_from_decisions(dundee_linetrees, model, sent_keys, valid):
    """
    Run linetrees2synprocdecpars ONCE PER SENTENCE so the binary state resets
    cleanly between sentences (batch mode accumulates context across trees).
    Skips sentences longer than MAX_SENT_LEN.
    """
    sent_wnums = {}
    for key in sent_keys:
        sent_wnums[key] = [wn for wn, _ in valid[key]]

    trees_with_keys = []
    with open(dundee_linetrees) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                tree_str, meta = line.rsplit('\t', 1)
                tid, sid = meta.split(',')
                key = (int(tid), int(sid))
                n_words = len(sent_wnums.get(key, []))
                if 0 < n_words <= MAX_SENT_LEN:
                    trees_with_keys.append((tree_str.strip(), key))

    n_total = len(trees_with_keys)
    print(f'  {n_total:,} sentences within length limit ({MAX_SENT_LEN} words)')
    print('  Running binary per-sentence (resets state between sentences)...')

    results = {}
    n_ok = n_fail = n_timeout = 0
    SENT_TIMEOUT = 5  # seconds per sentence

    for i, (tree_str, key) in enumerate(trees_with_keys):
        if i % 500 == 0:
            print(f'    {i}/{n_total}  ok={n_ok}  fail={n_fail}  timeout={n_timeout}',
                  flush=True)
        if key not in sent_wnums:
            continue
        wnums = sent_wnums[key]
        try:
            proc = subprocess.run(
                [BINARY],
                input=tree_str + '\n',
                capture_output=True, text=True,
                timeout=SENT_TIMEOUT
            )
            word_surps = _parse_decisions(proc.stdout, wnums, model)
            results.update({(key[0], wn): s for wn, s in word_surps.items()})
            n_ok += 1
        except subprocess.TimeoutExpired:
            n_timeout += 1
        except Exception:
            n_fail += 1

    print(f'  Done: {n_ok} ok, {n_fail} failed, {n_timeout} timeouts')
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    PTB_LINETREES  = OUT_DIR + '/ptb_wsj_binary.linetrees'
    PTB_DECISIONS  = OUT_DIR + '/ptb_wsj_decisions.txt'
    DUND_LINETREES = OUT_DIR + '/dundee_binary.linetrees'

    print('='*60)
    print('STEP 1: Build PTB binary linetrees')
    print('='*60)
    build_ptb_linetrees(PTB_LINETREES)

    print('\n' + '='*60)
    print('STEP 2: Extract decision sequences from PTB (training)')
    print('='*60)
    run_decpars(PTB_LINETREES, PTB_DECISIONS)

    print('\n' + '='*60)
    print('STEP 3: Build CPT model from PTB decisions')
    print('='*60)
    build_model(PTB_DECISIONS, MODEL_FILE)
    model = load_model(MODEL_FILE)

    print('\n' + '='*60)
    print('STEP 4: Parse Dundee sentences with Berkeley parser')
    print('='*60)
    result = parse_dundee_sentences(DUND_LINETREES)
    if result is None:
        lex = pd.read_csv(LEX_CSV)
        words_df = (lex[['text_id', 'wnum', 'sent_id', 'word']]
                    .drop_duplicates(subset=['text_id', 'wnum'])
                    .sort_values(['text_id', 'wnum']))
        sentences = defaultdict(list)
        for _, row in words_df.iterrows():
            key = (int(row['text_id']), int(row['sent_id']))
            sentences[key].append((int(row['wnum']), str(row['word'])))
        for key in sentences:
            sentences[key].sort(key=lambda x: x[0])
        valid = {k: v for k, v in sentences.items() if k[1] > 0}
        sent_keys = sorted(valid.keys())
    else:
        sent_keys, valid = result

    print('\n' + '='*60)
    print('STEP 5: Compute left-corner surprisal for Dundee')
    print('='*60)
    surprisal_map = compute_surprisal_from_decisions(DUND_LINETREES, model, sent_keys, valid)

    print(f'\n  Computed surprisal for {len(surprisal_map):,} word positions')
    vals = [v for v in surprisal_map.values() if v > 0]
    if vals:
        import statistics
        print(f'  Non-zero: {len(vals):,}  mean={statistics.mean(vals):.3f}  '
              f'median={statistics.median(vals):.3f}  max={max(vals):.3f}')

    # Merge with all word positions (include 0 for words without valid parses)
    lex = pd.read_csv(LEX_CSV)
    all_positions = lex[['text_id', 'wnum']].drop_duplicates()
    rows = []
    for _, row in all_positions.iterrows():
        k = (int(row['text_id']), int(row['wnum']))
        rows.append((k[0], k[1], surprisal_map.get(k, 0.0)))

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text_id', 'wnum', 'syn_surp_pcfg'])
        w.writerows(rows)

    print(f'\nSaved {len(rows):,} rows → {OUT_CSV}')

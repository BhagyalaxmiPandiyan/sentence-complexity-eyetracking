"""Speed test: POS-only PCFG on sample Dundee sentences."""
import sys, time, os
sys.path.insert(0, '/mnt/e/Project/Final One/src')
os.chdir('/mnt/e/Project/Final One')

# Inline the key pieces for a quick test
import tarfile, pickle, math
from collections import defaultdict
from nltk import Tree, Nonterminal, induce_pcfg
from nltk.parse import ViterbiParser

PTB_POS = frozenset([
    "CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD",
    "NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR",
    "RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ",
    "WDT","WP","WP$","WRB","#","$","''","``",",",".",":","-LRB-","-RRB-",
])

GRAMMAR_CACHE = '/mnt/e/Project/Final One/output/pos_only_pcfg.pkl'
PTB_ARCHIVE   = '/mnt/e/Project/Final One/Dataset/penn_treebank_3.tar.bz2'

def pos_only_productions(tree):
    try:
        t = tree.copy(deep=True)
        for idx in t.treepositions('leaves'):
            parent_idx = idx[:-1]
            parent = t[parent_idx]
            if isinstance(parent, Tree) and len(parent) == 1:
                pos_label = parent.label()
                if pos_label in PTB_POS:
                    parent[0] = pos_label
        return t.productions()
    except Exception:
        return []

if os.path.exists(GRAMMAR_CACHE):
    print('Loading grammar from cache...')
    with open(GRAMMAR_CACHE, 'rb') as f:
        grammar = pickle.load(f)
    print(f'Rules: {len(grammar.productions()):,}')
else:
    print('Building POS-only PCFG from WSJ sections 02-05...')
    all_productions = []
    n_trees = 0
    t0 = time.time()
    with tarfile.open(PTB_ARCHIVE, 'r:bz2') as tf:
        wsj = [m for m in tf.getmembers()
               if any(f'wsj/0{s}' in m.name for s in [2,3,4,5])
               and m.name.endswith('.mrg')]
        print(f'Files: {len(wsj)}')
        for m in wsj:
            fobj = tf.extractfile(m)
            if fobj is None: continue
            content = fobj.read().decode('utf-8', errors='ignore')
            depth, start = 0, 0
            for i, ch in enumerate(content):
                if ch == '(':
                    if depth == 0: start = i
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        try:
                            t = Tree.fromstring(content[start:i+1])
                            all_productions.extend(pos_only_productions(t))
                            n_trees += 1
                        except Exception:
                            pass
    print(f'Trees: {n_trees:,}  Productions: {len(all_productions):,}  '
          f'Time: {time.time()-t0:.0f}s')
    grammar = induce_pcfg(Nonterminal('S'), all_productions)
    print(f'Grammar rules: {len(grammar.productions()):,}')

parser = ViterbiParser(grammar, trace=0)

# Test on realistic Dundee-style POS sequences (short to long)
test_seqs = [
    ['DT', 'NN', 'VBD', 'DT', 'NN', '.'],           # 6 words
    ['DT', 'NNS', 'IN', 'DT', 'JJ', 'NN', 'VBD', 'RB', '.'],  # 9 words
    ['IN', 'DT', 'JJ', 'NN', ',', 'NNS', 'VBD', 'RBR', 'IN', 'DT', 'NN', 'NN', '.'],  # 13
    ['DT', 'NN', 'IN', 'NNP', 'VBD', 'DT', 'NN', 'IN', 'PRP$', 'NN', 'TO', 'VB', 'DT', 'NN', '.'],  # 15
]

print('\nParsing speed test:')
for seq in test_seqs:
    t1 = time.time()
    try:
        parses = list(parser.parse(seq))
        elapsed = time.time() - t1
        prob = parses[0].prob() if parses else 0.0
        print(f'  len={len(seq):2d}  parses={len(parses):3d}  '
              f'prob={prob:.2e}  time={elapsed:.2f}s')
    except Exception as e:
        print(f'  len={len(seq):2d}  FAILED: {e}  time={time.time()-t1:.2f}s')

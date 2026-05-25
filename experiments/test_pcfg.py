"""Quick benchmark: NLTK PCFG build + parse speed via WSL."""
import time, sys, tarfile, re
from nltk import Tree, Nonterminal, induce_pcfg
from nltk.parse import ViterbiParser

ptb_path = '/mnt/e/Project/Final One/Dataset/penn_treebank_3.tar.bz2'
productions = []
n = 0
t0 = time.time()

with tarfile.open(ptb_path, 'r:bz2') as tf:
    # Use only WSJ section 02 (small sample) for timing test
    wsj = [m for m in tf.getmembers()
           if 'parsed/mrg/wsj/02' in m.name and m.name.endswith('.mrg')]
    print(f'Section 02 files: {len(wsj)}')
    for m in wsj[:5]:
        fobj = tf.extractfile(m)
        if fobj is None:
            continue
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
                        productions.extend(t.productions())
                        n += 1
                    except Exception:
                        pass

print(f'Trees: {n:,}  Productions: {len(productions):,}  Time: {time.time()-t0:.1f}s')

S = Nonterminal('S')
grammar = induce_pcfg(S, productions)
print(f'Grammar rules: {len(grammar.productions()):,}')

parser = ViterbiParser(grammar)

# Time parsing a short sentence
test_sents = [
    'The dog saw the cat .'.split(),
    'Yields on mortgage-backed securities fell .'.split(),
    'The economy grew last quarter .'.split(),
]

for sent in test_sents:
    t1 = time.time()
    try:
        parses = list(parser.parse(sent))
        elapsed = time.time() - t1
        prob = parses[0].prob() if parses else 0.0
        print(f'  {" ".join(sent[:5])}...  parses={len(parses)}  prob={prob:.2e}  time={elapsed:.2f}s')
    except Exception as e:
        print(f'  FAILED: {e}')

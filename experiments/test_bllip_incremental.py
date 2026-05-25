"""Test BLLIP incremental surprisal on 10 sample Dundee sentences."""
import os, sys, time
from collections import defaultdict

BASE    = '/mnt/e/Project/Final One'
DATASET = BASE + '/Dataset'

from bllipparser import RerankingParser
print('Loading BLLIP model...')
t0 = time.time()
rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=False)
print(f'Loaded in {time.time()-t0:.1f}s')

def bllip_score(words):
    try:
        nbest = rrp.parse(' '.join(words))
        return nbest[0].parser_score if nbest else -1000.0
    except Exception:
        return -1000.0

def incremental_surp(words):
    prev = 0.0
    surps = []
    for k in range(1, len(words)+1):
        s = bllip_score(words[:k])
        surps.append(max(-(s - prev), 0.0))
        prev = s
    return surps

# Load first 10 Dundee sentences
info_dir = DATASET + '/dundee_corpus/per-word-info'
sentences = defaultdict(list)
path = os.path.join(info_dir, 'tx01wrdp.dat')
with open(path, encoding='latin-1') as f:
    for line in f:
        p = line.strip().split()
        if len(p) < 8: continue
        try:
            word = p[0]; wnum = int(p[6]); sid = int(p[7])
            if sid > 0: sentences[sid].append((wnum, word))
        except: pass

t1 = time.time()
for sid in sorted(sentences.keys())[:10]:
    wlist = [w for _, w in sorted(sentences[sid])]
    t2 = time.time()
    surps = incremental_surp(wlist)
    elapsed = time.time() - t2
    total_surp = sum(surps)
    print(f'  sent {sid:3d}  words={len(wlist):3d}  '
          f'total_surp={total_surp:7.2f}  time={elapsed:.2f}s')
    print(f'    words: {" ".join(wlist[:8])}...')
    print(f'    surps: {[round(s,1) for s in surps[:8]]}...')
    print()

print(f'10 sentences in {time.time()-t1:.1f}s')

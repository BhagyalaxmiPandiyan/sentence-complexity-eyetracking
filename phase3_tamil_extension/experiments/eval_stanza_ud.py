"""
Accuracy of Stanza's Tamil morphology on the UD Tamil-TTB TEST split
(hand-annotated gold; Stanza was trained on the train split, so this is fair).
Run: python experiments/eval_stanza_ud.py
"""
import os, sys, stanza
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sents, cur = [], []
for line in open(os.path.join(BASE, "tools", "UD_Tamil-TTB", "ta_ttb-ud-test.conllu"), encoding="utf-8"):
    if not line.strip():
        if cur: sents.append(cur); cur = []
    elif line[0].isdigit():
        c = line.rstrip("\n").split("\t")
        if "-" not in c[0] and "." not in c[0]:
            cur.append((c[1], c[2], c[3], c[5]))
nlp = stanza.Pipeline("ta", processors="tokenize,pos,lemma", tokenize_pretokenized=True, verbose=False)
doc = nlp([[w for w, *_ in s] for s in sents])
n = lem = pos = fe = case = ncase = 0
for s, ps in zip(sents, doc.sentences):
    for (w, gl, gp, gf), p in zip(s, ps.words):
        n += 1; lem += p.lemma == gl; pos += p.upos == gp; fe += (p.feats or "_") == gf
        g = dict(x.split("=") for x in gf.split("|")) if gf != "_" else {}
        if "Case" in g:
            ncase += 1; case += f"Case={g['Case']}" in (p.feats or "")
print(f"{len(sents)} test sentences, {n} words")
print(f"lemma {lem/n:.1%} | POS {pos/n:.1%} | all features exact {fe/n:.1%} | case {case/ncase:.1%}")

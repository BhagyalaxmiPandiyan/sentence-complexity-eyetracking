"""
Step 12 (external test): do the Tamil difficulty models recognise human simplifications?

Data: OasisSimp (Liu et al., LREC 2026), Tamil part: 520 complex sentences, each with
2-5 simplifications written by trained annotators (valid 104 + test 416 items).
Download https://oasissimpdataset.github.io/OasisSimp.zip into
Dataset/oasissimp/ and unzip (only OasisSimp/OasisSimp-TA/ is needed). The data are
not redistributed in this repository (no license file in the release).

This is the Tamil counterpart of the English System 2 test data (Wikipedia vs Simple
Wikipedia, human-aligned): every pair is (complex, simplified), so the question is the
same: which sentence is harder? Nothing here is trained on OasisSimp. All models are the
step-6 pair models (logistic regression on feature differences, no intercept) fit on the
StoryWeaver TRAIN pairs, exactly as metric_model.py does for 'word' (A) and 'morph' (A+B).

Pairs: one per (complex, simplification); simplifications identical to the complex
sentence are dropped (annotator left it unchanged). A text with several sentences gets
the mean of its sentence features (same rule as step 8). Accuracy = share of pairs where
the model scores the complex sentence as harder. 95% CIs and paired differences use
2,000 bootstrap draws over complex ITEMS (all pairs of an item move together).

Run: C:\\Python313\\python.exe src/step12_oasissimp.py      (~30 min first time, features cached)
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step3_textbook_features as s3  # noqa: E402
from metric_model import FEATS as SW_FEATS, PAIRS as SW_PAIRS  # noqa: E402

BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "Dataset", "oasissimp", "OasisSimp", "OasisSimp-TA")
CACHE = os.path.join(BASE, "output", "oasissimp_features.csv")
REPORT = os.path.join(BASE, "results", "oasissimp_summary.md")

A = ["n_words", "word_len", "word_zipf", "word_surprisal"]
B = ["morphs_per_word", "stem_surprisal", "suffix_surprisal", "feats_per_word", "lemma_zipf", "nonfinite_verbs"]
C = ["tree_depth", "dep_distance"]
MODELS = {"A: word-level (word metric)": A, "B: morphology": B, "C: syntax": C,
          "A + B (morph metric)": A + B, "A + C": A + C, "A + B + C": A + B + C}
SINGLE = {"n_words": +1, "word_len": +1, "morphs_per_word": +1, "nonfinite_verbs": +1, "tree_depth": +1}
SENT_END = re.compile(r"(?<=[.?!])\s+")
TAMIL = re.compile("[\u0B80-\u0BFF]")
N_BOOT, SEED = 2000, 42


def load_jsonl(path):
    """The release files hold pretty-printed JSON objects, not one object per line."""
    text, dec, i, out = open(path, encoding="utf-8").read(), json.JSONDecoder(), 0, []
    while True:
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text):
            return out
        obj, i = dec.raw_decode(text, i)
        out.append(obj)


DROPPED = 0


def load_pairs():
    global DROPPED
    rows, total = [], 0
    for split in ["valid", "test"]:
        for r in load_jsonl(os.path.join(DATA, f"OasisSimp-TA_{split}.jsonl")):
            total += len(r["simple"])
            for k, s in enumerate(dict.fromkeys(x.strip() for x in r["simple"])):   # unique refs
                if s and s != r["complex"].strip():
                    rows.append({"item": r["id"], "split": split, "ref": k,
                                 "complex": r["complex"].strip(), "simple": s})
    DROPPED = total - len(rows)
    return pd.DataFrame(rows)


def features(texts):
    cols = A + B + C
    done = pd.read_csv(CACHE).set_index("text") if os.path.exists(CACHE) else pd.DataFrame(columns=cols)
    todo = [t for t in texts if t not in done.index]
    if todo:
        print(f"computing features for {len(todo)} texts ...", flush=True)
        tools = s3.load_tools()
        new = {}
        for n, t in enumerate(todo, 1):
            parts = [p for p in SENT_END.split(t) if TAMIL.search(p)] or [t]
            new[t] = pd.DataFrame([s3.sentence_features(p, *tools) for p in parts]).mean()
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
                pd.concat([done, pd.DataFrame(new).T]).rename_axis("text").to_csv(CACHE, encoding="utf-8")
        done = pd.concat([done, pd.DataFrame(new).T]).rename_axis("text")
        done.to_csv(CACHE, encoding="utf-8")
    return done


def fit_models():
    f = pd.read_csv(SW_FEATS).drop_duplicates("sentence").set_index("sentence")
    p = pd.read_csv(SW_PAIRS)
    p = p[p.split == "train"]
    y = (p.harder == "b").astype(int).to_numpy()
    out = {}
    for name, cols in MODELS.items():
        D = f.loc[p.sent_b, cols].to_numpy() - f.loc[p.sent_a, cols].to_numpy()
        scale = D.std(0) + 1e-9
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000).fit(D / scale, y)
        out[name] = (cols, clf.coef_[0] / scale)
    return out


def main():
    pairs = load_pairs()
    texts = list(dict.fromkeys(pairs.complex.tolist() + pairs.simple.tolist()))
    print(f"{pairs.item.nunique()} complex items, {len(pairs)} pairs, {len(texts)} unique texts")
    F = features(texts)
    Fc, Fs = F.loc[pairs.complex], F.loc[pairs.simple]
    pairs["d_words"] = Fc.n_words.to_numpy() - Fs.n_words.to_numpy()
    lm = (pairs.d_words.abs() <= 1).to_numpy()

    correct = {}
    for name, (cols, w) in fit_models().items():
        correct[name] = (Fc[cols].to_numpy() @ w > Fs[cols].to_numpy() @ w).astype(float)
    for col, sign in SINGLE.items():
        correct[f"single: {col}"] = (sign * (Fc[col].to_numpy() - Fs[col].to_numpy()) > 0).astype(float)

    items = pairs.item.to_numpy()
    uniq = np.unique(items)
    idx = {u: np.flatnonzero(items == u) for u in uniq}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]

    def stats(c, mask=None):
        m = np.ones(len(c), bool) if mask is None else mask
        boot = [c[d][m[d]].mean() for d in draws if m[d].any()]
        return c[m].mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def diff(c1, c2, mask=None):
        m = np.ones(len(c1), bool) if mask is None else mask
        dd = np.array([c1[d][m[d]].mean() - c2[d][m[d]].mean() for d in draws if m[d].any()])
        return c1[m].mean() - c2[m].mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5), float(np.mean(dd <= 0))

    pct = lambda x: f"{100 * x:.1f}%"
    L = ["# External test: OasisSimp Tamil (human simplifications)", "",
         f"Data: OasisSimp Tamil valid + test (Liu et al., LREC 2026): {pairs.item.nunique()} complex sentences, "
         f"{len(pairs)} (complex, simplification) pairs, after dropping {DROPPED} simplifications identical to "
         "the complex sentence (or repeated within an item). Models trained on StoryWeaver train pairs only "
         "(never on OasisSimp). Accuracy = complex sentence scored harder; chance = 50%. "
         f"95% CI: {N_BOOT} bootstrap draws over complex items. Length-matched = word counts differ by ≤ 1 "
         f"({lm.sum()} pairs).", "",
         f"Mean words: complex {Fc.n_words.mean():.1f}, simplified {Fs.n_words.mean():.1f}; "
         f"simplification shorter in {np.mean(pairs.d_words > 0):.0%} of pairs, same length in "
         f"{np.mean(pairs.d_words == 0):.0%}, longer in {np.mean(pairs.d_words < 0):.0%}.", "",
         "## Pair accuracy", "", "| Model | All pairs | 95% CI | Length-matched | 95% CI |", "|---|---|---|---|---|"]
    for name, c in correct.items():
        a, lo, hi = stats(c)
        b, blo, bhi = stats(c, lm)
        L.append(f"| {name} | {pct(a)} | [{pct(lo)}, {pct(hi)}] | {pct(b)} | [{pct(blo)}, {pct(bhi)}] |")
    L += ["", "## Gain over the word-level model (paired bootstrap)", "",
          "| Comparison | Subset | Gain | 95% CI | p (one-sided, no gain) |", "|---|---|---|---|---|"]
    base = correct["A: word-level (word metric)"]
    for name in ["A + B (morph metric)", "A + C", "A + B + C", "B: morphology"]:
        for sub, m in [("all", None), ("length-matched", lm)]:
            g, lo, hi, p = diff(correct[name], base, m)
            L.append(f"| {name} vs A | {sub} | {100 * g:+.1f} pts | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {p:.3f} |")
    L += ["", "Mean feature values:", "", "| Feature | Complex | Simplified |", "|---|---|---|"]
    for col in A + B + C:
        L.append(f"| {col} | {Fc[col].mean():.2f} | {Fs[col].mean():.2f} |")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

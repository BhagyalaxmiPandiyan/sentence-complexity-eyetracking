"""
Phase 1, all features live: the 19 System 1 features of the thesis for any English sentence,
computed with the same resources and formulas as the pipeline, then all four reading times.

  wlen, sent_len, wiki_freq, aoa_mean, aoa_std   step 2 (letters, wordfreq, Kuperman AoA)
  fwd_prob, bwd_prob                             step 2 (BNC bigram counts, add-1 smoothing)
  lex_surprisal                                  step 3 (-log P(word | previous word), BNC)
  syn_surprisal                                  van Schijndel left-corner parser, run live in
                                                 WSL through lc_server.py (Berkeley parse ->
                                                 binarize -> linetrees2synprocdecpars -> CPT)
  entropy_red                                    step 3 (PTB POS-bigram entropy drop; POS tags
                                                 from the Berkeley parse)
  emb_depth, emb_diff, h1-h8                     step 3 formulas on a dependency parse (spaCy
                                                 en_core_web_sm, one token per typed word)

Models: models/system1_full.pkl = phase1_replication/output/system1_models.pkl (final run).
Needs data/bnc.sqlite (build_bnc_db.py). If WSL or the parser is unavailable, syn_surprisal
falls back to PTB POS-bigram surprisal and the page says so.
"""
import json
import math
import os
import pickle
import queue
import re
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
from wordfreq import word_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1_replication", "src"))
from syntactic_extractor import _pos_entropy, _pos_surp  # noqa: E402  (same helpers as step 3)

MEASURES = ["ffd", "fpd", "rpd", "td"]
STRIP = ".,!?;:'\"()"
LC_CMD = ["wsl", "-d", "Ubuntu", "-e", "python3", "/mnt/e/Project/Final One/toolkit/lc_server.py"]

with open(os.path.join(HERE, "models", "system1_full.pkl"), "rb") as f:
    MODELS = pickle.load(f)
with open(os.path.join(HERE, "data", "ptb_pos_lm.pkl"), "rb") as f:
    POS_UNI, POS_BI = pickle.load(f)

AOA = {}   # exactly step 2's load_aoa(): AoA_Kup, else AoA_Kup_lem; spread over 4 norms
_aoa = pd.read_csv(os.path.join(HERE, "data", "AoA.csv"), encoding="latin-1")
for _, r in _aoa.iterrows():
    v = r.get("AoA_Kup", float("nan"))
    if pd.isna(v):
        v = r.get("AoA_Kup_lem", float("nan"))
    vals = [r.get(c, float("nan")) for c in ["AoA_Kup", "AoA_Bird_lem", "AoA_Bristol_lem", "AoA_Cort_lem"]]
    vals = [x for x in vals if not pd.isna(x)]
    if not pd.isna(v):
        AOA[str(r["Word"]).lower().strip()] = (float(v), float(np.std(vals)) if len(vals) > 1 else 0.0)
AOA_DEFAULT = (float(np.mean([a for a, _ in AOA.values()])), float(np.mean([s for _, s in AOA.values()])))

_state = {"lc": None, "lc_status": "starting", "spacy": None, "db": None}
_lock = threading.Lock()


# ── background start-up: spaCy, BNC database, the WSL parser server ─────────
def _start():
    try:
        import spacy
        _state["spacy"] = spacy.load("en_core_web_sm")
    except Exception as e:
        _state["spacy_err"] = str(e)
    db = os.path.join(HERE, "data", "bnc.sqlite")
    if os.path.exists(db):
        con = sqlite3.connect(db, check_same_thread=False)
        _state["db"] = con
        _state["V"] = con.execute("SELECT COUNT(*) FROM unigram").fetchone()[0]
    try:
        p = subprocess.Popen(LC_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8", bufsize=1)
        q = queue.Queue()
        threading.Thread(target=lambda: [q.put(line) for line in p.stdout], daemon=True).start()
        while True:                                   # skip log lines until the server is ready
            line = q.get(timeout=90)
            if line.startswith("{") and json.loads(line).get("ready"):
                break
        _state["lc"], _state["lc_q"], _state["lc_status"] = p, q, "ready"
    except Exception as e:
        _state["lc_status"] = f"unavailable ({type(e).__name__})"


threading.Thread(target=_start, daemon=True).start()


def _lc_surprisal(words):
    """Left-corner surprisal per word from the live parser, or None."""
    if _state["lc"] is None:
        return None, None
    with _lock:
        try:
            _state["lc"].stdin.write(" ".join(words) + "\n")
            _state["lc"].stdin.flush()
            out = json.loads(_state["lc_q"].get(timeout=40))
        except Exception as e:
            _state["lc_status"] = f"error ({type(e).__name__})"
            return None, None
    if "surprisal" not in out:
        return None, None
    return out["surprisal"], out["tree"]


# ── BNC look-ups (step 2 and step 3 formulas) ────────────────────────────────
def _uni(w):
    r = _state["db"].execute("SELECT c FROM unigram WHERE w = ?", (w,)).fetchone()
    return r[0] if r else 0


def _bi(w1, w2):
    r = _state["db"].execute("SELECT c FROM bigram WHERE w1 = ? AND w2 = ?", (w1, w2)).fetchone()
    return r[0] if r else 0


def _norm(w):
    return str(w).lower().strip(STRIP)


# ── the 19 features ──────────────────────────────────────────────────────────
def features(sentence):
    """Returns (DataFrame one row per word with all features, info dict)."""
    words = sentence.strip().split()
    n = len(words)
    info = {}

    # Syntactic surprisal + PTB POS tags from the live left-corner / Berkeley parse
    syn, tree = _lc_surprisal(words)
    info["tree"], info["parser"] = tree, ("left-corner parser (live)" if syn is not None else None)
    tags = re.findall(r"\(([^()\s]+) [^()\s]+\)", tree) if tree else []

    # Dependency depths, one token per typed word (like the Dundee annotation)
    nlp = _state["spacy"]
    depths, heads = [0] * n, [""] * n
    if nlp is not None:
        from spacy.tokens import Doc
        doc = Doc(nlp.vocab, words=words)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        depths = [len(list(t.ancestors)) for t in doc]
        heads = ["(root)" if t.head.i == t.i else t.head.text for t in doc]
        if len(tags) != n:
            tags = [t.tag_ for t in doc]
    if len(tags) != n:
        tags = ["NN"] * n
    if syn is None:                                    # fallback: PTB POS-bigram surprisal
        syn = [_pos_surp(tags[i], tags[i - 1] if i else None, POS_UNI, POS_BI) for i in range(n)]
        info["parser"] = "POS-bigram fallback (WSL parser " + _state["lc_status"] + ")"

    V = _state.get("V") or 1
    have_db = _state["db"] is not None
    rows, prev_H, prev_depth = [], None, None
    for i, word in enumerate(words):
        w = _norm(word)
        aoa_m, aoa_s = AOA.get(w, AOA_DEFAULT)
        # step 2: forward / backward transition probability
        if have_db and i > 0:
            prev = _norm(words[i - 1])
            fwd = math.log((_bi(prev, w) + 1) / (_uni(prev) + V))
            lex = -math.log((_bi(prev, w) + 1) / (_uni(prev) + V))          # step 3, same counts
        else:   # no previous word: unseen-pair value (in Dundee the previous sentence's last word)
            fwd, lex = math.log(1.0 / V), -math.log(1.0 / V)
        if have_db and i < n - 1:
            nxt = _norm(words[i + 1])
            bwd = math.log((_bi(nxt, w) + 1) / (_uni(nxt) + V))             # bwd[w][nxt] = count(nxt, w)
        else:
            bwd = math.log(1.0 / V)
        # step 3: entropy reduction from PTB POS bigrams
        H = _pos_entropy(tags[i], POS_BI, POS_UNI)
        ent = max(0.0, prev_H - H) if prev_H is not None else 0.0
        # step 3: depth features
        d = depths[i]
        diff = d - prev_depth if prev_depth is not None else 0
        if prev_depth is None:
            h = [0] * 8
        else:
            h = [int(d != prev_depth), int(d > prev_depth), int(d < prev_depth), int(d == 0), 0,
                 int(d - prev_depth > 1), int(prev_depth - d > 1), int(d == prev_depth)]
        rows.append({
            "word": word, "pos": tags[i], "head": heads[i],
            "wlen": len(re.sub(r"[^a-zA-Z]", "", word)), "sent_len": n,
            "wiki_freq": math.log(word_frequency(w, "en") + 1e-9),
            "aoa_mean": aoa_m, "aoa_std": aoa_s,
            "fwd_prob": fwd, "bwd_prob": bwd, "lex_surprisal": lex,
            "syn_surprisal": float(syn[i]), "entropy_red": ent,
            "emb_depth": d, "emb_diff": diff,
            **{f"h{k + 1}": h[k] for k in range(8)},
        })
        prev_H, prev_depth = H, d
    info["bnc"] = have_db
    return pd.DataFrame(rows), info


def predict(df):
    out = {}
    for m in MEASURES:
        info = MODELS[m]
        X = info["scaler"].transform(df[info["features"]].fillna(0))
        p = info["model"].predict(X)
        p[p < info["threshold"]] = 0.0
        out[m] = p
    return out


def status():
    return {"parser": _state["lc_status"], "bnc": _state["db"] is not None, "spacy": _state["spacy"] is not None}


# ── the page section (used by app.py) ───────────────────────────────────────
import time  # noqa: E402

import ui  # noqa: E402

LEGEND = [
    ("Letters", "wlen", "word length in letters", "counted"),
    ("Sent. len", "sent_len", "words in the sentence", "counted"),
    ("Freq.", "wiki_freq", "how common the word is (log; nearer 0 = more common)", "wordfreq (Wikipedia)"),
    ("AoA", "aoa_mean", "age in years when the word is learned", "Kuperman et al. (2012)"),
    ("AoA sd", "aoa_std", "how much the AoA lists disagree", "Kuperman, Bird, Bristol norms"),
    ("Fwd p", "fwd_prob", "log P(word | previous word)", "BNC, 12.4 M word pairs"),
    ("Bwd p", "bwd_prob", "log P(word | next word)", "BNC"),
    ("Lex surpr.", "lex_surprisal", "how unexpected the word is after the previous word", "BNC"),
    ("Syn surpr.", "syn_surprisal", "how unexpected the grammar is at this word", "left-corner parser trained on Penn Treebank"),
    ("Entropy red.", "entropy_red", "how much this word reduces uncertainty about what comes next", "Penn Treebank POS bigrams"),
    ("Depth", "emb_depth", "how deep the word sits in the grammar tree (root = 0)", "dependency parse"),
    ("Depth diff", "emb_diff", "change in depth from the previous word", "dependency parse"),
    ("h1–h8", "h", "8 yes/no flags: depth changed, went deeper, went up, is root, (h5 unused), "
     "jumped down >1, jumped up >1, same depth", "dependency parse"),
]


def view(sentence):
    words = (sentence or "").strip().split()
    if not words:
        return ui.wrap("<p>Please type an English sentence.</p>")
    t0 = time.time()
    df, info = features(sentence)
    pred = predict(df)
    secs = time.time() - t0
    slow = int(np.argmax(pred["td"]))
    hdr = ("<tr><th rowspan=2>Word</th><th rowspan=2>POS</th>"
           "<th colspan=5 class='c'>Lexical (step 2)</th><th colspan=3 class='c'>BNC word pairs</th>"
           "<th colspan=2 class='c'>Parser</th><th colspan=3 class='c'>Grammar tree</th>"
           "<th colspan=4 class='c'>Predicted reading time (ms)</th></tr><tr>"
           + "".join(f"<th class='c'>{x}</th>" for x in
                     ["Letters", "Sent. len", "Freq.", "AoA", "AoA sd", "Fwd p", "Bwd p", "Lex surpr.",
                      "Syn surpr.", "Entropy red.", "Depth", "Depth diff", "h1–h8", "FFD", "FPD", "RPD", "TD"])
           + "</tr>")
    body = []
    for i, r in df.iterrows():
        h = "".join(str(r[f"h{k}"]) for k in range(1, 9))
        cells = [r.wlen, r.sent_len, f"{r.wiki_freq:.2f}", f"{r.aoa_mean:.1f}", f"{r.aoa_std:.2f}",
                 f"{r.fwd_prob:.2f}", f"{r.bwd_prob:.2f}", f"{r.lex_surprisal:.2f}",
                 f"{r.syn_surprisal:.2f}", f"{r.entropy_red:.2f}", r.emb_depth, f"{r.emb_diff:+d}",
                 f"<code>{h}</code>"] + [f"<b>{pred[m][i]:.0f}</b>" for m in MEASURES]
        body.append(f"<tr><td><b>{ui.esc(r.word)}</b></td><td class='c'>{ui.esc(r.pos)}</td>"
                    + "".join(f"<td class='c'>{c}</td>" for c in cells) + "</tr>")
    legend = "".join(f"<tr><td><b>{a}</b></td><td><code>{b}</code></td><td>{c}</td><td>{d}</td></tr>"
                     for a, b, c, d in LEGEND)
    parser_note = ("✓ live, same parser as the thesis" if info["parser"] and "live" in info["parser"]
                   else ui.esc(info["parser"] or ""))
    tree = ui.esc(info.get("tree") or "(no parse)")
    return ui.wrap(f"""
<div class="grid g3">
 <div class="card blue"><span class="tag">Words</span><div class="big" style="margin-top:6px">{len(words)}</div>
  <p class="muted">19 features each, computed in {secs:.1f} s</p></div>
 <div class="card orange"><span class="tag orange">Slowest word</span><div class="big orange" style="margin-top:6px">{ui.esc(words[slow])}</div>
  <p class="muted">{pred['td'][slow]:.0f} ms total time</p></div>
 <div class="card"><span class="tag grey">Whole sentence</span><div class="big" style="margin-top:6px">{pred['td'].sum():.0f} ms</div>
  <p class="muted">predicted total reading time · syntactic surprisal: {parser_note}</p></div>
</div>
<div class="card" style="margin-top:12px"><h3 style="margin-top:0">Step 2 · System 1 output: predicted reading time per word</h3>
{ui.bars(words, [("first fixation (FFD)", pred['ffd'], "#1F4E8C"), ("total time (TD)", pred['td'], "#B45309")], " ms")}</div>
<h3>Step 1 · All 19 features of every word (exactly what System 1 receives)</h3>
<div style="overflow-x:auto"><table class="t" style="font-size:13px">{hdr}{"".join(body)}</table></div>
<p class="muted">0 ms = readers are predicted to skip the word. h5 is always 0 (as in the thesis), so the model uses 19 values.</p>
<h3>What each feature means</h3>
<table class="t"><tr><th>Column</th><th>Name in code</th><th>Meaning</th><th>Source</th></tr>{legend}</table>
<details style="margin-top:10px"><summary>Grammar tree from the Berkeley parser (input to the left-corner parser)</summary>
<pre style="white-space:pre-wrap;font-size:13px">{tree}</pre></details>
<p class="muted">Same formulas and resources as phase1_replication steps 2–3. Differences from the stored Dundee
values: a typed sentence has no previous/next sentence, and depth comes from spaCy (Dundee had its own annotation).</p>""")

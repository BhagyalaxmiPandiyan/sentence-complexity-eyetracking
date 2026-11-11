"""
Phase 2 (English, GPT-2 enhancement) part of the toolkit.

For a typed English sentence: GPT-2 surprisal of every word and the predicted
reading times (FFD, FPD, RPD, TD) from System 1 with and without GPT-2
surprisal. Models: models/phase2_models.pkl (train_phase2.py). GPT-2 surprisal
uses the thesis function unchanged
(phase1_replication/experiments/compute_gpt2_surprisal.word_token_surprisals).

Not self-contained: needs phase1_replication/ next to toolkit/.
"""
import os
import pickle
import sys
import threading

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1_replication", "experiments"))

with open(os.path.join(HERE, "models", "phase2_models.pkl"), "rb") as f:
    MODELS = pickle.load(f)
MEASURES = ["ffd", "fpd", "rpd", "td"]

_ready = threading.Event()
_state = {}


def _load():
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import compute_gpt2_surprisal as cg
        _state["fn"] = cg.word_token_surprisals
        _state["tok"] = GPT2TokenizerFast.from_pretrained("gpt2")
        _state["lm"] = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
    _ready.set()


threading.Thread(target=_load, daemon=True).start()


def _predict(rows, rt, variant):
    m = MODELS[rt][variant]
    X = m["scaler"].transform(pd.DataFrame(rows)[m["features"]].fillna(0))
    p = m["model"].predict(X)
    p[p < m["threshold"]] = 0.0
    return p


def r2_table():
    lines = ["| Measure | Without GPT-2 | With GPT-2 surprisal |", "|---|---|---|"]
    for rt in MEASURES:
        b, g = MODELS[rt]["baseline"]["test_r2"], MODELS[rt]["gpt2"]["test_r2"]
        lines.append(f"| {rt.upper()} | {b:.3f} | **{g:.3f}** |")
    return "\n".join(lines)


def analyze(sentence, word_features):
    """Per-word GPT-2 surprisal + predicted reading times. `word_features` is app._word_features."""
    words = (sentence or "").strip().split()
    if not words:
        return "Please type an English sentence.", None
    _ready.wait()
    if "error" in _state:
        return "GPT-2 failed to load: " + _state["error"], None
    surp = _state["fn"](" ".join(words), _state["tok"], _state["lm"], "cpu")
    rows = []
    for i, w in enumerate(words):
        r = word_features(w, words, i)
        r["gpt2_surprisal"] = surp[i]
        rows.append(r)
    table = pd.DataFrame({"Word": words, "GPT-2 surprisal (nats)": [round(s, 2) for s in surp]})
    for rt in MEASURES:
        table[f"{rt.upper()} (ms)"] = _predict(rows, rt, "gpt2").round(0).astype(int)
    td_base = _predict(rows, "td", "baseline")
    table["TD without GPT-2 (ms)"] = td_base.round(0).astype(int)
    hard = table.loc[table["TD (ms)"].idxmax(), "Word"]
    surprising = table.loc[table["GPT-2 surprisal (nats)"].idxmax(), "Word"]
    md = (f"**Longest predicted reading time:** {hard} · **Most surprising word for GPT-2:** {surprising}\n\n"
          f"Predicted total reading time for the sentence: **{int(table['TD (ms)'].sum())} ms** "
          f"(without GPT-2: {int(td_base.sum())} ms). 0 ms = the model predicts readers skip the word.")
    return md, table

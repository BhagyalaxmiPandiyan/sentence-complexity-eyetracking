"""
System 2 — Sentence Readability Comparison (public demo)

Given two sentences, predicts which one is harder to read, using the
same pipeline as the thesis: System 1 predicts per-word reading time
(FFD) from lexical features, and System 2 (Logistic Regression) uses
those predicted reading times to classify which sentence is more complex.

Self-contained: only needs models/system1_models.pkl, models/system2_model.pkl,
and data/AoA.csv (all produced by train_system2.py). No dependency on the
rest of the repo, so this folder can be deployed on its own (e.g. to
Hugging Face Spaces).
"""

import math
import os
import pickle
import re

import gradio as gr
import numpy as np
import pandas as pd
from wordfreq import word_frequency

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load models and support data ──────────────────────────────────────────

with open(os.path.join(HERE, "models", "system1_models.pkl"), "rb") as f:
    SYSTEM1 = pickle.load(f)["ffd"]

with open(os.path.join(HERE, "models", "system2_model.pkl"), "rb") as f:
    SYSTEM2 = pickle.load(f)

AOA_DICT = {}
_aoa_df = pd.read_csv(os.path.join(HERE, "data", "AoA.csv"), encoding="latin-1")
for _, row in _aoa_df.iterrows():
    w = str(row["Word"]).lower().strip()
    v = row.get("AoA_Kup", float("nan"))
    if not pd.isna(v):
        vals = [row.get(c, float("nan")) for c in ["AoA_Kup", "AoA_Bird_lem", "AoA_Bristol_lem"]]
        vals = [x for x in vals if not pd.isna(x)]
        AOA_DICT[w] = (float(v), float(np.std(vals)) if len(vals) > 1 else 0.0)

_DEFAULT_AOA_MEAN = float(np.mean([v[0] for v in AOA_DICT.values()]))
_DEFAULT_AOA_STD = float(np.mean([v[1] for v in AOA_DICT.values()]))


def _word_freq(w):
    return math.log(word_frequency(w, "en") + 1e-9)


# ── Feature extraction (lexical-only path, matches step5_system2.py) ──────

def _word_features(word, sentence_words, pos):
    w = str(word).lower().strip(".,!?;:'\"()")
    wlen = len(re.sub(r"[^a-zA-Z]", "", word))
    aoa_mean, aoa_std = AOA_DICT.get(w, (_DEFAULT_AOA_MEAN, _DEFAULT_AOA_STD))
    return {
        "wlen": wlen,
        "sent_len": len(sentence_words),
        "wiki_freq": _word_freq(w),
        "aoa_mean": aoa_mean,
        "aoa_std": aoa_std,
        "fwd_prob": 0.0,
        "bwd_prob": 0.0,
        "total_surprisal": 0.0,
        "lex_surprisal": 0.0,
        "syn_surprisal": 0.0,
        "entropy_red": 0.0,
        "emb_depth": 0,
        "emb_diff": 0,
        "h1": 0, "h2": 0, "h3": 0, "h4": 0,
        "h5": 0, "h6": 0, "h7": 0, "h8": 0,
    }


def predict_ffd(sentence):
    """Predict per-word First Fixation Duration for a sentence (System 1)."""
    words = sentence.strip().split()
    if not words:
        return []
    rows = [_word_features(w, words, i) for i, w in enumerate(words)]
    feature_cols = SYSTEM1["features"]
    X = pd.DataFrame(rows)[feature_cols].fillna(0)
    X_scaled = SYSTEM1["scaler"].transform(X)
    preds = SYSTEM1["model"].predict(X_scaled)
    preds[preds < SYSTEM1["threshold"]] = 0.0
    return list(preds)


def _sentence_surprisal(words):
    return sum(_word_features(w, words, i)["lex_surprisal"] for i, w in enumerate(words))


def build_feature_vector(sentence_a, sentence_b, max_words=60):
    """Same layout as build_pair_features + build_extended_features in step5_system2.py."""
    ffd_a = predict_ffd(sentence_a)[:max_words]
    ffd_b = predict_ffd(sentence_b)[:max_words]
    ffd_a = ffd_a + [0.0] * (max_words - len(ffd_a))
    ffd_b = ffd_b + [0.0] * (max_words - len(ffd_b))

    words_a = sentence_a.strip().split()
    words_b = sentence_b.strip().split()

    extra = [
        len(words_a), len(words_b),
        float(np.mean(ffd_a)) if ffd_a else 0.0,
        float(np.mean(ffd_b)) if ffd_b else 0.0,
        _sentence_surprisal(words_a),
        _sentence_surprisal(words_b),
    ]

    return np.array([ffd_a + ffd_b + extra], dtype=np.float32)


# ── Prediction ──────────────────────────────────────────────────────────────

def compare_sentences(sentence_a, sentence_b):
    if not sentence_a.strip() or not sentence_b.strip():
        return "Please enter both sentences."

    X = build_feature_vector(sentence_a, sentence_b)
    X_scaled = SYSTEM2["scaler"].transform(X)
    prob_a_harder = SYSTEM2["model"].predict_proba(X_scaled)[0][1]

    if prob_a_harder >= 0.5:
        return f"**Sentence A is more complex** (confidence: {prob_a_harder * 100:.1f}%)"
    else:
        return f"**Sentence B is more complex** (confidence: {(1 - prob_a_harder) * 100:.1f}%)"


import phase2  # noqa: E402  (Phase 2; loads GPT-2 in the background)
import tamil  # noqa: E402  (Phase 3; starts loading its models in the background)

with gr.Blocks(title="Sentence Difficulty Toolkit") as demo:
    gr.Markdown("# Sentence Difficulty Toolkit\nMeasuring and reducing sentence difficulty in English and Tamil.")

    with gr.Tab("English · Phase 1"):
        gr.Markdown(
            "Predicts which of two sentences is harder to read, based on predicted eye-tracking "
            "reading times (System 1) fed into a pairwise readability classifier (System 2). "
            "Replication + extension of Singh et al. (2016)."
        )
        en_a = gr.Textbox(label="Sentence A", placeholder="e.g. The quick brown fox jumps over the lazy dog.")
        en_b = gr.Textbox(label="Sentence B", placeholder="e.g. The expeditious auburn vulpine traverses the indolent canine.")
        en_btn = gr.Button("Compare", variant="primary")
        en_out = gr.Markdown()
        gr.Examples(
            [["The dog ran to the park.",
              "The canine proceeded expeditiously towards the municipal recreational grounds."],
             ["Photosynthesis is the biochemical process whereby chlorophyll-containing organisms "
              "convert electromagnetic radiation into chemical energy.",
              "Plants use sunlight to make food."]],
            inputs=[en_a, en_b],
        )
        en_btn.click(compare_sentences, [en_a, en_b], en_out)

    with gr.Tab("English · Phase 2 (GPT-2)"):
        gr.Markdown(
            "Word-by-word reading times. Type one English sentence: GPT-2 measures how surprising "
            "each word is, and System 1 predicts how long readers look at it (Dundee measures FFD, "
            "FPD, RPD, TD). Adding GPT-2 surprisal improves all four measures on the Dundee test set "
            "(R², higher is better):\n\n" + phase2.r2_table()
        )
        p2_in = gr.Textbox(label="English sentence",
                           placeholder="e.g. The old man the boats while the children sleep.")
        p2_btn = gr.Button("Predict reading times", variant="primary")
        p2_out = gr.Markdown()
        p2_table = gr.Dataframe(label="Per word", wrap=True)
        gr.Examples([["The cat sat on the mat."],
                     ["The horse raced past the barn fell."],
                     ["Despite the ostensibly benign forecast, the expedition encountered a ferocious blizzard."]],
                    inputs=[p2_in])
        p2_btn.click(lambda s: phase2.analyze(s, _word_features), [p2_in], [p2_out, p2_table])

    with gr.Tab("Tamil · Phase 3"):
        gr.Markdown(
            "Predicts which of two Tamil sentences is harder to read, using the thesis's "
            "word-level difficulty metric (a pairwise classifier trained on StoryWeaver "
            "reading levels). Wait about a minute after start-up for the Tamil models to load."
        )
        gr.Markdown("### ① Compare two sentences")
        ta_a = gr.Textbox(label="Sentence A")
        ta_b = gr.Textbox(label="Sentence B")
        ta_btn = gr.Button("Compare", variant="primary")
        ta_out = gr.Markdown()
        gr.Examples(tamil.EXAMPLES, inputs=[ta_a, ta_b])
        ta_btn.click(tamil.compare_two, [ta_a, ta_b], ta_out)

        gr.Markdown("---\n### ② Analyse one sentence: word-level vs morphology-aware metric")
        an_in = gr.Textbox(label="Tamil sentence", lines=2)
        an_btn = gr.Button("Analyse", variant="primary")
        an_out = gr.Markdown()
        an_morph = gr.Dataframe(label="Step 1 · Word by word: every number the features are built from", wrap=True)
        an_feats = gr.Dataframe(label="Step 2 · The 12 sentence features and how each metric adds them up", wrap=True)
        gr.Examples([[tamil.EXAMPLE]], inputs=[an_in])
        an_btn.click(tamil.analyze, [an_in], [an_out, an_morph, an_feats])

        gr.Markdown("---\n### ③ Simplify and check: LLM simplification + both metrics + human study\n"
                    "Type a sentence and press **Simplify with Gemini** (live, free tier), "
                    "or pick one of the 30 StoryWeaver pairs from the human study. Then press **Compare**.")
        with gr.Row():
            sp_pick = gr.Dropdown(list(tamil.PAIRS), label="Or load a study pair (StoryWeaver)", value=None)
        sp_orig = gr.Textbox(label="Original", lines=2)
        sp_gem = gr.Button("Simplify with Gemini")
        sp_simp = gr.Textbox(label="Simplified (you can edit it)", lines=3)
        sp_msg = gr.Markdown()
        sp_btn = gr.Button("Compare original vs simplified", variant="primary")
        sp_out = gr.Markdown()
        sp_feats = gr.Dataframe(label="Feature changes", wrap=True)
        sp_pick.change(tamil.load_pair, [sp_pick], [sp_orig, sp_simp, sp_msg])
        sp_gem.click(tamil.simplify, [sp_orig], [sp_simp, sp_msg])
        sp_btn.click(tamil.compare, [sp_orig, sp_simp], [sp_out, sp_feats])


if __name__ == "__main__":
    demo.launch(share=True)

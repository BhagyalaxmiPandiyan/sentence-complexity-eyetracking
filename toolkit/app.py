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
import ui  # noqa: E402  (look and explanation pages)


def phase1_view(sentence_a, sentence_b):
    """Verdict + the System 1 output (predicted first-fixation time per word) for both sentences."""
    if not sentence_a.strip() or not sentence_b.strip():
        return ui.wrap("<p>Please enter both sentences.</p>")
    X = build_feature_vector(sentence_a, sentence_b)
    p_a = SYSTEM2["model"].predict_proba(SYSTEM2["scaler"].transform(X))[0][1]
    harder, conf = ("A", p_a) if p_a >= 0.5 else ("B", 1 - p_a)
    parts = []
    for name, sent, colour in [("A", sentence_a, "#1F4E8C"), ("B", sentence_b, "#B45309")]:
        words = sent.strip().split()
        ffd = predict_ffd(sent)
        feats = [_word_features(w, words, i) for i, w in enumerate(words)]
        rows = "".join(
            f'<tr><td>{ui.esc(w)}</td><td class="c">{f["wlen"]}</td><td class="c">{f["wiki_freq"]:.1f}</td>'
            f'<td class="c">{f["aoa_mean"]:.1f}</td><td class="c"><b>{v:.0f}</b></td></tr>'
            for w, f, v in zip(words, feats, ffd))
        tag = "orange" if name == harder else ""
        parts.append(f"""<div class="card">
<span class="tag {tag}">Sentence {name}{" · harder" if name == harder else ""}</span>
<p class="muted" style="margin-top:6px">{len(words)} words · mean predicted first fixation <b>{np.mean(ffd):.0f} ms</b></p>
{ui.bars(words, [("predicted first fixation (ms)", ffd, colour)], " ms")}
<details style="margin-top:8px"><summary>Step 1 · the features of every word</summary>
<table class="t" style="margin-top:6px"><tr><th>Word</th><th class="c">Letters</th><th class="c">log frequency</th>
<th class="c">Age learned</th><th class="c">Predicted FFD (ms)</th></tr>{rows}</table></details></div>""")
    return ui.wrap(f"""
<div class="verdict {"orange" if harder == "B" else ""}">Sentence {harder} is harder to read
 ({conf * 100:.1f}% confident)</div>
<p class="muted">Step 2 · System 1 predicted how long readers look at each word (0 ms = readers usually skip it).
Step 3 · System 2 compared the two lists and decided.</p>
<div class="grid g2">{"".join(parts)}</div>""")


def phase2_view(sentence):
    md, table = phase2.analyze(sentence, _word_features)
    if table is None:
        return ui.wrap(f"<p>{ui.esc(md)}</p>"), None
    words = table["Word"].tolist()
    surp = table["GPT-2 surprisal (nats)"]
    hard = table.loc[table["TD (ms)"].idxmax(), "Word"]
    top = table.loc[surp.idxmax(), "Word"]
    body = f"""
<div class="grid g3">
 <div class="card blue"><span class="tag">Most surprising word</span><div class="big" style="margin-top:6px">{ui.esc(top)}</div>
  <p class="muted">GPT-2 surprisal {surp.max():.2f}</p></div>
 <div class="card orange"><span class="tag orange">Longest reading time</span><div class="big orange" style="margin-top:6px">{ui.esc(hard)}</div>
  <p class="muted">{int(table["TD (ms)"].max())} ms total time</p></div>
 <div class="card"><span class="tag grey">Whole sentence</span><div class="big" style="margin-top:6px">{int(table["TD (ms)"].sum())} ms</div>
  <p class="muted">without GPT-2: {int(table["TD without GPT-2 (ms)"].sum())} ms</p></div>
</div>
<div class="grid g2" style="margin-top:12px">
 <div class="card"><h3 style="margin-top:0">Step 1 · How surprising is each word? (GPT-2)</h3>
  {ui.bars(words, [("surprisal", surp, "#1F4E8C")], "", "{:.2f}")}</div>
 <div class="card"><h3 style="margin-top:0">Step 2 · Predicted total reading time (TD)</h3>
  {ui.bars(words, [("with GPT-2", table["TD (ms)"], "#B45309"), ("without GPT-2", table["TD without GPT-2 (ms)"], "#9AA5B1")], " ms")}</div>
</div>
<p class="muted">Look for words where the orange bar is longer than the grey one: GPT-2 found them surprising,
so readers are predicted to slow down there. All four measures are in the table below.</p>"""
    return ui.wrap(body), table


EX_HARD = "ரஞ்சுவின் கிராமத்தில் உள்ள நீர்நிலை வறண்டுவிடுகிறது, தண்ணீரைக் கண்டுபிடிக்க அவள் புறப்படுகிறாள்."
EX_EASY = "அப்புறம், அதற்கு என்ன ஆகின்றது என்று பாருங்கள்."

with gr.Blocks(title="Sentence Difficulty Toolkit") as demo:
    gr.HTML(ui.wrap('<h2 style="margin-bottom:0">Sentence Difficulty Toolkit</h2>'
                    '<p class="muted">M.Tech project · Bhagyalaxmi Pandiyan (M25DE2070) · '
                    'supervisor Dr. Sidharth Ranjan · IIT Jodhpur</p>'))

    with gr.Tab("Home"):
        gr.HTML(ui.HOME)

    with gr.Tab("Phase 1 · English"):
        gr.HTML(ui.PHASE1_INTRO)
        with gr.Row():
            en_a = gr.Textbox(label="Sentence A", placeholder="e.g. The dog ran to the park.")
            en_b = gr.Textbox(label="Sentence B",
                              placeholder="e.g. The committee's protracted deliberations culminated in an unanticipated resolution.")
        en_btn = gr.Button("Which sentence is harder?", variant="primary")
        gr.Examples(
            [["The dog ran to the park.",
              "The committee's protracted deliberations culminated in an unanticipated resolution."],
             ["Photosynthesis is the biochemical process whereby chlorophyll-containing organisms "
              "convert electromagnetic radiation into chemical energy.",
              "Plants use sunlight to make food."]],
            inputs=[en_a, en_b])
        en_out = gr.HTML()
        en_btn.click(phase1_view, [en_a, en_b], en_out)

    with gr.Tab("Phase 2 · GPT-2"):
        gr.HTML(ui.PHASE2_INTRO)
        p2_in = gr.Textbox(label="English sentence", placeholder="e.g. I drank a cup of hot elephant this morning.")
        p2_btn = gr.Button("Predict reading times", variant="primary")
        gr.Examples([["I drank a cup of hot coffee this morning."],
                     ["I drank a cup of hot elephant this morning."],
                     ["Despite the ostensibly benign forecast, the expedition encountered a ferocious blizzard."]],
                    inputs=[p2_in])
        p2_out = gr.HTML()
        with gr.Accordion("All four reading times per word (FFD, FPD, RPD, TD)", open=False):
            p2_table = gr.Dataframe(wrap=True)
        p2_btn.click(phase2_view, [p2_in], [p2_out, p2_table])

    with gr.Tab("Phase 3 · Tamil"):
        gr.HTML(ui.wrap('<p class="muted">The Tamil models take about a minute to load after start-up.</p>'))
        with gr.Tab("① How we did it"):
            gr.HTML(ui.PHASE3_STORY)

        with gr.Tab("② Analyse a sentence"):
            gr.HTML(ui.ANALYSE_INTRO)
            an_in = gr.Textbox(label="Tamil sentence", lines=2)
            an_btn = gr.Button("Analyse", variant="primary")
            gr.Examples([[EX_HARD], [EX_EASY]], inputs=[an_in])
            an_top = gr.HTML()
            an_cards = gr.HTML()
            with gr.Accordion("Proof · every number per word (the 12 features are averages of these columns)", open=False):
                an_words = gr.Dataframe(wrap=True)
            with gr.Accordion("Proof · how each metric adds weight × feature into its score", open=False):
                an_proof = gr.Dataframe(wrap=True)
            an_btn.click(tamil.analyze_view, [an_in], [an_top, an_cards, an_words, an_proof])

        with gr.Tab("③ Which is harder?"):
            gr.HTML(ui.wrap("<h3>③ Compare two Tamil sentences</h3><p class='muted'>The Tamil version of "
                            "System 2: the word metric decides, and the morph metric is shown as agree / disagree.</p>"))
            with gr.Row():
                ta_a = gr.Textbox(label="Sentence A", lines=2)
                ta_b = gr.Textbox(label="Sentence B", lines=2)
            ta_btn = gr.Button("Which sentence is harder?", variant="primary")
            gr.Examples([[EX_EASY, EX_HARD]] + tamil.EXAMPLES, inputs=[ta_a, ta_b])
            ta_out = gr.HTML()
            ta_btn.click(tamil.compare_two_view, [ta_a, ta_b], ta_out)

        with gr.Tab("④ Simplify"):
            gr.HTML(ui.wrap(
                "<h3>④ Simplify a hard sentence and check it</h3>"
                + ui.flow([("1 · Hard sentence", "type one, or load one of the 30 StoryWeaver pairs from the human study"),
                           ("2 · LLM rewrites it", "Gemini, with the same prompt as the thesis (or the saved study version)"),
                           ("3 · Both metrics score it", "did the level go down?"),
                           ("4 · Human check", "for study pairs: what the native readers said")])))
            sp_pick = gr.Dropdown(list(tamil.PAIRS), label="Load a study pair (StoryWeaver, rated by native readers)",
                                  value=None)
            with gr.Row():
                sp_orig = gr.Textbox(label="Original (hard)", lines=3)
                sp_simp = gr.Textbox(label="Simplified (you can edit it)", lines=3)
            with gr.Row():
                sp_gem = gr.Button("Simplify with Gemini")
                sp_btn = gr.Button("Compare original vs simplified", variant="primary")
            sp_msg = gr.Markdown()
            sp_human = gr.HTML()
            sp_out = gr.HTML()
            sp_pick.change(tamil.load_pair_view, [sp_pick], [sp_orig, sp_simp, sp_human])
            sp_gem.click(tamil.simplify, [sp_orig], [sp_simp, sp_msg])
            sp_btn.click(tamil.compare_view, [sp_orig, sp_simp], sp_out)


if __name__ == "__main__":
    import sys
    # Local only by default (http://127.0.0.1:7860). Add --share for a temporary public link.
    demo.launch(share="--share" in sys.argv, css=ui.CSS, js=ui.FORCE_LIGHT, theme=gr.themes.Soft(primary_hue="blue"))

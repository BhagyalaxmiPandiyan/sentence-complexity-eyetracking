"""
Phase 3 (Tamil) part of the toolkit.

Scores Tamil sentences with the two thesis metrics, shows the morpheme breakdown,
simplifies with Gemini (same prompt as the thesis), and compares original vs
simplified. Reuses the thesis code in phase3_tamil_extension/src unchanged:

    step3_textbook_features.load_tools / sentence_features   the 12 features
    metric_model.fit_metrics / score                          word and morph metrics

Unlike the English tab this is NOT self-contained: it needs the phase3 folder
(tools/indic_nlp_resources, output/*.csv) next to toolkit/.
"""
import os
import re
import sys
import threading

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE3 = os.path.join(os.path.dirname(HERE), "phase3_tamil_extension")
sys.path.insert(0, os.path.join(PHASE3, "src"))

import metric_model as mm                       # noqa: E402
import step3_textbook_features as s3            # noqa: E402

SCORES_CSV = os.path.join(PHASE3, "output", "simplification_scores.csv")
RATINGS_CSV = os.path.join(PHASE3, "output", "human_ratings_long.csv")
SW_FEATS_CSV = os.path.join(PHASE3, "output", "storyweaver_features.csv")

SENT_END = re.compile(r"(?<=[.?!])\s+")          # same split as step8_score_simplified.py
TAMIL_CHAR = re.compile("[஀-௿]")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

# Same rules as Dataset/complex_sentences/prompt_batch*.txt (the thesis simplifications)
PROMPT = """You are an expert Tamil teacher. Rewrite the Tamil sentence below so that a child in Class 3-5 (about 8-10 years old) can read and understand it easily.

Rules:
1. Keep the same meaning. Do not add new information and do not drop important information.
2. Use common, everyday Tamil words instead of rare or literary words.
3. Use shorter words: avoid long words with many suffixes joined together where a simpler form exists.
4. You may split one long sentence into two or three short sentences.
5. Write natural, correct written Tamil (not colloquial spoken Tamil, not English, no transliteration).
6. If a sentence is already simple, still write the simplest version you can.

Output only the simplified Tamil text, nothing else.

Sentence: {text}"""

FEATURE_LABELS = [
    ("n_words", "Words in the sentence", "A · word-level"),
    ("word_len", "Mean letters per word", "A · word-level"),
    ("word_zipf", "Mean word frequency (Zipf, higher = more common)", "A · word-level"),
    ("word_surprisal", "Mean GPT-2 surprisal per word (bits)", "A · word-level"),
    ("morphs_per_word", "Mean morphemes per word", "B · morphology"),
    ("stem_surprisal", "Mean stem surprisal (bits)", "B · morphology"),
    ("suffix_surprisal", "Mean suffix surprisal (bits)", "B · morphology"),
    ("feats_per_word", "Marked grammar features per word", "B · morphology"),
    ("lemma_zipf", "Mean dictionary-form frequency (Zipf)", "B · morphology"),
    ("nonfinite_verbs", "Non-finite verbs (participles, converbs …)", "B · morphology"),
]

# ── loading (in a background thread, so the English tab starts at once) ──────
_ready = threading.Event()
_state = {}


def _load():
    try:
        _state["tools"] = s3.load_tools()                      # Morfessor, Stanza, Tamil GPT-2
        _state["metrics"] = mm.fit_metrics()                   # word + morph weights
        sw = pd.read_csv(SW_FEATS_CSV)
        sc = mm.score(sw, _state["metrics"])
        _state["levels"] = pd.concat([sw["level"], sc], axis=1).groupby("level").mean()
    except Exception as e:                                     # shown in the UI instead of crashing
        _state["error"] = f"{type(e).__name__}: {e}"
    _ready.set()


threading.Thread(target=_load, daemon=True).start()


def _wait():
    _ready.wait()
    if "error" in _state:
        raise RuntimeError("Tamil tools failed to load: " + _state["error"])


# ── features and scores ──────────────────────────────────────────────────────
def text_features(text):
    """The 12 thesis features, averaged over the sentences of `text` (as in step 8)."""
    parts = [p for p in SENT_END.split(text.strip()) if TAMIL_CHAR.search(p)]
    if not parts:
        return None, 0
    rows = [s3.sentence_features(p, *_state["tools"]) for p in parts]
    return pd.DataFrame(rows).mean(), len(parts)


def text_scores(f):
    return {name: float(f[cols].to_numpy() @ w) for name, (cols, w) in _state["metrics"].items()}


def nearest_level(metric, value):
    lv = _state["levels"][f"score_{metric}"]
    return int((lv - value).abs().idxmin())


def morpheme_table(text):
    """One row per word: Morfessor morphemes + Stanza lemma / POS / marked features."""
    seg, nlp = _state["tools"][0], _state["tools"][1]
    info = []
    for p in [p for p in SENT_END.split(text.strip()) if TAMIL_CHAR.search(p)]:
        for s in nlp(p).sentences:
            for t in s.tokens:
                ws = [w for w in t.words if w.upos != "PUNCT"]
                if not ws:
                    continue
                feats = [f for w in ws for f in ((w.feats or "").split("|") if w.feats else [])]
                marked = [f for f in feats if f not in s3.UNMARKED and not f.startswith(s3.IGNORED)]
                info.append({
                    "word": t.text,
                    "lemma": " + ".join(w.lemma or "" for w in ws),
                    "pos": " + ".join(w.upos for w in ws),
                    "marked": ", ".join(marked) or "–",
                    "nonfinite": "yes" if s3.NONFINITE & set(feats) else "",
                })
    rows = []
    for d in info:
        w = d["word"]
        morphs = [w]
        if s3.TAMIL.fullmatch(w):
            m = seg.morph_analyze(w)
            if "".join(m) == w:                     # same rule as step 3
                morphs = m
        rows.append([w, " + ".join(morphs), len(morphs), d["lemma"], d["pos"], d["marked"], d["nonfinite"]])
    return pd.DataFrame(rows, columns=["Word", "Morphemes (Morfessor)", "# morphemes",
                                       "Dictionary form (Stanza)", "POS", "Marked features", "Non-finite verb"])


def _feature_rows(fs):
    return [[label, group] + [round(float(f[col]), 2) for f in fs] for col, label, group in FEATURE_LABELS]


# ── UI callbacks ─────────────────────────────────────────────────────────────
def analyze(text):
    if not text or not TAMIL_CHAR.search(text):
        return "Please type a Tamil sentence.", None, None
    _wait()
    f, n = text_features(text)
    sc = text_scores(f)
    md = (
        f"| Metric | Difficulty score | Reads like StoryWeaver level |\n|---|---|---|\n"
        f"| **Word-level** (group A) | {sc['word']:+.2f} | **{nearest_level('word', sc['word'])}** |\n"
        f"| **Morphology-aware** (groups A + B) | {sc['morph']:+.2f} | **{nearest_level('morph', sc['morph'])}** |\n\n"
        f"Higher score = harder. Scores only mean something compared with each other "
        f"(StoryWeaver level 1 = easiest, 4 = hardest)."
        + (f" Text has {n} sentences; features are averaged over them." if n > 1 else "")
    )
    feats = pd.DataFrame(_feature_rows([f]), columns=["Feature", "Group", "Value"])
    return md, morpheme_table(text), feats


def _gemini_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key and sys.platform == "win32":                  # set with setx but terminal not reopened
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "GEMINI_API_KEY")[0]
        except OSError:
            key = None
    return key


def simplify(text):
    if not text or not TAMIL_CHAR.search(text):
        return "", "Please type a Tamil sentence first."
    key = _gemini_key()
    if not key:
        return "", "No GEMINI_API_KEY found, so live simplification is off. Load a saved study pair instead."
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT.format(text=text.strip())}]}],
            "generationConfig": {"temperature": 0}}
    last = ""
    for model in GEMINI_MODELS:
        try:
            r = requests.post(GEMINI_URL.format(model), headers={"x-goog-api-key": key}, json=body, timeout=60)
            if r.ok:
                out = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                return out, f"Simplified by **{model}** (free tier)."
            last = f"{model}: HTTP {r.status_code}"
        except Exception as e:
            last = f"{model}: {type(e).__name__}"
    return "", f"Gemini did not answer ({last}). Free-tier quota may be used up; load a saved study pair instead."


def compare(original, simplified):
    if not (original and simplified and TAMIL_CHAR.search(original) and TAMIL_CHAR.search(simplified)):
        return "Please fill in both boxes with Tamil text.", None
    _wait()
    fo, _ = text_features(original)
    fs, _ = text_features(simplified)
    so, ss = text_scores(fo), text_scores(fs)
    lines = ["| Metric | Original | Simplified | Change | Verdict |", "|---|---|---|---|---|"]
    for m, name in [("word", "Word-level"), ("morph", "Morphology-aware")]:
        d = ss[m] - so[m]
        verdict = "✅ simplified is easier" if d < 0 else "❌ simplified is not easier"
        lines.append(f"| **{name}** | {so[m]:+.2f} | {ss[m]:+.2f} | {d:+.2f} | {verdict} |")
    feats = pd.DataFrame(_feature_rows([fo, fs]), columns=["Feature", "Group", "Original", "Simplified"])
    return "\n".join(lines), feats


def compare_two(sentence_a, sentence_b):
    """System-2-style verdict: which of two Tamil sentences is harder?

    The metric is the step-6 pair model (logistic regression, no intercept), so
    score(A) - score(B) is the log-odds that A is harder; the sigmoid turns it
    into a probability. Verdict from the word-level metric (the best one in the
    thesis); the morphology metric is reported as agree / disagree.
    """
    if not (sentence_a and sentence_b and TAMIL_CHAR.search(sentence_a) and TAMIL_CHAR.search(sentence_b)):
        return "Please enter both Tamil sentences."
    _wait()
    fa, _ = text_features(sentence_a)
    fb, _ = text_features(sentence_b)
    sa, sb = text_scores(fa), text_scores(fb)
    p_a = 1 / (1 + np.exp(-(sa["word"] - sb["word"])))
    harder, conf = ("A", p_a) if p_a >= 0.5 else ("B", 1 - p_a)
    morph_a_harder = sa["morph"] > sb["morph"]
    agrees = morph_a_harder == (harder == "A")
    return (f"**Sentence {harder} is more complex** (confidence: {conf * 100:.1f}%)\n\n"
            f"Morphology-aware metric: {'agrees' if agrees else 'disagrees'}.")


# ── saved study pairs (StoryWeaver only: textbook sentences are copyrighted) ─
def _load_pairs():
    s = pd.read_csv(SCORES_CSV)
    s = s[s.source == "storyweaver"]
    h = pd.read_csv(RATINGS_CSV)
    pairs = {}
    for r in s.itertuples():
        hr = h[h.id == r.id]
        pairs[f"{r.id} · StoryWeaver level {r.level}"] = {
            "original": r.original, "simplified": r.simplified, "n": len(hr),
            "diff_orig": hr.diff_orig.mean(), "diff_simp": hr.diff_simp.mean(),
            "meaning": hr.meaning.mean(), "picked_simp": (hr.easier == "simplified").mean(),
            "err_simp": hr.err_simp.mean(),
        }
    return pairs


PAIRS = _load_pairs()


def load_pair(label):
    p = PAIRS.get(label)
    if not p:
        return "", "", ""
    md = (f"**What the {p['n']} native readers said about this pair** (human study, Phase C)\n\n"
          f"| | Original | Simplified |\n|---|---|---|\n"
          f"| Difficulty (1 = very easy, 5 = very hard) | {p['diff_orig']:.2f} | {p['diff_simp']:.2f} |\n"
          f"| Picked as easier | | {p['picked_simp']:.0%} of readers |\n"
          f"| Has errors | | {p['err_simp']:.0%} of readers |\n"
          f"| Meaning kept (1–5) | {p['meaning']:.2f} | |")
    return p["original"], p["simplified"], md


EXAMPLE = PAIRS[next(iter(PAIRS))]["original"] if PAIRS else ""

# Two study pairs for the examples: hard first, then easy first
_p = list(PAIRS.values())
EXAMPLES = [[_p[0]["original"], _p[0]["simplified"]],
            [_p[1]["simplified"], _p[1]["original"]]] if len(_p) > 1 else []

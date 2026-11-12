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
    ("tree_depth", "Dependency tree depth", "C · syntax (not in either metric)"),
    ("dep_distance", "Mean word-to-head distance", "C · syntax (not in either metric)"),
]
HOW = {   # how each sentence value is built from the word table (shown in the feature table)
    "n_words": "count of words",
    "word_len": "mean of 'Letters'",
    "word_zipf": "mean of 'Word freq.' (Tamil words only)",
    "word_surprisal": "mean of 'GPT-2 surpr.'",
    "morphs_per_word": "mean of '# parts'",
    "stem_surprisal": "mean of 'Stem surpr.'",
    "suffix_surprisal": "mean of 'Suffix surpr.'",
    "feats_per_word": "sum of '# marks' ÷ words",
    "lemma_zipf": "mean of 'Root freq.'",
    "nonfinite_verbs": "count of 'Non-finite' = yes",
    "tree_depth": "largest 'Depth'",
    "dep_distance": "mean of 'Distance'",
}

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
        _state["feat_levels"] = sw.groupby("level")[[c for c, _, _ in FEATURE_LABELS]].mean()
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


WORD_COLS = ["Word", "Letters", "Word freq.", "GPT-2 surpr.", "Morphemes (Morfessor)", "# parts",
             "Stem surpr.", "Suffix surpr.", "Root (Stanza)", "Root freq.", "Grammar marks", "# marks",
             "Non-finite", "Depends on", "Distance", "Depth"]


def word_table(text):
    """One row per word with every number the 12 sentence features are built from.
    Uses exactly the step-3 rules (s3.char_surprisal, s3.WORD, Morfessor, Stanza), so the
    column means equal the sentence features for a one-sentence input."""
    from wordfreq import zipf_frequency
    seg, nlp, tok, lm = _state["tools"]
    rows = []
    for p in [p for p in SENT_END.split(text.strip()) if TAMIL_CHAR.search(p)]:
        per_char = s3.char_surprisal(p, tok, lm)
        info = {}                                   # Stanza token start offset -> parse info
        for s in nlp(p).sentences:
            by_id = {w.id: w for w in s.words}
            for t in s.tokens:
                ws = [w for w in t.words if w.upos != "PUNCT"]
                if not ws:
                    continue
                feats = [f for w in ws for f in ((w.feats or "").split("|") if w.feats else [])]
                marked = [f for f in feats if f not in s3.UNMARKED and not f.startswith(s3.IGNORED)]
                w0 = ws[0]
                depth, h = 0, w0.id
                while h and depth < 100:
                    h = by_id[h].head if h in by_id else 0
                    depth += 1
                head = by_id.get(w0.head)
                lz = [zipf_frequency(w.lemma, "ta") for w in ws if w.lemma and s3.TAMIL.search(w.lemma)]
                info[t.start_char] = {
                    "lemma": " + ".join(w.lemma or "" for w in ws),
                    "lemma_z": round(float(np.mean(lz)), 2) if lz else "",
                    "marked": ", ".join(m.split("=")[1] if "=" in m else m for m in marked) or "–",
                    "n_marked": len(marked),
                    "nonfinite": "yes" if s3.NONFINITE & set(feats) else "",
                    "head": head.text if head else "(main verb)",
                    "dist": abs(w0.id - w0.head) if w0.head else "",
                    "depth": depth,
                }
        for m in s3.WORD.finditer(p):
            w = m.group()
            morphs = [w]
            if s3.TAMIL.fullmatch(w):
                seg_m = seg.morph_analyze(w)
                if "".join(seg_m) == w:             # same rule as step 3
                    morphs = seg_m
            stem_end = m.start() + len(morphs[0])
            d = info.get(m.start(), {})
            rows.append([
                w, len(w),
                round(zipf_frequency(w, "ta"), 2) if s3.TAMIL.fullmatch(w) else "",
                round(float(per_char[m.start():m.end()].sum()), 2),
                " + ".join(morphs), len(morphs),
                round(float(per_char[m.start():stem_end].sum()), 2),
                round(float(per_char[stem_end:m.end()].sum()), 2),
                d.get("lemma", ""), d.get("lemma_z", ""), d.get("marked", ""), d.get("n_marked", ""),
                d.get("nonfinite", ""), d.get("head", ""), d.get("dist", ""), d.get("depth", ""),
            ])
    return pd.DataFrame(rows, columns=WORD_COLS)


def feature_proof(f):
    """Sentence features with how each is computed and what it adds to each metric's score."""
    rows = []
    weights = {name: dict(zip(cols, w)) for name, (cols, w) in _state["metrics"].items()}
    for col, label, group in FEATURE_LABELS:
        v = float(f[col])
        cw, cm = weights["word"].get(col), weights["morph"].get(col)
        rows.append([label, group, round(v, 2), HOW[col],
                     f"{cw:+.3f} × {v:.2f} = {cw * v:+.2f}" if cw is not None else "–",
                     f"{cm:+.3f} × {v:.2f} = {cm * v:+.2f}" if cm is not None else "–"])
    sc = text_scores(f)
    rows.append(["SCORE = sum of the column above", "", "", "", f"{sc['word']:+.2f}", f"{sc['morph']:+.2f}"])
    return pd.DataFrame(rows, columns=["Feature", "Group", "Value", "How it is computed",
                                       "Word metric: weight × value", "Morph metric: weight × value"])


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
    md += ("\n\n**How to read the tables below:** the word table shows every number per word "
           "(Morfessor parts, Stanza root and grammar marks, Tamil GPT-2 surprisal in bits, "
           "frequency on the 0–7 Zipf scale, where 0 = not in the list). The feature table shows "
           "how each sentence value comes from that word table, and how weight × value adds up "
           "to each metric's score.")
    return md, word_table(text), feature_proof(f)


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


# ── visual views (used by app.py) ────────────────────────────────────────────
import ui  # noqa: E402

KEY = ["n_words", "word_len", "word_zipf", "morphs_per_word", "nonfinite_verbs"]
LABEL = {c: lab for c, lab, _ in FEATURE_LABELS}


def _feature_compare_table(cols_vals, cols=None):
    """HTML table: feature | value(s) | typical StoryWeaver level 1 | level 4."""
    fl = _state["feat_levels"]
    cols = cols or [c for c, _, _ in FEATURE_LABELS]
    head = "".join(f'<th class="c">{ui.esc(n)}</th>' for n, _ in cols_vals)
    rows = []
    for c in cols:
        grp = next(g for k, _, g in FEATURE_LABELS if k == c).split(" ")[0]
        vals = "".join(f'<td class="c"><b>{float(f[c]):.2f}</b></td>' for _, f in cols_vals)
        rows.append(f"<tr><td>{grp} · {ui.esc(LABEL[c])}</td>{vals}"
                    f'<td class="c">{fl.loc[1, c]:.2f}</td><td class="c">{fl.loc[4, c]:.2f}</td></tr>')
    return (f'<table class="t"><tr><th>Feature</th>{head}<th class="c">Typical level 1</th>'
            f'<th class="c">Typical level 4</th></tr>{"".join(rows)}</table>')


def analyze_view(text):
    if not text or not TAMIL_CHAR.search(text):
        return ui.wrap("<p>Please type a Tamil sentence.</p>"), "", None, None
    _wait()
    f, n = text_features(text)
    sc = text_scores(f)
    lw, lm = nearest_level("word", sc["word"]), nearest_level("morph", sc["morph"])
    top = ui.wrap(f"""
<div class="grid g2">
 <div class="card blue"><span class="tag">Word metric · group A</span>
  <div class="big" style="margin-top:6px">reads like level {lw}</div><p class="muted">score {sc['word']:+.2f}</p>{ui.levels(lw)}</div>
 <div class="card orange"><span class="tag orange">Morph metric · groups A + B</span>
  <div class="big orange" style="margin-top:6px">reads like level {lm}</div><p class="muted">score {sc['morph']:+.2f}</p>{ui.levels(lm, "orange")}</div>
</div>
<p class="muted">Step 4 · Result. StoryWeaver level 1 = easiest, 4 = hardest. {"Text has " + str(n) + " sentences; features are averaged." if n > 1 else ""}</p>""")
    wt = word_table(text)
    cards = ui.wrap("<h3>Step 2 · What the tools found in every word</h3>" + ui.word_cards(wt)
                    + "<h3>Step 3 · The 12 sentence features, next to typical easy and hard sentences</h3>"
                    + _feature_compare_table([("This sentence", f)]))
    return top, cards, wt, feature_proof(f)


def compare_two_view(sentence_a, sentence_b):
    if not (sentence_a and sentence_b and TAMIL_CHAR.search(sentence_a) and TAMIL_CHAR.search(sentence_b)):
        return ui.wrap("<p>Please enter both Tamil sentences.</p>")
    _wait()
    fa, _ = text_features(sentence_a)
    fb, _ = text_features(sentence_b)
    sa, sb = text_scores(fa), text_scores(fb)
    p_a = 1 / (1 + np.exp(-(sa["word"] - sb["word"])))
    harder, conf = ("A", p_a) if p_a >= 0.5 else ("B", 1 - p_a)
    agrees = (sa["morph"] > sb["morph"]) == (harder == "A")

    def card(name, s):
        lv = nearest_level("word", s["word"])
        cls = "orange" if name == harder else "blue"
        return (f'<div class="card {cls}"><span class="tag {cls}">Sentence {name}'
                f'{" · harder" if name == harder else ""}</span>'
                f'<p style="margin-top:6px">word score <b>{s["word"]:+.2f}</b> · morph score <b>{s["morph"]:+.2f}</b></p>'
                f'{ui.levels(lv, cls if cls == "orange" else "")}</div>')
    return ui.wrap(f"""
<div class="verdict orange">Sentence {harder} is harder ({conf * 100:.1f}% confident) ·
 morph metric {"agrees ✓" if agrees else "disagrees ✗"}</div>
<div class="grid g2">{card("A", sa)}{card("B", sb)}</div>
<h3>Why: the features of both sentences</h3>
{_feature_compare_table([("A", fa), ("B", fb)])}
<p class="muted">Confidence = sigmoid(score A − score B) of the word metric, the same pair model as the thesis.</p>""")


def compare_view(original, simplified):
    if not (original and simplified and TAMIL_CHAR.search(original) and TAMIL_CHAR.search(simplified)):
        return ui.wrap("<p>Please fill in both boxes with Tamil text.</p>")
    _wait()
    fo, _ = text_features(original)
    fs, _ = text_features(simplified)
    so, ss = text_scores(fo), text_scores(fs)
    cards = []
    for m, name, cls in [("word", "Word metric", "blue"), ("morph", "Morph metric", "orange")]:
        d = ss[m] - so[m]
        ok = d < 0
        cards.append(
            f'<div class="card {"green" if ok else ""}"><span class="tag {"green" if ok else "grey"}">{name}</span>'
            f'<div class="big {"orange" if cls == "orange" else ""}" style="margin-top:6px">'
            f'level {nearest_level(m, so[m])} → {nearest_level(m, ss[m])}</div>'
            f'<p>score {so[m]:+.2f} → {ss[m]:+.2f} ({d:+.2f}) · <b>{"✓ simplified is easier" if ok else "✗ not easier"}</b></p></div>')
    return ui.wrap(f"""
<div class="grid g2">{"".join(cards)}</div>
<h3>What changed in the features</h3>
{_feature_compare_table([("Original", fo), ("Simplified", fs)])}""")


def load_pair_view(label):
    o, s, md = load_pair(label)
    if not label or label not in PAIRS:
        return o, s, ""
    p = PAIRS[label]
    return o, s, ui.wrap(f"""
<div class="card green"><span class="tag green">What {p['n']} native readers said about this pair</span>
<div class="grid g4" style="margin-top:8px">
 <div><div class="big" style="color:#1E7B45">{p['diff_orig']:.2f} → {p['diff_simp']:.2f}</div><p class="muted">difficulty (1 easy – 5 hard)</p></div>
 <div><div class="big" style="color:#1E7B45">{p['picked_simp']:.0%}</div><p class="muted">picked the simplified one as easier</p></div>
 <div><div class="big" style="color:#1E7B45">{p['meaning']:.2f}</div><p class="muted">meaning kept (1–5)</p></div>
 <div><div class="big" style="color:#1E7B45">{p['err_simp']:.0%}</div><p class="muted">saw an error in the simplified one</p></div>
</div></div>""")

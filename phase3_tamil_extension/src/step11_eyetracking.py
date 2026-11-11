"""
Step 11 (Phase C, part 2): the Tamil webcam eye-tracking pilot.

Question: when Tamil readers actually read, do word-level features or
morphology features explain how long they look at words and sentences?
This is the Tamil version of Phase 1's System 1 (Singh et al. 2016), run on
the data collected with Tool/TamilReadingWeb (WebGazer, 100 Dinamani news
sentences from UD Tamil-TTB, ENTER after each sentence).

Input : Tamil Reading Study data.xlsx   (the Google Sheet, downloaded)
          words      one row per reader x word: FFD, FPD, RPD, TD (Dundee rules)
          sentences  one row per reader x sentence: reading time, probe answer, gaze quality
Output: output/eyetracking_word_data.csv       word measures (averaged over readers) + features
        output/eyetracking_sentence_data.csv   sentence reading time + both metric scores
        results/eyetracking_summary.md         all tables

Steps
  1. Readers: keep real participants (not PILOT/DIAG test sessions) who answered
     >= 80% of comprehension probes correctly (same rule as motr_study).
  2. Word measures: for each word, sum over readers whose gaze on that sentence
     passed the quality check, divided by the number of such readers. Skipped words
     count as 0 -- exactly Phase 1's "word-level average" approach.
  3. Word features (same tools as step 3, but per word instead of per sentence):
       controls  position in sentence, first/last word on its line, sentence length
       A word    word length (aksharas), word frequency (Zipf), GPT-2 surprisal
       B morph   morphemes, stem surprisal, suffix surprisal, marked UD features,
                 lemma frequency, non-finite verb
  4. System 1: linear regression, 10-fold cross-validation grouped by sentence
     (a sentence's words are never in train and test at once). R^2 on held-out
     words; 2,000 sentence-bootstrap draws for the CI of (A+B) - A.
  5. Sentence level: reading time (ENTER press, so valid even when gaze is poor),
     log-transformed, outliers > 3 SD removed, z-scored per reader, averaged.
     Spearman with the step-8 metrics (word, morph) and the bootstrap difference.

Run: C:\\Python313\\python.exe src/step11_eyetracking.py ["path/to/Tamil Reading Study data.xlsx"]
     (~3 min, CPU). Use the system Python, like steps 3/5: its Stanza 1.14 matches the
     downloaded Tamil models; the project .venv has Stanza 1.15, which refuses them.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step3_textbook_features import (IGNORED, NONFINITE, TAMIL, UNMARKED,  # noqa: E402
                                     char_surprisal, load_tools, sentence_features)
from metric_model import fit_metrics, score  # noqa: E402

BASE = os.path.dirname(HERE)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
# --with=R03  : extra check -- also keep these readers even if they fail the probe rule
#               (reported separately; the main analysis keeps the 80% rule).
EXTRA = {p for a in sys.argv[1:] if a.startswith("--with=") for p in a[7:].split(",") if p}
SUFFIX = ("_with_" + "_".join(sorted(EXTRA))) if EXTRA else ""
XLSX = ARGS[0] if ARGS else os.path.join(BASE, "Tamil Reading Study data.xlsx")
STIMULI = os.path.join(BASE, "motr_study", "stimuli", "sentences.csv")
OUT_WORDS = os.path.join(BASE, "output", f"eyetracking_word_data{SUFFIX}.csv")
OUT_SENTS = os.path.join(BASE, "output", f"eyetracking_sentence_data{SUFFIX}.csv")
REPORT = os.path.join(BASE, "results", f"eyetracking_summary{SUFFIX}.md")

TEST_IDS = {"PILOT", "DIAG", "CHECK"}
MIN_PROBE_ACC = 0.80
# Two different people typed the code R02: the session started on 2026-11-11 is a new
# reader (confirmed by the experimenter) and is renamed R05.
RENAME_FROM = {"R02": ("2026-11-11", "R05")}
# Same person, second session (confirmed by the experimenter): 123 is reader 1111 again.
# Only first readings count, so 123's sentences that 1111 already read are dropped;
# his 50 new sentences are kept as a normal reader.
SEEN_BEFORE = {"123": "1111"}
REPEATS = {}
MEASURES = ["FFD", "FPD", "RPD", "TD"]
N_BOOT = 2000
SEED = 42

CONTROLS = ["position", "first_on_line", "last_on_line", "sent_len"]
A = ["wlen", "word_zipf", "word_surprisal"]
B = ["n_morphs", "stem_surprisal", "suffix_surprisal", "n_feats", "lemma_zipf", "nonfinite"]
MODELS = {"controls only": CONTROLS,
          "A: word-level": CONTROLS + A,
          "B: morphology": CONTROLS + B,
          "A + B": CONTROLS + A + B}


# ---------------------------------------------------------------- 1. readers
def pick_readers(sents):
    s = sents[(sents.practice == 0) & ~sents.pid.astype(str).isin(TEST_IDS)]
    q = s.groupby("pid").agg(sentences=("sid", "size"),
                             gaze_ok=("sentence_ok", "mean"),
                             probe_acc=("correct", "mean"),
                             probes=("correct", "count"),
                             median_rt_s=("reading_time_ms", lambda x: x.median() / 1000))
    q["kept"] = (q.probe_acc >= MIN_PROBE_ACC) & ~q.index.isin(list(REPEATS))
    q["why"] = ["no: " + REPEATS[p] if p in REPEATS
                else "yes" if a >= MIN_PROBE_ACC
                else "yes, extra check only (probe accuracy < 80%)" if p in EXTRA
                else "no (probe accuracy < 80%)" for p, a in zip(q.index, q.probe_acc)]
    q.loc[q.index.isin(list(EXTRA)), "kept"] = True
    return q


def split_sessions(df):
    for pid, (start, new) in RENAME_FROM.items():
        df.loc[(df.pid == pid) & (df.ts.astype(str) >= start), "pid"] = new
    return df


def drop_second_readings(df, sents):
    """Remove sentences a repeat reader had already read in an earlier session."""
    for pid, earlier in SEEN_BEFORE.items():
        seen = set(sents[(sents.pid == earlier) & (sents.practice == 0)].sid)
        df = df[~((df.pid == pid) & df.sid.isin(seen))]
    return df


# ---------------------------------------------------------------- 2. word measures
def word_measures(words, readers):
    w = words[words.pid.isin(readers) & (words.SENTENCE_OK == 1)]
    per_word = w.groupby(["sid", "WNUM"])
    m = per_word[MEASURES].mean()                      # sum / n readers with good gaze (skips = 0)
    m["n_readers"] = per_word.size()
    m["skip_rate"] = 1 - per_word.FIXATED.mean()
    return m.reset_index()


def reader_reliability(words, readers):
    """Spearman between each pair of readers on words both read with good gaze."""
    w = words[words.pid.isin(readers) & (words.SENTENCE_OK == 1)]
    rows = []
    for meas in MEASURES:
        wide = w.pivot_table(index=["sid", "WNUM"], columns="pid", values=meas)
        rs = []
        for i, a in enumerate(readers):
            for b in readers[i + 1:]:
                both = wide[[a, b]].dropna()
                rs.append((spearmanr(both[a], both[b])[0], len(both)))
        rows.append({"measure": meas, "rho": np.mean([r for r, _ in rs]),
                     "words": int(np.mean([n for _, n in rs]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. word features
def word_features(stim, seg, tok, lm):
    import stanza
    from wordfreq import zipf_frequency
    nlp = stanza.Pipeline("ta", processors="tokenize,mwt,pos,lemma", tokenize_pretokenized=True,
                          verbose=False, download_method=None)
    rows = []
    for _, s in stim.iterrows():
        text = s.text
        per_char = char_surprisal(text, tok, lm)
        tokens = text.split()
        cores = [t.rstrip(".,") for t in tokens]
        doc = nlp([cores + ["."]])                     # one pretokenized sentence, 1 token per word
        pos = 0
        for i, (t, core) in enumerate(zip(tokens, cores)):
            start = text.index(t, pos); pos = start + len(t)
            end = start + len(core)
            morphs = seg.morph_analyze(core) if TAMIL.fullmatch(core) else [core]
            if "".join(morphs) != core:
                morphs = [core]
            stem_end = start + len(morphs[0])
            ud = doc.sentences[0].tokens[i].words
            feats = [f for u in ud for f in ((u.feats or "").split("|") if u.feats else [])]
            lemmas = [u.lemma for u in ud if u.lemma and TAMIL.search(u.lemma)]
            rows.append({
                "sid": s.sid, "WNUM_in_sent": i + 1, "word": t,
                "word_zipf": zipf_frequency(core, "ta"),
                "word_surprisal": per_char[start:end].sum(),
                "n_morphs": len(morphs),
                "stem_surprisal": per_char[start:stem_end].sum(),
                "suffix_surprisal": per_char[stem_end:end].sum(),
                "n_feats": sum(1 for f in feats if f not in UNMARKED and not f.startswith(IGNORED)),
                "lemma_zipf": np.mean([zipf_frequency(l, "ta") for l in lemmas]) if lemmas else 0.0,
                "nonfinite": int(bool(NONFINITE & set(feats))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. System 1
def cv_predictions(df, cols, y):
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=10).split(df, groups=df.sid):
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(df.iloc[tr][cols], df.iloc[tr][y])
        pred[te] = m.predict(df.iloc[te][cols])
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def system1(df, rng):
    sids = df.sid.unique()
    idx = {s: np.flatnonzero(df.sid.to_numpy() == s) for s in sids}
    draws = [np.concatenate([idx[s] for s in rng.choice(sids, len(sids))]) for _ in range(N_BOOT)]
    rows, gains = [], []
    for meas in MEASURES:
        y = df[meas].to_numpy()
        preds = {name: cv_predictions(df, cols, meas) for name, cols in MODELS.items()}
        for name, p in preds.items():
            boot = [r2(y[d], p[d]) for d in draws]
            rows.append({"measure": meas, "model": name, "R2": r2(y, p),
                         "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
        for new, old in [("A + B", "A: word-level"), ("A: word-level", "controls only"),
                         ("B: morphology", "controls only")]:
            d = [r2(y[i], preds[new][i]) - r2(y[i], preds[old][i]) for i in draws]
            gains.append({"measure": meas, "comparison": f"{new} vs {old}",
                          "gain": r2(y, preds[new]) - r2(y, preds[old]),
                          "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
                          "p": float(np.mean(np.array(d) <= 0))})
    return pd.DataFrame(rows), pd.DataFrame(gains)


def full_fit_coefs(df, meas="TD"):
    cols = MODELS["A + B"]
    m = make_pipeline(StandardScaler(), LinearRegression()).fit(df[cols], df[meas])
    out = pd.DataFrame({"feature": cols, "std_coef_ms": m[-1].coef_})
    out["spearman_with_" + meas] = [spearmanr(df[c], df[meas])[0] for c in cols]
    return out


# ---------------------------------------------------------------- 5. sentence level
def sentence_rt(sents, readers):
    s = sents[(sents.practice == 0) & sents.pid.isin(readers)].copy()
    s["log_rt"] = np.log(s.reading_time_ms)
    s["log_rt_word"] = np.log(s.reading_time_ms / s.n_words)
    keep = []
    for _, g in s.groupby("pid"):
        z = (g.log_rt - g.log_rt.mean()) / g.log_rt.std()
        keep.append(g[z.abs() <= 3])
    s = pd.concat(keep)
    dropped = int(((sents.practice == 0) & sents.pid.isin(readers)).sum() - len(s))
    for c in ["log_rt", "log_rt_word"]:
        s[c + "_z"] = s.groupby("pid")[c].transform(lambda x: (x - x.mean()) / x.std())
    out = s.groupby("sid").agg(rt_z=("log_rt_z", "mean"), rt_per_word_z=("log_rt_word_z", "mean"),
                               mean_rt_s=("reading_time_ms", lambda x: x.mean() / 1000),
                               n_readers=("pid", "size")).reset_index()
    return out, dropped


def boot_spearman(x, y, rng, other=None):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    b, d = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        b.append(spearmanr(x[i], y[i])[0])
        if other is not None:
            d.append(b[-1] - spearmanr(np.asarray(other)[i], y[i])[0])
    res = (spearmanr(x, y)[0], np.percentile(b, 2.5), np.percentile(b, 97.5))
    if other is not None:
        d = np.array(d)
        res += (spearmanr(x, y)[0] - spearmanr(other, y)[0],
                np.percentile(d, 2.5), np.percentile(d, 97.5), float(np.mean(d <= 0)))
    return res


# ---------------------------------------------------------------- report
def fmt(x, nd=3):
    return f"{x:+.{nd}f}"


def main():
    rng = np.random.default_rng(SEED)
    xl = pd.read_excel(XLSX, sheet_name=None)
    words, sents = xl["words"], xl["sentences"]
    val = xl["validation"]
    for d in (words, sents, val):
        d["pid"] = d.pid.astype(str)
        split_sessions(d)
    words = drop_second_readings(words, sents)
    sents = drop_second_readings(sents, sents)
    stim = pd.read_csv(STIMULI, encoding="utf-8-sig")

    q = pick_readers(sents)
    readers = sorted(q.index[q.kept])
    print("readers kept:", readers)

    # word level
    meas = word_measures(words, readers)
    # word layout from the reader with the most sentences (lines differ slightly by screen size)
    first = words[words.pid.isin(readers)].groupby("pid").sid.nunique().idxmax()
    layout = (words[words.pid == first][["sid", "WNUM", "LINE", "LINE_POS", "WLEN", "WORD"]]
              .drop_duplicates(["sid", "WNUM"]))
    layout["position"] = layout.groupby("sid").WNUM.rank().astype(int)
    layout["sent_len"] = layout.groupby("sid").WNUM.transform("size")
    layout["first_on_line"] = (layout.LINE_POS == 1).astype(int)
    last = layout.groupby(["sid", "LINE"]).LINE_POS.transform("max")
    layout["last_on_line"] = (layout.LINE_POS == last).astype(int)
    layout = layout.rename(columns={"WLEN": "wlen"})

    print("computing word + sentence features (Stanza, Morfessor, Tamil GPT-2) ...", flush=True)
    seg, nlp_sent, tok, lm = load_tools()
    wf = word_features(stim, seg, tok, lm)
    df = layout.merge(wf, left_on=["sid", "position"], right_on=["sid", "WNUM_in_sent"])
    assert (df.WORD == df.word).all(), "word alignment between sheet and stimuli failed"
    df = df.merge(meas, on=["sid", "WNUM"]).drop(columns=["WNUM_in_sent", "word"])
    df.to_csv(OUT_WORDS, index=False, encoding="utf-8")
    print(f"wrote {OUT_WORDS}  ({len(df)} words, {df.sid.nunique()} sentences)")

    rel = reader_reliability(words, readers)
    s1, gains = system1(df, rng)
    coefs = full_fit_coefs(df)

    # sentence level
    srt, dropped = sentence_rt(sents, readers)
    sf = pd.DataFrame([sentence_features(t, seg, nlp_sent, tok, lm) for t in stim.text])
    sf = pd.concat([stim[["sid", "band", "text"]], sf], axis=1)
    sf = pd.concat([sf, score(sf, fit_metrics())], axis=1)
    sd = sf.merge(srt, on="sid")
    sd.to_csv(OUT_SENTS, index=False, encoding="utf-8")
    sent_rows = []
    for target, label in [("rt_z", "sentence reading time"), ("rt_per_word_z", "reading time per word")]:
        w_ = boot_spearman(sd.score_word, sd[target], rng)
        m_ = boot_spearman(sd.score_morph, sd[target], rng, other=sd.score_word)
        l_ = boot_spearman(sd.n_words, sd[target], rng)
        sent_rows.append((label, w_, m_, l_))
    sent_rel = None
    s2 = sents[(sents.practice == 0) & sents.pid.isin(readers)].pivot_table(
        index="sid", columns="pid", values="reading_time_ms")
    pairs = [s2[[a, b]].dropna() for i, a in enumerate(readers) for b in readers[i + 1:]]
    if pairs:   # mean Spearman over all reader pairs
        sent_rel = (np.mean([spearmanr(p.iloc[:, 0], p.iloc[:, 1])[0] for p in pairs]),
                    int(np.mean([len(p) for p in pairs])))
    bands = sd.groupby("band")[["mean_rt_s", "rt_z", "rt_per_word_z"]].mean().reindex(["easy", "medium", "hard"])

    # ------------------------------------------------ write report
    v = val[val.pid.astype(str).isin(readers)]
    import json
    kept_s = sents[(sents.practice == 0) & sents.pid.isin(readers)]
    widths = [b[2] - b[0] for x in kept_s.boxes for b in json.loads(x)]
    w25, w50, w75 = np.percentile(widths, [25, 50, 75])
    L = ["# Phase C, part 2: Tamil webcam eye-tracking pilot", "",
         "Data: `Tamil Reading Study data.xlsx` (Tool/TamilReadingWeb, WebGazer webcam tracking, "
         "100 Dinamani news sentences from UD Tamil-TTB). Script: `src/step11_eyetracking.py`.", "",
         "## 1. Readers", "",
         "| Reader | Sentences | Gaze OK | Probe accuracy | Median reading time (s) | Kept |",
         "|---|---|---|---|---|---|"]
    for pid, r in q.iterrows():
        L.append(f"| {pid} | {r.sentences} | {r.gaze_ok:.0%} | {r.probe_acc:.0%} ({r.probes}) | "
                 f"{r.median_rt_s:.1f} | {r.why} |")
    L += ["", f"Test sessions ({', '.join(sorted(TEST_IDS))}) are not counted as readers. "
          "Two people typed the code R02; the session started on 2026-11-11 is a different reader "
          "and is listed as R05. Reader 123 is reader 1111 in a second session: only his 50 "
          "sentences not already read as 1111 are used (first readings only). "
          + (f"EXTRA CHECK: {', '.join(sorted(EXTRA))} kept despite failing the 80% probe rule; "
             "the main analysis is `eyetracking_summary.md`. " if EXTRA else "")
          + "Kept: " + ", ".join(readers) + ". "
          f"Calibration check (median gaze error, 9 validation dots) for kept readers: "
          f"{', '.join(f'{p}: {int(v[v.pid.astype(str) == p].median_error_px.median())} px' for p in readers)} "
          f"(median over all checks). The words on screen were {w50:.0f} px wide (median; middle half "
          f"{w25:.0f}-{w75:.0f} px), so the gaze error is about one word: gaze is often credited to a "
          "neighbouring word, and word-level measures are noisy by design of webcam tracking.", "",
         "## 2. How consistent are the readers? (ceiling for any model)", "",
         "| Measure | Spearman between readers | Words both read with good gaze |", "|---|---|---|"]
    for _, r in rel.iterrows():
        L.append(f"| {r.measure} | {fmt(r.rho)} | {r.words} |")
    if sent_rel:
        L += ["", f"Sentence reading time (ENTER press): Spearman between readers {fmt(sent_rel[0])} "
              f"over {sent_rel[1]} sentences."]
    L += ["", "## 3. System 1: predicting word reading times", "",
          f"{len(df)} words in {df.sid.nunique()} sentences; each word's measure is averaged over the "
          f"readers with good gaze on that sentence (skipped = 0 ms, as in Phase 1). Mean readers per "
          f"word: {df.n_readers.mean():.2f}; skip rate {df.skip_rate.mean():.0%}. Linear regression, "
          "10-fold cross-validation grouped by sentence; R² on held-out words, 95% CI from "
          f"{N_BOOT} sentence-bootstrap draws.", "",
          "| Measure | Controls only | A: word-level | B: morphology | A + B |", "|---|---|---|---|---|"]
    for meas_ in MEASURES:
        cells = []
        for name in MODELS:
            r = s1[(s1.measure == meas_) & (s1.model == name)].iloc[0]
            cells.append(f"{r.R2:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
        L.append(f"| {meas_} | " + " | ".join(cells) + " |")
    L += ["", "Controls = position in sentence, first/last word on its line, sentence length. "
          "A = word length (aksharas), word frequency, GPT-2 surprisal. "
          "B = morphemes, stem/suffix surprisal, marked UD features, lemma frequency, non-finite verb.", "",
          "| Measure | Comparison | R² gain | 95% CI | p (one-sided, no gain) |", "|---|---|---|---|---|"]
    for _, r in gains.iterrows():
        L.append(f"| {r.measure} | {r.comparison} | {fmt(r.gain)} | [{fmt(r.lo)}, {fmt(r.hi)}] | {r.p:.3f} |")
    L += ["", "Full-data fit of A + B on TD (standardised coefficients, ms per SD) and the plain "
          "Spearman correlation of each feature with TD:", "",
          "| Feature | Coefficient (ms/SD) | Spearman with TD |", "|---|---|---|"]
    for _, r in coefs.iterrows():
        L.append(f"| {r.feature} | {r.std_coef_ms:+.1f} | {fmt(r.spearman_with_TD)} |")
    L += ["", "## 4. Sentence level: do the two metrics track reading time?", "",
          f"Reading time = time from sentence onset to ENTER (log, outliers > 3 SD per reader removed: "
          f"{dropped}; z-scored per reader; averaged). Metrics are the step-8 sentence scores "
          "(trained on StoryWeaver, never on this data). 95% CIs from sentence bootstrap.", "",
          "| Target | word metric | morph metric | morph - word | p (morph not better) | n_words alone |",
          "|---|---|---|---|---|---|"]
    for label, w_, m_, l_ in sent_rows:
        L.append(f"| {label} | {fmt(w_[0])} [{fmt(w_[1])}, {fmt(w_[2])}] | "
                 f"{fmt(m_[0])} [{fmt(m_[1])}, {fmt(m_[2])}] | {fmt(m_[3])} [{fmt(m_[4])}, {fmt(m_[5])}] | "
                 f"{m_[6]:.3f} | {fmt(l_[0])} [{fmt(l_[1])}, {fmt(l_[2])}] |")
    L += ["", "Reading time by complexity band (bands set before the study from words, dependency "
          "distance, tree depth and word length; `motr_study/select_sentences.py`):", "",
          "| Band | Mean reading time (s) | Reading time z | Per-word reading time z |", "|---|---|---|---|"]
    for b, r in bands.iterrows():
        L.append(f"| {b} | {r.mean_rt_s:.1f} | {fmt(r.rt_z, 2)} | {fmt(r.rt_per_word_z, 2)} |")
    L += ["", "## Reading the result", "",
          "- Section 2 is the ceiling: if two readers barely agree on a word's reading time, no feature "
          "model can predict it well.",
          "- Section 3, `A + B vs A`: a CI above 0 would mean morphology adds to the word-level baseline "
          "for real reading times. A CI that straddles 0 means no evidence it does.",
          "- Section 4 is the same question at the sentence level, using the metrics from Phase B/C.",
          "- With this few readers everything here is a pilot: report it as feasibility evidence, not "
          "as a final answer."]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()

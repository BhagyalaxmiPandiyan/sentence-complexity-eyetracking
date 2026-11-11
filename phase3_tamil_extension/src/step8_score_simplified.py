"""
Step 8 (Phase B -> C): read the LLM simplifications, score original vs simplified
with both metrics, and build the native-speaker rating form.

Input : Dataset/complex_sentences/original.csv               (step 7)
        Dataset/complex_sentences/simplified_<system>.txt     one file per LLM, the chat
                                                              replies pasted as-is ("S01 | ...")
Output: output/simplification_scores.csv     one row per (sentence, system): both texts,
                                             all 12 features before/after, both metric scores
        results/simplification_summary.md    did each metric see the simplification as easier?
        Dataset/human_study/items_key.csv    which form item is which sentence / system,
                                             and whether A or B is the original
        Dataset/human_study/make_google_form.gs   Google Apps Script that creates the forms

An LLM may split one sentence into several. Features are then computed per
sentence and averaged, i.e. the metric scores the *typical sentence* of the
simplified text (otherwise splitting would look harder because n_words grows).

Run: python src/step8_score_simplified.py         (CPU, ~2 min per 60 sentences)
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_model import A, B, fit_metrics, score                  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS = os.path.join(BASE, "Dataset", "complex_sentences")
HS = os.path.join(BASE, "Dataset", "human_study")
OUT = os.path.join(BASE, "output", "simplification_scores.csv")
SUMMARY = os.path.join(BASE, "results", "simplification_summary.md")

LINE = re.compile(r"^[\s*#>-]*(S\d{2})[\s*]*[|:\-–]\s*(.+?)\s*$")   # tolerates markdown bold/bullets
SENT_END = re.compile(r"(?<=[.?!])\s+")
TAMIL_CHAR = re.compile("[஀-௿]")
ITEMS_PER_FORM = 30
N_BOOT = 2000
SEED = 0
FEATS = A + B + ["tree_depth", "dep_distance"]


# ---------------------------------------------------------------- read replies
def read_system(path, ids):
    got = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = LINE.match(line.strip())
            if m and m.group(1) in ids:
                got[m.group(1)] = m.group(2)
    missing = sorted(set(ids) - set(got))
    if missing:
        print(f"  WARNING {os.path.basename(path)}: no line for {', '.join(missing)}")
    return got


def text_features(text, tools, cache):
    """Average the 12 features over the sentences of text."""
    from step3_textbook_features import sentence_features
    parts = [p for p in SENT_END.split(text.strip()) if TAMIL_CHAR.search(p)]
    rows = []
    for p in parts:
        if p not in cache:
            cache[p] = sentence_features(p, *tools)
        rows.append(cache[p])
    f = pd.DataFrame(rows).mean().to_dict()
    f["n_sentences"] = len(parts)
    return f


# ---------------------------------------------------------------- summary
def boot_ci(x, rng):
    m = rng.choice(x, size=(N_BOOT, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def summarize(df):
    rng = np.random.default_rng(SEED)
    L = ["# Phase B: do the metrics see the LLM simplifications as easier?", "",
         "Score = metric difficulty (higher = harder, see src/metric_model.py). "
         "drop = score(original) - score(simplified); > 0 means the metric agrees the "
         "simplification is easier. 95% CI = bootstrap over sentences.", ""]
    for system, d in df.groupby("system"):
        L += [f"## System: {system}  ({len(d)} sentences)", "",
              "| Metric | Simplified judged easier | Mean drop | 95% CI |", "|---|---|---|---|"]
        for m in ["word", "morph"]:
            drop = (d[f"score_{m}_orig"] - d[f"score_{m}_simp"]).to_numpy()
            ci = boot_ci(drop, rng)
            L.append(f"| {m} | {np.mean(drop > 0):.0%} | {drop.mean():+.3f} | "
                     f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        L += ["", "Feature means, original -> simplified (per sentence):", "",
              "| Feature | Original | Simplified | Change |", "|---|---|---|---|"]
        for c in FEATS:
            o, s = d[f"{c}_orig"].mean(), d[f"{c}_simp"].mean()
            L.append(f"| {c} | {o:.2f} | {s:.2f} | {s - o:+.2f} |")
        same = (d.original.str.strip() == d.simplified.str.strip()).sum()
        L += ["", f"Sentences split into several: {(d.n_sentences_simp > 1).sum()}; "
              f"unchanged copies: {same}; mean sentences per output: {d.n_sentences_simp.mean():.2f}.",
              ""]
    L += ["Caveat: the metrics only say whether the text *looks* simpler. Whether it is "
          "easier for readers, keeps the meaning and is correct Tamil is what the human "
          "study (step 9) measures."]
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- human study form
INTRO = ("ஒவ்வொரு கேள்வியிலும் A, B என்று இரண்டு தமிழ் வாக்கியங்கள் உள்ளன. இரண்டையும் "
         "கவனமாகப் படித்து கேள்விகளுக்குப் பதில் அளியுங்கள். சரி / தவறு என்று எதுவும் இல்லை; "
         "உங்கள் உணர்வே முக்கியம்.\n\n"
         "Each item shows two Tamil sentences, A and B. Read both carefully and answer. "
         "There are no right or wrong answers; we want your honest impression. "
         "Imagine a child in Class 3-5 reading them. About 20 minutes.")


def build_form(df):
    rng = np.random.default_rng(SEED)
    items = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    rows, forms = [], []
    for start in range(0, len(items), ITEMS_PER_FORM):
        form_no = start // ITEMS_PER_FORM + 1
        qs = []
        for k, r in enumerate(items.iloc[start:start + ITEMS_PER_FORM].itertuples(), 1):
            a_is_orig = bool(rng.random() < 0.5)
            a, b = (r.original, r.simplified) if a_is_orig else (r.simplified, r.original)
            rows.append({"form": form_no, "item": k, "id": r.id, "system": r.system,
                         "A_is": "original" if a_is_orig else "simplified"})
            qs.append({"k": k, "a": a, "b": b})
        forms.append({"no": form_no, "items": qs})
    os.makedirs(HS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(HS, "items_key.csv"), index=False, encoding="utf-8")

    gs = GS_TEMPLATE.replace("__FORMS__", json.dumps(forms, ensure_ascii=False, indent=1)) \
                    .replace("__INTRO__", json.dumps(INTRO, ensure_ascii=False))
    with open(os.path.join(HS, "make_google_form.gs"), "w", encoding="utf-8") as fh:
        fh.write(gs)
    print(f"human study: {len(rows)} items in {len(forms)} form(s) -> {HS}")


GS_TEMPLATE = r"""// Creates the Tamil simplification rating form(s) in your Google Drive.
// Generated by src/step8_score_simplified.py -- do not edit the items by hand
// (items_key.csv must match them).
// How: script.google.com -> New project -> paste this whole file -> Run 'makeForms'
//      -> allow access -> the form links appear under 'Execution log'.
// IMPORTANT: question titles start with "<item>." -- step 9 reads answers by these titles.
var FORMS = __FORMS__;
var INTRO = __INTRO__;

function makeForms() {
  FORMS.forEach(function (f) {
    var form = FormApp.create('Tamil sentence difficulty study - Form ' + f.no);
    form.setDescription(INTRO);
    form.setProgressBar(true);
    form.addTextItem().setTitle('Your name or initials').setRequired(true);
    form.addMultipleChoiceItem().setTitle('How well do you read Tamil?')
        .setChoiceValues(['Native speaker', 'Fluent (learned later)', 'Basic']).setRequired(true);
    f.items.forEach(function (it) {
      form.addPageBreakItem().setTitle('Item ' + it.k + ' / ' + f.items.length);
      form.addSectionHeaderItem().setTitle('A:  ' + it.a);
      form.addSectionHeaderItem().setTitle('B:  ' + it.b);
      form.addMultipleChoiceItem().setTitle(it.k + '. Easier: which sentence is easier to read? / எது எளிது?')
          .setChoiceValues(['A', 'B', 'Same']).setRequired(true);
      form.addScaleItem().setTitle(it.k + '. Difficulty A / A-வின் கடினம்')
          .setBounds(1, 5).setLabels('Very easy', 'Very hard').setRequired(true);
      form.addScaleItem().setTitle(it.k + '. Difficulty B / B-யின் கடினம்')
          .setBounds(1, 5).setLabels('Very easy', 'Very hard').setRequired(true);
      form.addScaleItem().setTitle(it.k + '. Meaning: do A and B say the same thing? / ஒரே பொருளா?')
          .setBounds(1, 5).setLabels('Completely different', 'Exactly the same').setRequired(true);
      form.addMultipleChoiceItem().setTitle(it.k + '. Errors: which has Tamil grammar or word mistakes? / பிழைகள்?')
          .setChoiceValues(['Neither', 'A', 'B', 'Both']).setRequired(true);
    });
    Logger.log('Form ' + f.no + '  share: ' + form.getPublishedUrl() + '   edit: ' + form.getEditUrl());
  });
}
"""


# ---------------------------------------------------------------- main
def main():
    orig = pd.read_csv(os.path.join(CS, "original.csv"))
    files = sorted(glob.glob(os.path.join(CS, "simplified_*.txt")))
    if not files:
        sys.exit(f"No simplified_<system>.txt files in {CS}. See HOW_TO_SIMPLIFY.md.")

    from step3_textbook_features import load_tools
    tools, cache = load_tools(), {}
    metrics = fit_metrics()
    rows = []
    for path in files:
        system = os.path.basename(path)[len("simplified_"):-len(".txt")]
        simp = read_system(path, set(orig.id))
        print(f"{system}: {len(simp)} simplifications")
        for r in orig.itertuples():
            if r.id not in simp:
                continue
            fo = text_features(r.sentence, tools, cache)
            fs = text_features(simp[r.id], tools, cache)
            row = {"id": r.id, "source": r.source, "level": r.level, "system": system,
                   "original": r.sentence, "simplified": simp[r.id]}
            row.update({f"{k}_orig": v for k, v in fo.items()})
            row.update({f"{k}_simp": v for k, v in fs.items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    for side in ["orig", "simp"]:
        X = df[[f"{c}_{side}" for c in FEATS]].set_axis(FEATS, axis=1)
        for name, col in score(X, metrics).items():
            df[f"{name}_{side}"] = col
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}")
    summarize(df)
    build_form(df)


if __name__ == "__main__":
    main()

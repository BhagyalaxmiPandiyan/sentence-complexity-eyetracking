"""
Pick the 100 Tamil newspaper sentences for the MoTR reading study.

Source: UD Tamil-TTB (tools/UD_Tamil-TTB), 600 sentences from Dinamani
(www.dinamani.com) news pages, CC BY-NC-SA 3.0 (Ramasamy & Zabokrtsky, LREC 2012).

Filters (so every sentence reads cleanly on screen):
  - unique text (the treebank has one duplicate sentence)
  - 6..16 words, ends with a single full stop
  - no digits, Latin letters, brackets, quotes, colons or abbreviation dots

Complexity spread: each candidate gets a score = z(words) + z(mean dependency
distance) + z(max tree depth) + z(mean word length). Candidates are cut into
easy / medium / hard thirds and we sample 33 / 33 / 34 from them (seed 42),
so the study covers the whole range like the Dundee texts do.

Also builds:
  - 3 practice sentences (from the leftovers, never in the 100)
  - 20 comprehension probes: "was this word in the sentence?" (10 yes / 10 no)

Output: stimuli/sentences.csv  and  web/stimuli.js
"""
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TTB = HERE.parent / "tools" / "UD_Tamil-TTB"
SEED = 42
N_PER_BAND = {"easy": 33, "medium": 33, "hard": 34}
N_PRACTICE = 3
N_PROBES = 20

BAD_CHARS = re.compile(r"[0-9A-Za-z()\[\]\"'“”‘’:;!?]")
TAMIL_WORD = re.compile("^[஀-௿]+$")


def read_conllu(path):
    sents, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# sent_id = "):
            cur = {"sent_id": line[12:], "rows": []}
        elif line.startswith("# text = "):
            cur["text"] = line[9:].strip()
        elif line.startswith("# orig_file_sentence"):
            cur["orig"] = line.split()[-1]
        elif line and not line.startswith("#"):
            cols = line.split("\t")
            if cols[0].isdigit():  # syntactic words only (skip 11-12 multiword ranges)
                cur["rows"].append((int(cols[0]), int(cols[6]), cols[3]))
        elif not line and cur:
            sents.append(cur)
            cur = None
    if cur:
        sents.append(cur)
    return sents


def tree_stats(rows):
    head = {i: h for i, h, _ in rows}
    dists = [abs(i - h) for i, h, pos in rows if h != 0 and pos != "PUNCT"]

    def depth(i):
        d = 0
        while head.get(i, 0) != 0 and d < 100:
            i, d = head[i], d + 1
        return d

    return (np.mean(dists) if dists else 0.0), max(depth(i) for i in head)


def main():
    sents = []
    for f in sorted(TTB.glob("ta_ttb-ud-*.conllu")):
        sents += read_conllu(f)
    print(f"Treebank sentences: {len(sents)}")

    rows, seen = [], set()
    for s in sents:
        text = s["text"]
        words = text.split()
        if text in seen:
            continue
        seen.add(text)
        if not (6 <= len(words) <= 16) or not text.endswith(".") or text.endswith(".."):
            continue
        if BAD_CHARS.search(text) or "." in text[:-1]:
            continue
        mdd, depth = tree_stats(s["rows"])
        clean = [w.strip(".,") for w in words]
        rows.append({
            "treebank_id": s["sent_id"],
            "dinamani_file": s["orig"].split("#")[0],
            "text": text,
            "n_words": len(words),
            "mean_dep_dist": round(mdd, 3),
            "max_depth": depth,
            "mean_word_len": round(np.mean([len(w) for w in clean]), 3),
        })
    df = pd.DataFrame(rows)
    print(f"Clean, unique candidates: {len(df)}")

    feats = ["n_words", "mean_dep_dist", "max_depth", "mean_word_len"]
    z = (df[feats] - df[feats].mean()) / df[feats].std()
    df["complexity"] = z.sum(axis=1).round(3)
    df["band"] = pd.qcut(df["complexity"], 3, labels=["easy", "medium", "hard"])

    picked = pd.concat([
        df[df["band"] == b].sample(n, random_state=SEED) for b, n in N_PER_BAND.items()
    ])
    leftovers = df.drop(picked.index)
    practice = leftovers[leftovers["band"] == "easy"].sample(N_PRACTICE, random_state=SEED)

    picked = picked.sort_values("complexity").reset_index(drop=True)
    picked.insert(0, "sid", [f"S{i + 1:03d}" for i in range(len(picked))])
    assert picked["text"].is_unique and len(picked) == 100

    # Comprehension probes: every 5th sentence (by complexity order -> spread over bands)
    rng = random.Random(SEED)
    picked["probe_word"], picked["probe_answer"] = "", ""
    probe_idx = list(range(2, 100, 5))[:N_PROBES]
    answers = ["yes", "no"] * (N_PROBES // 2)
    rng.shuffle(answers)
    # whole content words only: skip short words and sandhi forms ending in க் ச் த் ப் (e.g. எனத்)
    content = lambda t: [w.strip(".,") for w in t.split() if len(w.strip(".,")) >= 4
                         and TAMIL_WORD.match(w.strip(".,"))
                         and not w.strip(".,").endswith(("க்", "ச்", "த்", "ப்"))]
    for i, ans in zip(probe_idx, answers):
        own = content(picked.at[i, "text"])
        if ans == "yes":
            word = rng.choice(own)
        else:
            others = [w for j, t in enumerate(picked["text"]) if j != i
                      for w in content(t) if w not in picked.at[i, "text"]]
            word = rng.choice(others)
        picked.at[i, "probe_word"], picked.at[i, "probe_answer"] = word, ans

    out_csv = HERE / "stimuli" / "sentences.csv"
    out_csv.parent.mkdir(exist_ok=True)
    picked.to_csv(out_csv, index=False, encoding="utf-8-sig")

    stim = [{"sid": r.sid, "text": r.text,
             **({"probe": {"word": r.probe_word, "answer": r.probe_answer}} if r.probe_word else {})}
            for r in picked.itertuples()]
    prac = [{"sid": f"P{i + 1}", "text": t} for i, t in enumerate(practice["text"])]
    js = ("// Generated by select_sentences.py - do not edit by hand.\n"
          "// Source: Dinamani news sentences via UD Tamil-TTB (CC BY-NC-SA 3.0).\n"
          f"const STIMULI = {json.dumps(stim, ensure_ascii=False, indent=1)};\n"
          f"const PRACTICE = {json.dumps(prac, ensure_ascii=False, indent=1)};\n")
    (HERE / "web" / "stimuli.js").write_text(js, encoding="utf-8")

    print(f"\nSelected {len(picked)} unique sentences -> {out_csv.relative_to(HERE)}")
    print(picked.groupby("band", observed=True)[feats].mean().round(2).to_string())
    print("\nSentences per Dinamani source file:")
    print(picked["dinamani_file"].value_counts().sort_index().to_string())
    print(f"Probes: {(picked['probe_answer'] != '').sum()} "
          f"({(picked['probe_answer'] == 'yes').sum()} yes / {(picked['probe_answer'] == 'no').sum()} no)")
    print(f"Practice: {len(prac)} (not among the 100)")


if __name__ == "__main__":
    main()

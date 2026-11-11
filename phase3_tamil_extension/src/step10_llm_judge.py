"""
Step 10 (Phase C, extra): Gemini as an LLM judge, reported SEPARATELY from the human raters.

Gemini answers exactly the questions the human raters saw (same A/B order, same 5 questions),
one item at a time. Then we ask: how close is Gemini to native Tamil readers?

Input : Dataset/human_study/make_google_form.gs   (step 8) the exact A/B texts shown to raters
        Dataset/human_study/items_key.csv         (step 8) which side is the original
        output/human_ratings_long.csv             (step 9) human answers
        output/simplification_scores.csv          (step 8) metric scores
Output: output/llm_judge_<model>.csv              one row per item, Gemini's raw answers (cache)
        results/llm_judge_<model>.md

Needs the GEMINI_API_KEY environment variable (free key from aistudio.google.com).
Free tier has a requests-per-minute limit, so the script waits between calls and retries on 429.
Already-answered items are kept in the cache file, so re-running only fills in what is missing.

Run: python src/step10_llm_judge.py [--model gemini-3.5-flash] [--batch 10]
     (gemini-2.5-flash returns 404 for new keys; Pro models are not on the free tier)
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = os.path.join(BASE, "Dataset", "human_study")
LONG = os.path.join(BASE, "output", "human_ratings_long.csv")
SCORES = os.path.join(BASE, "output", "simplification_scores.csv")
SUMMARY = os.path.join(BASE, "results", "llm_judge_{}.md")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
WAIT = 7          # seconds between calls (free tier ~10 requests/minute)
N_BOOT = 2000
SEED = 0

# Same instructions and questions as the Google Form (make_google_form.gs).
# Several items go in one request (like a rater working through the form), because the
# free tier allows only ~20 requests per model per day.
PROMPT = """Each item shows two Tamil sentences, A and B. Read both carefully and answer. There are no right or wrong answers; we want your honest impression. Imagine a child in Class 3-5 reading them.

For EVERY item below, answer these questions:
1. Easier: which sentence is easier to read? (A, B or Same)
2. Difficulty A: 1 (Very easy) to 5 (Very hard)
3. Difficulty B: 1 (Very easy) to 5 (Very hard)
4. Meaning: do A and B say the same thing? 1 (Completely different) to 5 (Exactly the same)
5. Errors: which has Tamil grammar or word mistakes? (Neither, A, B or Both)

{items}"""
ITEM = "Item {k}\nA:  {a}\nB:  {b}\n"

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "item": {"type": "INTEGER"},
            "easier": {"type": "STRING", "enum": ["A", "B", "Same"]},
            "difficulty_a": {"type": "INTEGER"},
            "difficulty_b": {"type": "INTEGER"},
            "meaning": {"type": "INTEGER"},
            "errors": {"type": "STRING", "enum": ["Neither", "A", "B", "Both"]},
        },
        "required": ["item", "easier", "difficulty_a", "difficulty_b", "meaning", "errors"],
    },
}


# ---------------------------------------------------------------- ask Gemini
def form_items():
    """The exact A/B texts the raters saw, read from the Apps Script file."""
    src = open(os.path.join(HS, "make_google_form.gs"), encoding="utf-8").read()
    forms = json.loads(re.search(r"var FORMS = (\[.*?\n\]);", src, re.S).group(1))
    return [{"form": f["no"], "item": it["k"], "a": it["a"], "b": it["b"]}
            for f in forms for it in f["items"]]


def ask(model, key, batch):
    text = PROMPT.format(items="\n".join(ITEM.format(k=it["item"], a=it["a"], b=it["b"]) for it in batch))
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": SCHEMA}}
    for attempt in range(4):
        r = requests.post(URL.format(model), headers={"x-goog-api-key": key}, json=body, timeout=600)
        if r.status_code == 429 and "PerDay" in r.text:
            sys.exit(f"Daily free quota for {model} is used up; answers so far are cached. "
                     "Run again after the quota resets (midnight US Pacific = ~12:30 pm India).")
        if r.status_code in (429, 500, 503):
            time.sleep(60 * (attempt + 1))
            continue
        r.raise_for_status()
        answers = {a["item"]: a for a in json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])}
        missing = [it["item"] for it in batch if it["item"] not in answers]
        if missing:
            sys.exit(f"Gemini skipped items {missing}; re-run to retry this batch.")
        for a in answers.values():
            for q in ["difficulty_a", "difficulty_b", "meaning"]:
                a[q] = int(min(5, max(1, a[q])))
        return answers
    sys.exit(f"Gemini kept refusing (last status {r.status_code}): {r.text[:300]}")


def run_judge(model, batch_size):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (setx GEMINI_API_KEY \"...\", then reopen the terminal).")
    cache = os.path.join(BASE, "output", f"llm_judge_{model}.csv")
    done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["form", "item"])
    have = set(zip(done.form, done.item))
    todo = [it for it in form_items() if (it["form"], it["item"]) not in have]
    print(f"{model}: {len(have)} items cached, {len(todo)} to ask, {batch_size} per request")
    # batches never mix the two forms, so item numbers stay unique inside a request
    batches = [b for f in sorted({it["form"] for it in todo})
               for fi in [[it for it in todo if it["form"] == f]]
               for b in (fi[i:i + batch_size] for i in range(0, len(fi), batch_size))]
    for n, batch in enumerate(batches, 1):
        answers = ask(model, key, batch)
        rows = pd.DataFrame([{**answers[it["item"]], "form": it["form"]} for it in batch])
        rows[["form", "item", "easier", "difficulty_a", "difficulty_b", "meaning", "errors"]].to_csv(
            cache, mode="a", header=not os.path.exists(cache), index=False, encoding="utf-8")
        print(f"  request {n}/{len(batches)}: form {batch[0]['form']} items "
              f"{batch[0]['item']}-{batch[-1]['item']} done")
        time.sleep(WAIT)
    return pd.read_csv(cache)


# ---------------------------------------------------------------- compare with humans
def to_orig_simp(d):
    """A/B answers -> original/simplified (same mapping as step 9)."""
    a_orig = d.A_is == "original"
    d["diff_orig"] = np.where(a_orig, d.difficulty_a, d.difficulty_b).astype(float)
    d["diff_simp"] = np.where(a_orig, d.difficulty_b, d.difficulty_a).astype(float)
    d["easier"] = np.select([d.easier == "A", d.easier == "B"],
                            [np.where(a_orig, "original", "simplified"),
                             np.where(a_orig, "simplified", "original")], "same")
    d["err_orig"] = (d.errors == "Both") | ((d.errors == "A") & a_orig) | ((d.errors == "B") & ~a_orig)
    d["err_simp"] = (d.errors == "Both") | ((d.errors == "A") & ~a_orig) | ((d.errors == "B") & a_orig)
    return d


def texts(df, o, s):
    """Stack original and simplified texts into one vector of 2 x items."""
    return np.concatenate([df[o].to_numpy(float), df[s].to_numpy(float)])


def boot_ci(df, f, rng):
    idx = np.arange(len(df))
    v = [f(df.iloc[rng.choice(idx, len(idx))]) for _ in range(N_BOOT)]
    return np.nanpercentile(v, [2.5, 97.5])


def compare(model, g):
    key = pd.read_csv(os.path.join(HS, "items_key.csv"))
    g = to_orig_simp(g.merge(key, on=["form", "item"]))
    h = pd.read_csv(LONG)
    sc = pd.read_csv(SCORES)
    rng = np.random.default_rng(SEED)

    hum = h.groupby(["form", "item"]).agg(
        h_orig=("diff_orig", "mean"), h_simp=("diff_simp", "mean"),
        h_pick_simp=("easier", lambda e: (e == "simplified").mean()),
        h_majority=("easier", lambda e: e.value_counts().idxmax()),
        h_meaning=("meaning", "mean")).reset_index()
    it = g.merge(hum, on=["form", "item"]).merge(sc, on=["id", "system"])

    rho = lambda df, o, s: spearmanr(texts(df, o, s), texts(df, "h_orig", "h_simp")).statistic
    gem_rho = rho(it, "diff_orig", "diff_simp")
    gem_ci = boot_ci(it, lambda df: rho(df, "diff_orig", "diff_simp"), rng)

    # Baseline: how well does ONE human agree with the mean of the OTHER humans?
    loo = []
    for r, hr in h.groupby("rater"):
        rest = h[(h.form == hr.form.iloc[0]) & (h.rater != r)].groupby("item")[["diff_orig", "diff_simp"]].mean()
        m = hr.set_index("item")[["diff_orig", "diff_simp"]].join(rest, rsuffix="_rest")
        loo.append(spearmanr(texts(m, "diff_orig", "diff_simp"),
                             texts(m, "diff_orig_rest", "diff_simp_rest")).statistic)
    loo = np.array(loo)
    # Same comparison for Gemini: Gemini vs the mean of all humans, per form, averaged
    gem_form = np.mean([rho(f, "diff_orig", "diff_simp") for _, f in it.groupby("form")])

    agree_major = (it.easier == it.h_majority).mean()
    L = [f"# Phase C extra: Gemini ({model}) as an LLM judge", "",
         "Gemini answered the same 60 items and 5 questions as the human raters, with the same A/B order. "
         "It is reported separately and is NOT counted as a rater.", "",
         "## 1. Gemini's answers next to the humans'", "",
         "| Judge | Picks simplified as easier | ...picks original | Difficulty orig -> simp | Meaning kept (1-5) "
         "| Errors in orig | Errors in simp |", "|---|---|---|---|---|---|---|",
         f"| Humans (12, mean) | {it.h_pick_simp.mean():.0%} | {(h.easier == 'original').mean():.0%} | "
         f"{it.h_orig.mean():.2f} -> {it.h_simp.mean():.2f} | {it.h_meaning.mean():.2f} | "
         f"{h.err_orig.mean():.0%} | {h.err_simp.mean():.0%} |",
         f"| Gemini | {(it.easier == 'simplified').mean():.0%} | {(it.easier == 'original').mean():.0%} | "
         f"{it.diff_orig.mean():.2f} -> {it.diff_simp.mean():.2f} | {it.meaning.mean():.2f} | "
         f"{it.err_orig.mean():.0%} | {it.err_simp.mean():.0%} |", "",
         "## 2. Does Gemini agree with native readers?", "",
         f"- Difficulty, Spearman(Gemini, mean of humans) over {2 * len(it)} texts: "
         f"**{gem_rho:+.3f}** [95% CI {gem_ci[0]:+.3f}, {gem_ci[1]:+.3f}]",
         f"- Same, computed per form and averaged: {gem_form:+.3f}",
         f"- Baseline, one human vs the mean of the other humans in the same form: "
         f"mean {loo.mean():+.3f} (range {loo.min():+.3f} to {loo.max():+.3f}, {len(loo)} raters)",
         f"- 'Which is easier?': Gemini gives the human majority answer on {agree_major:.0%} of items", "",
         "If Gemini's number is at or above the human baseline, Gemini agrees with the crowd at least as "
         "well as a typical single native reader does.", "",
         "## 3. Which metric agrees with Gemini?", "",
         "| Metric | Spearman with Gemini difficulty | 95% CI |", "|---|---|---|"]
    for m in ["word", "morph"]:
        f = lambda df, m=m: spearmanr(texts(df, f"score_{m}_orig", f"score_{m}_simp"),
                                      texts(df, "diff_orig", "diff_simp")).statistic
        ci = boot_ci(it, f, rng)
        L.append(f"| {m} | {f(it):+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    with open(SUMMARY.format(model), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--batch", type=int, default=10, help="items per request")
    args = ap.parse_args()
    compare(args.model, run_judge(args.model, args.batch))

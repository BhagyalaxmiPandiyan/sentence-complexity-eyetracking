"""
Step 5: features + pairs for the StoryWeaver dataset (step 4), so step 6 can
evaluate it exactly like the textbook data.

Input : output/storyweaver_sentences.csv     (step 4)
Output: output/storyweaver_features.csv      same 12 features as step 3, one row per sentence
        output/storyweaver_pairs.csv         graded pairs, same columns as graded_pairs.csv

  - Levels 1-4 only ("5" is not one of StoryWeaver's documented reading levels).
  - Pairs only between levels >= MIN_GAP apart: 1-3, 1-4, 2-4
    (neighbouring levels overlap too much to trust the label).
  - Train/test split is by BOOK: all sentences of one book's blurb go to the same side.
  - class_a / class_b hold the StoryWeaver level (column names kept so step 6 works unchanged).

Run: python src/step5_storyweaver_features.py      (CPU, ~1 hour; features are cached,
     delete output/storyweaver_features.csv to recompute)
then: python src/step6_evaluate_metric.py storyweaver
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_textbook_features import load_tools, sentence_features    # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(BASE, "output", "storyweaver_sentences.csv")
FEATS = os.path.join(BASE, "output", "storyweaver_features.csv")
PAIRS = os.path.join(BASE, "output", "storyweaver_pairs.csv")

LEVELS = [1, 2, 3, 4]
MIN_GAP = 2
PAIRS_PER_LEVEL_PAIR = {"train": 2000, "test": 500}
TEST_FRACTION = 0.2
SEED = 42


def book_split(book_id):
    """Deterministic book-level split (same book -> same side, every run)."""
    return "test" if zlib.crc32(str(book_id).encode()) % 100 < TEST_FRACTION * 100 else "train"


def compute_features():
    s = pd.read_csv(IN)
    s = s[s.level.isin(LEVELS)].reset_index(drop=True)
    seg, nlp, tok, lm = load_tools()
    rows = []
    for i, sent in enumerate(s.sentence):
        rows.append(sentence_features(sent, seg, nlp, tok, lm))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(s)}", flush=True)
    out = pd.concat([s[["book_id", "level", "sentence"]], pd.DataFrame(rows)], axis=1)
    out.to_csv(FEATS, index=False, encoding="utf-8")
    print(f"wrote {FEATS}  ({len(out)} sentences)")
    return out


def build_pairs(f):
    rng = np.random.default_rng(SEED)
    f = f.assign(split=f.book_id.map(book_split))
    rows = []
    for split in ["train", "test"]:
        by_level = {l: g for l, g in f[f.split == split].groupby("level")}
        for lo in LEVELS:
            for hi in LEVELS:
                if hi - lo < MIN_GAP or lo not in by_level or hi not in by_level:
                    continue
                easy, hard = by_level[lo], by_level[hi]
                n = min(PAIRS_PER_LEVEL_PAIR[split], len(easy) * len(hard))
                ei = rng.integers(0, len(easy), n)
                hj = rng.integers(0, len(hard), n)
                for i, j in {(int(i), int(j)) for i, j in zip(ei, hj)}:
                    e, h = easy.iloc[i], hard.iloc[j]
                    flip = rng.random() < 0.5
                    a, b = (h, e) if flip else (e, h)
                    rows.append({
                        "sent_a": a.sentence, "class_a": a.level,
                        "sent_b": b.sentence, "class_b": b.level,
                        "harder": "a" if flip else "b",
                        "gap": hi - lo,
                        "length_matched": abs(a.n_words - b.n_words) <= 1,
                        "split": split,
                    })
    p = pd.DataFrame(rows)
    p.insert(0, "pair_id", range(len(p)))
    p.to_csv(PAIRS, index=False, encoding="utf-8")
    print(f"pairs: {len(p)} (train {sum(p.split == 'train')}, test {sum(p.split == 'test')}), "
          f"length-matched {p.length_matched.sum()}")


def main():
    f = pd.read_csv(FEATS) if os.path.exists(FEATS) else compute_features()
    print(f.drop(columns=["book_id", "sentence"]).groupby("level").mean().round(2).to_string())
    build_pairs(f)


if __name__ == "__main__":
    main()

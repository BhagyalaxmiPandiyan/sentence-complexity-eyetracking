"""
Step 2: build graded sentence PAIRS: "which of these two sentences is harder?"
Same pairwise setup as Phase 1 System 2, but the label comes from the textbook
grade instead of human judgment.

  - Only pairs whose classes differ by >= MIN_GAP (neighbouring grades overlap
    too much for the label to be trustworthy).
  - Each pair is shown in random order; `harder` says which side is from the
    higher class (a/b), so a metric can't win by always picking one side.
  - `length_matched` marks pairs whose word counts differ by <= 1. Higher
    grades have longer sentences, so on the full set sentence length alone
    already scores well; the length-matched subset shows whether a metric
    (e.g. the morphology-aware one) adds anything BEYOND length.
  - train/test split is by textbook page: all sentences from one page go to the
    same side, so a model can't be tested on a passage it was trained on.

Output: output/graded_pairs.csv
Run:    python src/step2_build_pairs.py
"""
import os, sys, zlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(BASE, "output", "graded_sentences.csv")
OUT = os.path.join(BASE, "output", "graded_pairs.csv")

MIN_GAP = 3
PAIRS_PER_CLASS_PAIR = 150        # cap so the big Class 8 book doesn't dominate
TEST_FRACTION = 0.2
SEED = 42


def page_split(cls, page):
    """Deterministic page-level split (same page -> same side, every run)."""
    return "test" if zlib.crc32(f"{cls}-{page}".encode()) % 100 < TEST_FRACTION * 100 else "train"


def main():
    rng = np.random.default_rng(SEED)
    s = pd.read_csv(IN)
    s["split"] = [page_split(c, p) for c, p in zip(s["class"], s["page"])]

    rows = []
    classes = sorted(s["class"].unique())
    for split in ["train", "test"]:
        by_class = {c: g for c, g in s[s.split == split].groupby("class")}
        for lo in classes:
            for hi in classes:
                if hi - lo < MIN_GAP or lo not in by_class or hi not in by_class:
                    continue
                easy, hard = by_class[lo], by_class[hi]
                n = min(PAIRS_PER_CLASS_PAIR, len(easy) * len(hard))
                ei = rng.integers(0, len(easy), n)
                hj = rng.integers(0, len(hard), n)
                for i, j in {(int(i), int(j)) for i, j in zip(ei, hj)}:
                    e, h = easy.iloc[i], hard.iloc[j]
                    flip = rng.random() < 0.5
                    a, b = (h, e) if flip else (e, h)
                    rows.append({
                        "sent_a": a.sentence, "class_a": a["class"],
                        "sent_b": b.sentence, "class_b": b["class"],
                        "harder": "a" if flip else "b",
                        "gap": hi - lo,
                        "length_matched": abs(a.n_words - b.n_words) <= 1,
                        "split": split,
                    })
    p = pd.DataFrame(rows)
    p.insert(0, "pair_id", range(len(p)))
    p.to_csv(OUT, index=False, encoding="utf-8")

    print(f"sentences: {len(s)} ({(s.split == 'test').mean():.0%} test by page)")
    print(f"pairs: {len(p)}  (train {sum(p.split == 'train')}, test {sum(p.split == 'test')}), "
          f"length-matched {p.length_matched.sum()}, harder=a {np.mean(p.harder == 'a'):.1%}")
    print(f"wrote {OUT}")

    # --- naive baselines: pick the side with the larger value as "harder"
    def acc(df, feat):
        fa, fb = df.sent_a.map(feat), df.sent_b.map(feat)
        pred = np.where(fa > fb, "a", np.where(fb > fa, "b", rng.choice(["a", "b"], len(df))))
        return (pred == df.harder).mean()

    n_words = lambda t: len(t.split())
    word_len = lambda t: np.mean([len(w.strip(".,?!;:'\"“”‘’")) for w in t.split()])
    print("\nNaive baselines (test pairs), accuracy of 'larger value = harder':")
    for name, df in [("all pairs", p[p.split == "test"]),
                     ("length-matched", p[(p.split == "test") & p.length_matched])]:
        print(f"  {name:15s} n={len(df):5d}  sentence length {acc(df, n_words):.1%}  "
              f"mean word length {acc(df, word_len):.1%}")


if __name__ == "__main__":
    main()

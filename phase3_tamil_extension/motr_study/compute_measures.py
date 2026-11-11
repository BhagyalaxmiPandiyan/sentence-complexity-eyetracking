"""
Turn MoTR mouse data into Dundee-style reading measures (FFD, FPD, RPD, TD).

Input (any mix):
  - the "trials" tab of the Google Sheet, downloaded as CSV   (File -> Download -> .csv)
  - JSON files readers downloaded at the end (test mode / upload failed)

Each mouse "visit" to a word (mouse enters the word's box ... leaves it) is one fixation.
Visits shorter than MIN_VISIT_MS are the mouse just passing over a word on the way somewhere
else (like a saccade), so they are dropped. After that the measures are computed with exactly
the same rules as phase1_replication/src/step1_parse_dundee.py:
  FFD = first fixation on the word
  FPD = first fixation + fixations straight after it on the same word
  RPD = fixations on the word before the reader first moves to a word further right
  TD  = all fixations on the word
Skipped words get 0 (Phase 1 also lets skipping readers contribute 0 to the average).

Readers with comprehension-probe accuracy below MIN_PROBE_ACC are excluded.

Usage:  python compute_measures.py data/trials.csv data/motr_tamil_R04.json ...
Output: output/word_measures_by_reader.csv   one row per reader x word
        output/word_measures.csv             averaged over readers (Phase 1 format)
        output/sentence_measures.csv         averaged per sentence
        output/reader_summary.csv            probe accuracy, reading speed, kept / excluded
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
MIN_VISIT_MS = 40
MIN_PROBE_ACC = 0.7
MEASURES = ["ffd", "fpd", "rpd", "td"]


def load_trials(paths):
    frames = []
    for p in map(Path, paths):
        if p.suffix.lower() == ".json":
            d = json.loads(p.read_text(encoding="utf-8"))
            frames.append(pd.DataFrame([r for r in d["results"] if r.get("type") == "trial"]))
        else:
            frames.append(pd.read_csv(p, dtype=str, encoding="utf-8"))
    df = pd.concat(frames, ignore_index=True)
    df = df.astype({"practice": int, "n_words": int, "rt_ms": float})
    # a reader may be in both the sheet and a JSON backup, or an upload may be retried
    return df.drop_duplicates(["pid", "practice", "sid"], keep="first")


def word_measures(visits, n_words):
    """visits: [[word_index, start_ms, dur_ms], ...] in time order -> {wnum: {ffd, fpd, rpd, td}}"""
    fix = [(w, dur) for w, _, dur in visits if dur >= MIN_VISIT_MS]
    out = {}
    for wn in range(n_words):
        idx = [k for k, (w, _) in enumerate(fix) if w == wn]
        if not idx:
            out[wn] = {m: 0 for m in MEASURES}
            continue
        durs = [fix[k][1] for k in idx]
        fpd = durs[0]
        for a, b in zip(idx, idx[1:]):
            if any(fix[k][0] != wn for k in range(a + 1, b)):
                break  # left the word -> first pass over
            fpd += fix[b][1]
        forward = next((k for k in range(idx[0] + 1, len(fix)) if fix[k][0] > wn), None)
        rpd = sum(fix[k][1] for k in idx if forward is None or k < forward)
        out[wn] = {"ffd": durs[0], "fpd": fpd, "rpd": rpd, "td": sum(durs)}
    return out


def main(paths):
    df = load_trials(paths)
    main_tr = df[df["practice"] == 0].copy()

    probes = main_tr[main_tr["probe_word"].fillna("") != ""]
    readers = pd.DataFrame({
        "n_sentences": main_tr.groupby("pid").size(),
        "probe_acc": probes.groupby("pid")["correct"].apply(lambda s: s.astype(int).mean()),
        "median_rt_ms": main_tr.groupby("pid")["rt_ms"].median(),
        "resized_trials": main_tr.groupby("pid")["resized"].apply(lambda s: s.astype(int).sum()),
    })
    readers["kept"] = readers["probe_acc"] >= MIN_PROBE_ACC
    keep = set(readers.index[readers["kept"]])

    rows = []
    for r in main_tr[main_tr["pid"].isin(keep)].itertuples():
        words = r.text.split()
        for wn, m in word_measures(json.loads(r.visits), r.n_words).items():
            rows.append({"pid": r.pid, "sid": r.sid, "wnum": wn + 1, "word": words[wn],
                         "fixated": int(m["td"] > 0), **m})
    by_reader = pd.DataFrame(rows)

    words = (by_reader.groupby(["sid", "wnum", "word"])
             .agg(n_readers=("pid", "nunique"), skip_rate=("fixated", lambda s: 1 - s.mean()),
                  **{m: (m, "mean") for m in MEASURES})
             .reset_index().round(2))
    sents = (by_reader.groupby(["sid", "pid"])[MEASURES].sum().groupby("sid").mean()
             .join(main_tr[main_tr["pid"].isin(keep)].groupby("sid")["rt_ms"].mean())
             .round(1).reset_index())

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    by_reader.to_csv(out / "word_measures_by_reader.csv", index=False, encoding="utf-8-sig")
    words.to_csv(out / "word_measures.csv", index=False, encoding="utf-8-sig")
    sents.to_csv(out / "sentence_measures.csv", index=False, encoding="utf-8-sig")
    readers.round(3).to_csv(out / "reader_summary.csv", encoding="utf-8-sig")

    print(readers.round(3).to_string())
    print(f"\nKept {len(keep)} / {len(readers)} readers (probe accuracy >= {MIN_PROBE_ACC})")
    fixated = by_reader[by_reader["fixated"] == 1]
    if len(fixated):
        print(f"Words: {len(words)}   skip rate: {1 - by_reader['fixated'].mean():.1%}")
        for m in MEASURES:
            print(f"  {m.upper()}: {fixated[m].mean():.1f} ms (fixated words)")
    print(f"\nWrote {out.relative_to(HERE)}/word_measures.csv and friends")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])


"""
Word-level eye-tracking measures from one natural_reading.py session.

Steps:
  1. Gaze samples while each sentence was on screen (drift-corrected).
  2. Light smoothing (running median of 3 samples) against webcam jitter.
  3. Fixation detection (I-DT): a run of samples whose spread stays
     within --dispersion px for at least --min-dur ms. A gap of more
     than --max-gap ms without valid samples (blink, lost face) ends it.
  4. Each fixation is assigned to the nearest text line, then to a word
     on that line by its horizontal position (the word box is widened by
     half the gap on each side). Fixations far from every line count as
     off-screen.
  5. FFD, FPD, RPD, TD per word, with the same definitions as the
     English Dundee data in phase1_replication/src/step1_parse_dundee.py:
        FFD - first fixation on the word
        FPD - first fixation plus directly following ones on the word
        RPD - fixations on the word before the eye first moves to a
              word further right (all of TD if it never does)
        TD  - all fixations on the word
     Skipped words get 0 and fixated = 0.
  6. Noise filter: fixations longer than --max-fix ms are dropped;
     sentences with too little valid gaze or gaze mostly off the text
     are excluded (sentence_ok = 0); word values above mean + 3 SD
     (per measure, this reader) are outliers. keep = 1 for the rest.

Writes into the session folder:
    fixations.csv      one row per detected fixation (with noise label)
    word_measures.csv  one row per word per trial (with filter flags)
    sentences.csv      one row per sentence: reading time, gaze quality,
                       question, answer, correct
    dundee_ma1p.dat, dundee_ma2p.dat, dundee_wrdp.dat
                       the main trials in Dundee Corpus format
                       (see dundee_format.py)
    <participant>_reading_data.xlsx
                       all of the above as one Excel workbook

Checks printed at the end:
    - gaze data quality and validation accuracy from session.json
    - word length effect (longer words should get longer TD)
    - autopilot runs only: detected vs scripted (true) measures

Usage:
    .\\.venv\\Scripts\\python.exe analyze_reading.py data\\P01_20261101_103000
"""

import argparse
import csv
import json
import os

import numpy as np

import dundee_format
import excel_export
from tamil_render import count_graphemes

MEASURES = ["ffd", "fpd", "rpd", "td"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def sentence_windows(events):
    """trial_index -> (sentence_on time, sentence_off time)."""
    on, windows = {}, {}
    for row in events:
        trial = int(row["trial_index"])
        if row["event"] == "sentence_on":
            on[trial] = float(row["t"])
        elif row["event"] == "sentence_off" and trial in on:
            windows[trial] = (on[trial], float(row["t"]))
    return windows


def running_median(values, size=3):
    if len(values) < size:
        return values.copy()
    half = size // 2
    padded = np.concatenate([values[:half], values, values[-half:]])
    return np.array([np.median(padded[i:i + size])
                     for i in range(len(values))])


def detect_fixations(t, x, y, dispersion, min_dur, max_gap):
    """I-DT on valid samples. Returns [(start, end, x, y)] in seconds/px.

    end is the last sample time plus one sample interval, so a fixation
    covering n samples lasts about n sample intervals.
    """
    if len(t) < 2:
        return []
    interval = float(np.median(np.diff(t)))
    fixations = []
    i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t):
            if t[j + 1] - t[j] > max_gap:
                break
            xs, ys = x[i:j + 2], y[i:j + 2]
            if (xs.max() - xs.min()) + (ys.max() - ys.min()) > dispersion:
                break
            j += 1
        if t[j] - t[i] + interval >= min_dur:
            fixations.append((t[i], t[j] + interval,
                              float(np.median(x[i:j + 1])),
                              float(np.median(y[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return fixations


def assign_word(fx, fy, boxes, gap, line_band):
    centres = sorted({(y1 + y2) / 2 for _, y1, _, y2 in boxes})
    line_y = min(centres, key=lambda c: abs(fy - c))
    if abs(fy - line_y) > line_band:
        return None
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        if (y1 + y2) / 2 == line_y and x1 - gap / 2 <= fx <= x2 + gap / 2:
            return index
    return None


def gaze_gaps(rows, t_off, min_dur):
    """Runs without valid gaze (blink, face lost): [(start, duration_s)]."""
    gaps, start = [], None
    for row in rows:
        t = float(row["t"])
        if row["valid"] != "1":
            start = t if start is None else start
        elif start is not None:
            if t - start >= min_dur:
                gaps.append((start, t - start))
            start = None
    if start is not None and t_off - start >= min_dur:
        gaps.append((start, t_off - start))
    return gaps


def word_measures(sequence, n_words):
    """Phase 1 definitions. sequence: [(word, duration_ms)] in order."""
    result = {w: dict.fromkeys(MEASURES, 0.0) | {"n_fix": 0}
              for w in range(n_words)}
    for word in range(n_words):
        positions = [k for k, (w, _) in enumerate(sequence) if w == word]
        if not positions:
            continue
        durations = [sequence[k][1] for k in positions]
        first = positions[0]

        fpd = durations[0]
        for k in range(first + 1, len(sequence)):
            if sequence[k][0] != word:
                break
            fpd += sequence[k][1]

        forward = [k for k in range(first + 1, len(sequence))
                   if sequence[k][0] > word]
        if forward:
            rpd = sum(d for k, d in zip(positions, durations)
                      if k < forward[0])
        else:
            rpd = sum(durations)

        result[word] = {"ffd": durations[0], "fpd": fpd, "rpd": rpd,
                        "td": sum(durations), "n_fix": len(positions)}
    return result


def true_sequences(events, windows):
    """Autopilot only: trial_index -> [(word, duration_ms)] as scripted."""
    sequences = {}
    for row in events:
        if row["event"] != "autopilot_truth" or not row["detail"].startswith("{"):
            continue
        trial = int(row["trial_index"])
        if trial not in windows:
            continue
        truth = json.loads(row["detail"])
        sequences.setdefault(trial, []).append(
            (truth["word"], truth["dur"] * 1000))
    return sequences


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="session folder from natural_reading.py")
    parser.add_argument("--dispersion", type=float, default=100,
                        help="max x-spread + y-spread inside a fixation (px)")
    parser.add_argument("--min-dur", type=float, default=80,
                        help="shortest fixation (ms)")
    parser.add_argument("--max-fix", type=float, default=1200,
                        help="longest fixation (ms); longer ones are noise "
                             "(staring, lost tracking) and are dropped")
    parser.add_argument("--max-gap", type=float, default=100,
                        help="longest sample gap inside a fixation (ms)")
    parser.add_argument("--line-band", type=float, default=150,
                        help="max vertical distance from the nearest "
                             "line (px)")
    parser.add_argument("--min-blink", type=float, default=60,
                        help="shortest gaze gap written as *Blink in the "
                             "Dundee files (ms)")
    parser.add_argument("--min-valid", type=float, default=0.7,
                        help="sentences with a smaller share of valid gaze "
                             "samples are excluded")
    parser.add_argument("--max-off-text", type=float, default=0.5,
                        help="sentences with a larger share of fixation "
                             "time off the text are excluded")
    parser.add_argument("--outlier-sd", type=float, default=3.0,
                        help="word values above mean + this many SD (per "
                             "measure, this reader) are excluded")
    parser.add_argument("--no-smoothing", action="store_true")
    args = parser.parse_args()

    folder = args.folder
    session = json.load(open(os.path.join(folder, "session.json"),
                             encoding="utf-8"))
    participant = session["participant"]
    gap = session["region_gap_px"]
    events = read_csv(os.path.join(folder, "events.csv"))
    gaze = read_csv(os.path.join(folder, "gaze.csv"))
    trials = {int(r["trial_index"]): r
              for r in read_csv(os.path.join(folder, "trials.csv"))}
    windows = sentence_windows(events)
    if not windows:
        raise SystemExit("No sentence_on/sentence_off events: was this "
                         "session recorded with natural_reading.py?")

    samples = {}
    for row in gaze:
        trial = int(row["trial_index"])
        if trial in windows:
            samples.setdefault(trial, []).append(row)

    fixation_rows, word_rows, sentence_rows, sequences = [], [], [], {}
    screens, dundee_events = [], []
    n_samples = n_valid = 0
    for trial, (t_on, t_off) in sorted(windows.items()):
        info = trials[trial]
        words = info["regions"].split(" | ")
        boxes = json.loads(info["boxes"])
        critical = {int(c) for c in info["critical_region"].split()}

        rows = [r for r in samples.get(trial, [])
                if t_on <= float(r["t"]) <= t_off]
        n_rows = len(rows)
        timeline = [(start, info["item_id"], "blink", length * 1000, None,
                     None) for start, length in gaze_gaps(
                         rows, t_off, args.min_blink / 1000)]
        rows = [r for r in rows if r["valid"] == "1"]
        n_samples += n_rows
        n_valid += len(rows)
        t = np.array([float(r["t"]) for r in rows])
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        if not args.no_smoothing:
            x, y = running_median(x), running_median(y)

        # Fixations; too-long ones are noise and left out of all measures
        sequence, on_text_ms, off_text_ms, n_long = [], 0.0, 0.0, 0
        for number, (start, end, fx, fy) in enumerate(detect_fixations(
                t, x, y, args.dispersion, args.min_dur / 1000,
                args.max_gap / 1000)):
            word = assign_word(fx, fy, boxes, gap, args.line_band)
            duration = (end - start) * 1000
            noise = ("too_long" if duration > args.max_fix
                     else "off_text" if word is None else "")
            fixation_rows.append([
                trial, number, round(start - t_on, 4), round(duration, 1),
                round(fx, 1), round(fy, 1), "" if word is None else word,
                noise,
            ])
            if noise == "too_long":
                n_long += 1
                continue
            if word is None:
                off_text_ms += duration
            else:
                on_text_ms += duration
                sequence.append((word, duration))
            timeline.append((start, info["item_id"], "fix", duration, fx,
                             word))
        sequences[trial] = sequence
        if info["phase"] == "main":
            screens.append((info["item_id"], words, boxes))
            dundee_events += [event[1:] for event in sorted(
                timeline, key=lambda e: e[0])]

        # Sentence quality: exclude sentences the webcam did not follow
        valid_share = len(rows) / max(n_rows, 1)
        off_share = off_text_ms / max(on_text_ms + off_text_ms, 1)
        problems = []
        if valid_share < args.min_valid:
            problems.append("little valid gaze")
        if off_share > args.max_off_text:
            problems.append("gaze mostly off the text")
        if not sequence:
            problems.append("no fixation on a word")
        sentence_rows.append({
            "participant": participant, "trial_index": trial,
            "phase": info["phase"], "item_id": info["item_id"],
            "condition": info["condition"], "n_words": len(words),
            "n_lines": len({box[1] for box in boxes}),
            "reading_time_ms": round((t_off - t_on) * 1000),
            "valid_gaze_pct": round(100 * valid_share, 1),
            "fixations": len(sequence), "too_long_dropped": n_long,
            "off_text_pct": round(100 * off_share, 1),
            "drift_x": float(info["offset_x"]),
            "drift_y": float(info["offset_y"]),
            "question": info["question"], "correct_answer": info["answer"],
            "response": info["response"],
            "answer_correct": int(info["correct"]) if info["correct"] else "",
            "answer_rt_ms": float(info["question_rt_ms"])
            if info["question_rt_ms"] else "",
            "sentence_ok": int(not problems),
            "excluded_because": "; ".join(problems),
        })

        measures = word_measures(sequence, len(words))
        for index, word in enumerate(words):
            m = measures[index]
            word_rows.append({
                "participant": participant, "trial_index": trial,
                "phase": info["phase"], "item_id": info["item_id"],
                "type": info["type"], "condition": info["condition"],
                "word_index": index, "word": word,
                "n_letters": count_graphemes(word),
                "critical": int(index in critical),
                **{k: round(m[k], 1) for k in MEASURES},
                "n_fix": m["n_fix"], "fixated": int(m["n_fix"] > 0),
                "sentence_ok": int(not problems),
            })

    # Outliers: per measure, fixated words of good sentences of this reader
    pool = [w for w in word_rows if w["fixated"] and w["sentence_ok"]]
    limits = {k: float(np.mean([w[k] for w in pool]) + args.outlier_sd
                       * np.std([w[k] for w in pool])) if pool else np.inf
              for k in MEASURES}
    for w in word_rows:
        high = [k.upper() for k in MEASURES if w["fixated"]
                and w["sentence_ok"] and w[k] > limits[k]]
        w["outlier"] = ", ".join(high)
        w["keep"] = int(w["sentence_ok"] and not high)

    # Dundee screens in item_id order, events in the order they happened
    screens.sort(key=lambda screen: screen[0])
    dundee_paths, dundee_sheets, table = dundee_format.write_session(
        folder, screens, dundee_events)
    position = {(r["item_id"], r["serial"] - 1): r for r in table}
    for w in word_rows:
        r = position.get((w["item_id"], w["word_index"]))
        w["screen"] = r["screen"] if r else ""
        w["wnum"] = r["wnum"] if r else ""

    fixation_header = ["trial_index", "fixation", "start_s", "duration_ms",
                       "x", "y", "word_index", "noise"]
    word_header = ["participant", "trial_index", "phase", "item_id",
                   "screen", "wnum", "type", "condition", "word_index",
                   "word", "n_letters", "critical"] + MEASURES + [
                   "n_fix", "fixated", "sentence_ok", "outlier", "keep"]
    sentence_header = list(sentence_rows[0])
    write_csv(os.path.join(folder, "fixations.csv"), fixation_header,
              fixation_rows)
    write_csv(os.path.join(folder, "word_measures.csv"), word_header,
              [[w[k] for k in word_header] for w in word_rows])
    write_csv(os.path.join(folder, "sentences.csv"), sentence_header,
              [[s[k] for k in sentence_header] for s in sentence_rows])

    # Same data as one Excel workbook, Dundee tables included
    main_words = [w for w in word_rows if w["phase"] == "main"]
    clean = [[w["wnum"], w["screen"], w["item_id"], w["condition"],
              w["word"], w["n_letters"]] + [w[k] for k in MEASURES]
             for w in sorted(main_words, key=lambda w: (w["wnum"]))
             if w["keep"] and w["fixated"]]
    excel_path = os.path.join(folder, f"{participant}_reading_data.xlsx")
    excel_export.write_workbook(excel_path, [
        ("README", None, readme_lines(session, args)),
        ("word_measures_clean", ["WNUM", "SCREEN", "ITEM_ID", "BAND",
                                 "WORD", "WLEN", "FFD", "FPD", "RPD",
                                 "TD"], clean),
        ("word_measures_all", word_header,
         [[w[k] for k in word_header] for w in word_rows]),
        ("sentences_answers", sentence_header,
         [[s[k] for k in sentence_header] for s in sentence_rows]),
        ("RT_dataset1P", *dundee_sheets["ma1p"]),
        ("RT_dataset2P", *dundee_sheets["ma2p"]),
        ("per-word-info", *dundee_sheets["wrdp"]),
        ("fixations", fixation_header, fixation_rows),
    ])

    # ---------- Report ----------
    durations = [r[3] for r in fixation_rows]
    noise = [r[7] for r in fixation_rows]
    good = [s for s in sentence_rows if s["sentence_ok"]]
    kept = [w for w in word_rows if w["keep"] and w["fixated"]]
    answered = [s for s in sentence_rows
                if s["phase"] == "main" and s["answer_correct"] != ""]
    print(f"Session {os.path.basename(folder)} | gaze source: "
          f"{session['gaze_source']} | {len(windows)} sentences")
    for label, v in session.get("validation", {}).items():
        if "median_error" in v:
            print(f"  Validation {label:<7}: median error x "
                  f"{v['median_abs_error_x']} px, y "
                  f"{v['median_abs_error_y']} px")
    print(f"  Valid gaze samples while reading: {n_valid}/{n_samples} "
          f"({100 * n_valid / max(n_samples, 1):.0f}%)")
    print(f"  Fixations: {len(fixation_rows)} "
          f"({len(fixation_rows) / len(windows):.1f} per sentence), mean "
          f"duration {np.mean(durations) if durations else 0:.0f} ms")
    print(f"  Noise filter: {noise.count('too_long')} too-long fixations "
          f"dropped, {noise.count('off_text')} off the text; "
          f"{len(windows) - len(good)} sentences excluded; "
          f"{sum(1 for w in word_rows if w['outlier'])} outlier words")
    print(f"  Answers: {sum(int(s['answer_correct']) for s in answered)}"
          f"/{len(answered)} correct")
    fixated = [w for w in word_rows if w["fixated"]]
    print(f"  Words fixated: {len(fixated)}/{len(word_rows)} "
          f"(skipping {100 * (1 - len(fixated) / len(word_rows)):.0f}%), "
          f"{len(kept)} kept after cleaning")
    for name in MEASURES:
        values = [w[name] for w in kept]
        print(f"  Mean {name.upper()} (kept words): "
              f"{np.mean(values) if values else 0:.0f} ms")
    r = correlation([w["n_letters"] for w in kept], [w["td"] for w in kept])
    print(f"  Word length effect: r(letters, TD) = {r:.2f} "
          "(should be clearly positive in real reading)")

    truth = true_sequences(events, windows)
    if truth:
        print("\n  Autopilot check: detected vs scripted (true) measures, "
              "all words")
        detected, scripted = [], []
        for trial, true_seq in truth.items():
            n = len(trials[trial]["regions"].split(" | "))
            found = word_measures(sequences[trial], n)
            real = word_measures(true_seq, n)
            for word in range(n):
                detected.append(found[word])
                scripted.append(real[word])
        skip_agree = np.mean([(d["n_fix"] > 0) == (s["n_fix"] > 0)
                              for d, s in zip(detected, scripted)])
        print(f"    fixated/skipped agreement: {100 * skip_agree:.0f}%")
        for name in MEASURES:
            d = [w[name] for w in detected]
            s = [w[name] for w in scripted]
            print(f"    {name.upper()}: r = {correlation(d, s):.2f}, "
                  f"mean abs error = {np.mean(np.abs(np.subtract(d, s))):.0f}"
                  f" ms (true mean {np.mean(s):.0f} ms)")

    print()
    for name in ("fixations.csv", "word_measures.csv", "sentences.csv"):
        print(f"Saved: {os.path.join(folder, name)}")
    for path in dundee_paths + [excel_path]:
        print(f"Saved: {path}")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def readme_lines(session, args):
    return [
        f"Tamil reading study - participant {session['participant']}",
        f"Screen {session['screen'][0]} x {session['screen'][1]}, font "
        f"{session['font_size']} px, gaze source {session['gaze_source']}",
        "",
        "Sheets",
        "  word_measures_clean  FFD/FPD/RPD/TD (ms) per fixated word after "
        "the noise filter, in WNUM order (like dundee_rt.csv)",
        "  word_measures_all    every word of every sentence with the "
        "filter flags (sentence_ok, outlier, keep)",
        "  sentences_answers    per sentence: reading time (sentence on -> "
        "ENTER), gaze quality, question, answer, correct",
        "  RT_dataset1P         Dundee MA1P: events in time order",
        "  RT_dataset2P         Dundee MA2P: words in text order, skipped "
        "words TEXT = -99",
        "  per-word-info        Dundee TX01WRDP: word layout and lengths",
        "  fixations            every detected fixation, with noise label",
        "",
        "Measures (same definitions as the English Dundee data)",
        "  FFD  first fixation duration",
        "  FPD  first-pass duration: first fixation + directly following "
        "ones on the word",
        "  RPD  regression-path duration: fixations on the word before the "
        "eye first moves to a word further right",
        "  TD   total duration: all fixations on the word",
        "",
        "Noise filter",
        f"  1. Gaze smoothed with a running median of 3 samples",
        f"  2. Fixations shorter than {args.min_dur:.0f} ms are not "
        f"counted; longer than {args.max_fix:.0f} ms are dropped",
        f"  3. Sentences excluded if valid gaze < {100 * args.min_valid:.0f}%"
        f" or > {100 * args.max_off_text:.0f}% of fixation time off the text",
        f"  4. Word values above mean + {args.outlier_sd:g} SD (per measure, "
        f"this reader) marked as outliers",
        "  keep = 1 when the sentence is ok and the word is no outlier",
    ]


if __name__ == "__main__":
    main()

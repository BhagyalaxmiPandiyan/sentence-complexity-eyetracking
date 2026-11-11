
"""
Merge participants' results into one Tamil eye-tracking corpus laid out
like the Dundee Corpus folder in phase1_replication\\Dataset\\dundee_corpus,
plus one Excel workbook with everything (researcher only).

Input: the results zips participants sent back (P07_results_*.zip),
session folders from data\\, or folders containing either. Each session
must have been analysed (run_session.bat does this; otherwise run
analyze_reading.py on the folder first).

Readers get Dundee subject letters a, b, c, ... in participant-code
order (A, B, ... after z). If one participant has several sessions, the
newest completed one is used.

Output folder (default tamil_eye_corpus\\):
    RT_dataset1P\\sa01ma1p.dat ...   events in time order, one per reader
    RT_dataset2P\\sa01ma2p.dat ...   words in text order, one per reader
    per-word-info\\tx01wrdp.dat      word info (layout of the first reader)
    per-word-info-by-reader\\        each reader's own layout (line breaks
                                     depend on their screen)
    Sentences\\sent1.txt             the sentences, punctuation split off
    participants.csv                 letter, code, screen, accuracy, ...
    tamil_rt.csv                     cleaned FFD/FPD/RPD/TD per word per
                                     reader, columns of phase 1 dundee_rt.csv
    tamil_eye_corpus.xlsx            all tables as Excel sheets
    README.txt

Usage:
    .\\.venv\\Scripts\\python.exe build_tamil_corpus.py results
    .\\.venv\\Scripts\\python.exe build_tamil_corpus.py E:\\collected --out E:\\tamil_eye_corpus
"""

import argparse
import csv
import glob
import io
import json
import os
import shutil
import string
import zipfile

import numpy as np

import excel_export
from analyze_reading import MEASURES
from dundee_format import CLOSING, MA1_HEADER, MA2_HEADER, OPENING, \
    TEXT_FILE, WRDP_HEADER

SESSION_FILES = ("dundee_ma1p.dat", "dundee_ma2p.dat", "dundee_wrdp.dat",
                 "session.json", "word_measures.csv", "sentences.csv")
LETTERS = string.ascii_lowercase + string.ascii_uppercase


def find_sessions(paths):
    """Yield (name, {file name: text}) for every analysed session."""
    for path in paths:
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "sentences.csv")):
                files = {}
                for name in SESSION_FILES:
                    with open(os.path.join(path, name),
                              encoding="utf-8-sig") as file:
                        files[name] = file.read()
                yield os.path.basename(os.path.normpath(path)), files
                continue
            inner = sorted(glob.glob(os.path.join(path, "*.zip"))
                           + [p for p in glob.glob(os.path.join(path, "*"))
                              if os.path.isdir(p)])
            yield from find_sessions(inner)
        elif path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                folders = {os.path.dirname(name) for name in archive.namelist()
                           if name.endswith("/sentences.csv")}
                for folder in sorted(folders):
                    yield os.path.basename(folder), {
                        name: archive.read(f"{folder}/{name}").decode(
                            "utf-8-sig") for name in SESSION_FILES}
            if not folders:
                print(f"  skipped {path}: no analysed session inside "
                      "(run analyze_reading.py on it first)")


def read_table(text, header=True):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    return lines[1:] if header else lines


def read_dicts(text):
    return list(csv.DictReader(io.StringIO(text)))


def number(value):
    """Text fields back to int / float where possible (for Excel)."""
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def mean(values):
    return round(float(np.mean(values)), 1) if values else ""


def tokenised(words):
    """Dundee Sentences style: punctuation split off with spaces."""
    tokens = []
    for word in words:
        core = word.lstrip(OPENING)
        bare = core.rstrip(CLOSING)
        tokens += list(word[:len(word) - len(core)]) + [bare] + list(
            core[len(bare):])
    return " ".join(t for t in tokens if t)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+",
                        help="results zips, session folders, or folders "
                             "containing them")
    parser.add_argument("--out", default="tamil_eye_corpus")
    args = parser.parse_args()

    # Newest completed session per participant
    chosen = {}
    for name, files in find_sessions(args.inputs):
        info = json.loads(files["session.json"])
        if info.get("gaze_source") != "webcam":
            print(f"  note: {name} used {info.get('gaze_source')} gaze "
                  "(test data)")
        code = info["participant"]
        key = (bool(info.get("completed")), name)
        if code not in chosen or key > chosen[code][0]:
            chosen[code] = (key, name, files, info)
    if not chosen:
        raise SystemExit("No analysed sessions found.")
    if len(chosen) > len(LETTERS):
        raise SystemExit(f"At most {len(LETTERS)} readers are supported.")

    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    for folder in ("RT_dataset1P", "RT_dataset2P", "per-word-info",
                   "per-word-info-by-reader", "Sentences"):
        os.makedirs(os.path.join(out, folder))

    text = f"{TEXT_FILE:02d}"
    rt_rows, participant_rows, all_words, all_sentences = [], [], [], []
    ma1_rows, ma2_rows, word_info = [], [], None
    for subj_id, code in enumerate(sorted(chosen)):
        _, name, files, info = chosen[code]
        letter = LETTERS[subj_id]
        for source, folder, target in (
                ("dundee_ma1p.dat", "RT_dataset1P", f"s{letter}{text}ma1p.dat"),
                ("dundee_ma2p.dat", "RT_dataset2P", f"s{letter}{text}ma2p.dat"),
                ("dundee_wrdp.dat", "per-word-info-by-reader",
                 f"s{letter}{text}wrdp.dat")):
            with open(os.path.join(out, folder, target), "w",
                      encoding="utf-8", newline="\n") as file:
                file.write(files[source])
        if word_info is None:
            word_info = files["dundee_wrdp.dat"]
        ma1_rows += [[letter] + list(map(number, r))
                     for r in read_table(files["dundee_ma1p.dat"])]
        ma2_rows += [[letter] + list(map(number, r))
                     for r in read_table(files["dundee_ma2p.dat"])]

        olen = {int(r[12]): int(r[7]) for r in
                read_table(files["dundee_wrdp.dat"], header=False)}
        words = [w for w in read_dicts(files["word_measures.csv"])
                 if w["phase"] == "main"]
        for w in sorted(words, key=lambda w: int(w["wnum"])):
            w["subject"] = letter
            all_words.append(w)
            if w["fixated"] == "1" and w["keep"] == "1":
                wnum = int(w["wnum"])
                rt_rows.append([TEXT_FILE, wnum, subj_id,
                                TEXT_FILE * 10000 + int(w["screen"]),
                                w["word"], olen[wnum]]
                               + [int(round(float(w[k]))) for k in MEASURES])

        sentences = [s for s in read_dicts(files["sentences.csv"])
                     if s["phase"] == "main"]
        for s in sentences:
            s["subject"] = letter
        all_sentences += sentences
        answered = [s for s in sentences if s["answer_correct"] != ""]
        validation = info.get("validation", {})
        participant_rows.append([
            letter, code, name, info.get("gaze_source"),
            "x".join(map(str, info.get("screen", []))), info.get("font_size"),
            info.get("completed"), len(sentences),
            sum(s["sentence_ok"] == "1" for s in sentences),
            round(np.mean([int(s["answer_correct"]) for s in answered]), 3)
            if answered else "",
        ] + [validation.get(label, {}).get("median_error", "")
             for label in ("start", "end")])

    with open(os.path.join(out, "per-word-info", f"tx{text}wrdp.dat"), "w",
              encoding="utf-8", newline="\n") as file:
        file.write(word_info)
    wrdp = [list(map(number, r)) for r in read_table(word_info, header=False)]

    screens = {}
    for row in wrdp:
        screens.setdefault(row[2], []).append(row[0])
    with open(os.path.join(out, "Sentences", f"sent{TEXT_FILE}.txt"), "w",
              encoding="utf-8", newline="\n") as file:
        for screen in sorted(screens):
            file.write(tokenised(screens[screen]) + "\n")

    # Averages across readers (cleaned values only)
    by_wnum, by_item = {}, {}
    for w in all_words:
        by_wnum.setdefault(int(w["wnum"]), []).append(w)
        by_item.setdefault(w["item_id"], []).append(w)
    word_means = []
    for row in wrdp:
        mine = by_wnum.get(row[12], [])
        good = [w for w in mine if w["sentence_ok"] == "1"]
        kept = [w for w in good if w["keep"] == "1" and w["fixated"] == "1"]
        word_means.append(
            [row[12], row[2], mine[0]["item_id"] if mine else "",
             mine[0]["condition"] if mine else "", row[0], row[7],
             len(good), len(kept),
             round(1 - sum(w["fixated"] == "1" for w in good)
                   / len(good), 3) if good else ""]
            + [mean([float(w[k]) for w in kept]) for k in MEASURES])

    sentence_means = []
    item_of = {w[1]: w[2] for w in word_means}  # screen -> item_id
    for screen in sorted(screens):
        item_id = item_of[screen]
        rows = [s for s in all_sentences if s["item_id"] == item_id]
        ok = [s for s in rows if s["sentence_ok"] == "1"]
        kept = [w for w in by_item.get(item_id, [])
                if w["keep"] == "1" and w["fixated"] == "1"]
        answered = [s for s in rows if s["answer_correct"] != ""]
        sentence_means.append(
            [screen, item_id, rows[0]["condition"] if rows else "",
             len(screens[screen]), tokenised(screens[screen]),
             len(rows), len(ok),
             mean([float(s["reading_time_ms"]) for s in ok])]
            + [mean([float(w[k]) for w in kept]) for k in MEASURES]
            + [round(np.mean([int(s["answer_correct"]) for s in answered]),
                     3) if answered else ""])

    rt_header = ["text_id", "wnum", "subj_id", "sent_id", "word",
                 "wlen"] + MEASURES
    participant_header = [
        "letter", "participant", "session", "gaze_source", "screen",
        "font_size", "completed", "sentences", "sentences_ok",
        "question_accuracy", "validation_error_start_px",
        "validation_error_end_px"]
    word_means_header = ["WNUM", "SCREEN", "ITEM_ID", "BAND", "WORD",
                         "WLEN", "READERS", "READERS_KEPT", "SKIP_RATE"] + [
                         f"MEAN_{k.upper()}" for k in MEASURES]
    sentence_means_header = [
        "SCREEN", "ITEM_ID", "BAND", "N_WORDS", "SENTENCE", "READERS",
        "READERS_OK", "MEAN_READING_TIME_MS"] + [
        f"MEAN_{k.upper()}" for k in MEASURES] + ["QUESTION_ACCURACY"]
    word_header = ["subject"] + [k for k in all_words[0] if k != "subject"]
    sentence_header = ["subject"] + [k for k in all_sentences[0]
                                     if k != "subject"]

    write_csv(os.path.join(out, "tamil_rt.csv"), rt_header, rt_rows)
    write_csv(os.path.join(out, "participants.csv"), participant_header,
              participant_rows)
    readme = README.format(n=len(chosen), screens=len(screens))
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as file:
        file.write(readme)
    excel_path = os.path.join(out, "tamil_eye_corpus.xlsx")
    excel_export.write_workbook(excel_path, [
        ("README", None, readme.splitlines()),
        ("participants", participant_header, participant_rows),
        ("tamil_rt", rt_header, rt_rows),
        ("word_means", word_means_header, word_means),
        ("sentence_means", sentence_means_header, sentence_means),
        ("word_measures_all", word_header,
         [[number(w[k]) for k in word_header] for w in all_words]),
        ("sentences_answers", sentence_header,
         [[number(s[k]) for k in sentence_header] for s in all_sentences]),
        ("RT_dataset1P", ["SUBJ"] + MA1_HEADER, ma1_rows),
        ("RT_dataset2P", ["SUBJ"] + MA2_HEADER, ma2_rows),
        ("per-word-info", WRDP_HEADER, wrdp),
    ])

    print(f"{len(chosen)} readers, {len(screens)} sentences, "
          f"{len(rt_rows)} cleaned word observations")
    for row in participant_rows:
        print(f"  s{row[0]}: {row[1]} ({row[2]}), {row[8]}/{row[7]} "
              f"sentences ok, question accuracy {row[9]}, validation "
              f"error {row[10]} / {row[11]} px")
    print(f"Corpus written to {os.path.abspath(out)}")
    print(f"Excel workbook: {os.path.abspath(excel_path)}")


README = """Tamil webcam eye-tracking corpus, Dundee Corpus layout

{n} native Tamil readers read {screens} Tamil news sentences (UD Tamil-TTB,
Dinamani; CC BY-NC-SA 3.0) one at a time while a webcam estimated gaze
(MediaPipe face mesh + per-reader calibration). Each sentence stayed on
screen until the reader pressed ENTER. Sentence order was randomised per
reader. Fixations were detected with I-DT.

Folders and files follow the Dundee Corpus (Kennedy 2003,
Corpus_Notes_V2.doc):
  RT_dataset1P   s?01ma1p.dat  events in the order they happened
                 WORD TEXT LINE OLEN WLEN XPOS WNUM FDUR OBLP WDLP LAUN TXFR
  RT_dataset2P   s?01ma2p.dat  words in text order, a row per fixation,
                 skipped words with TEXT = -99
                 WORD TEXT LINE OLEN WLEN XPOS WNUM FDUR OBLP WDLP FXNO TXFR
  per-word-info  tx01wrdp.dat  14 columns as TX01WRDP.DAT
  Sentences      sent1.txt     one sentence per line, punctuation split off

Noise filter (applied to tamil_rt.csv and the means; the .dat files keep
every fixation except the too-long ones):
  - gaze smoothed with a running median of 3 samples
  - fixations shorter than 80 ms not counted, longer than 1200 ms dropped
  - sentences excluded when valid gaze < 70% or > 50% of fixation time
    is off the text (sentence_ok = 0)
  - word values above mean + 3 SD per measure per reader are outliers
  - keep = 1 when the sentence is ok and the word is no outlier

tamil_eye_corpus.xlsx sheets:
  participants       reader letter, screen, sentences ok, answer accuracy,
                     gaze validation error
  tamil_rt           cleaned FFD/FPD/RPD/TD per fixated word per reader
                     (same columns as phase 1 dundee_rt.csv)
  word_means         per word: mean FFD/FPD/RPD/TD over readers, skip rate
  sentence_means     per sentence: mean reading time and measures, accuracy
  word_measures_all  every word, every reader, with the filter flags
  sentences_answers  every sentence, every reader: reading time, gaze
                     quality, question, answer, correct
  RT_dataset1P/2P    the Dundee event and word files, all readers (SUBJ)
  per-word-info      the Dundee word file

Differences from the English Dundee data:
  - One Text File (01); every sentence is its own screen (TEXT 1-{screens},
    in sentence-id order, the same for all readers).
  - Lengths and positions count Tamil letters (graphemes).
  - XPOS/OBLP/WDLP are estimated from the pixel position inside the word.
  - LINE depends on the reader's screen width (see
    per-word-info-by-reader); tx01wrdp.dat holds the first reader's layout.
  - *Blink rows are gaps without valid webcam gaze (blink or face lost);
    *Offscreen rows are fixations on no word.
  - Files are UTF-8 (open with encoding="utf-8", not latin-1).
  - TXFR is the frequency of the word in these {screens} sentences.
"""


if __name__ == "__main__":
    main()

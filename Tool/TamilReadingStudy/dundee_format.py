
"""
Write one natural_reading.py session in the Dundee Corpus file formats
(Kennedy 2003, Corpus_Notes_V2.doc), so the phase 1 code that reads the
English Dundee files can read the Tamil data the same way.

Mapping to Dundee:
    Text File  = the whole Tamil sentence set (TEXT_FILE, "01")
    Screen     = one sentence; screen number = its rank in item_id order
                 (S001 -> 1 ... S100 -> 100), the same for every reader
                 whatever order the sentences were shown in
    WNUM       = word index across all sentences, in screen order
    Lengths    = Tamil letters (graphemes), not Unicode code points
    XPOS, OBLP = character positions estimated from the pixel position
                 of the fixation inside the word box
    TXFR       = how often the word occurs in all the sentences
    *Blink     = a gap without valid gaze (blink or face lost)
    *Offscreen = a fixation on no word of any line
Practice sentences are left out, as Dundee has none.

Files (whitespace separated, UTF-8):
    dundee_ma1p.dat  events in the order they happened
        WORD TEXT LINE OLEN WLEN XPOS WNUM FDUR OBLP WDLP LAUN TXFR
    dundee_ma2p.dat  words in text order, one row per fixation,
                     skipped words once with TEXT = -99
        WORD TEXT LINE OLEN WLEN XPOS WNUM FDUR OBLP WDLP FXNO TXFR
    dundee_wrdp.dat  per-word info (14 columns, no header) as TX01WRDP.DAT
"""

import os
from collections import Counter

from tamil_render import count_graphemes

TEXT_FILE = 1
MISSING = -99
# Dundee punctuation codes (Corpus_Notes_V2 appendix)
PUNCT_CODES = {".": 1, ",": 2, ";": 3, ":": 4, "?": 5, "!": 6}
OPENING = "\"'‘“(["
CLOSING = ".,;:?!\"'’”)]"

MA1_HEADER = ["WORD", "TEXT", "LINE", "OLEN", "WLEN", "XPOS", "WNUM",
              "FDUR", "OBLP", "WDLP", "LAUN", "TXFR"]
MA2_HEADER = MA1_HEADER[:10] + ["FXNO", "TXFR"]
# TX01WRDP.DAT has no header line; these names are for spreadsheets
WRDP_HEADER = ["WORD", "TEXT", "SCREEN", "LINE", "LINE_POS", "SERIAL",
               "START_POS", "OLEN", "WLEN", "PUNCT", "OMARKS", "EMARKS",
               "WNUM", "TXFR"]


def split_marks(token):
    """'word,' -> ('word', opening marks, closing marks)."""
    core = token.lstrip(OPENING)
    omarks = len(token) - len(core)
    bare = core.rstrip(CLOSING)
    return bare, omarks, len(core) - len(bare)


def word_table(screens):
    """Per-word info for all screens. screens: [(item_id, words, boxes)]
    already in screen order. Returns a list of dicts (one per word)."""
    bare_words = [split_marks(w)[0] for _, words, _ in screens for w in words]
    frequency = Counter(bare_words)

    table, wnum = [], 0
    for screen, (item_id, words, boxes) in enumerate(screens, start=1):
        line_tops = sorted({box[1] for box in boxes})
        position = Counter()
        start = {}  # line -> next initial-letter position
        for serial, (token, box) in enumerate(zip(words, boxes), start=1):
            wnum += 1
            line = line_tops.index(box[1]) + 1
            position[line] += 1
            bare, omarks, emarks = split_marks(token)
            olen = count_graphemes(token)
            # Dundee: first word at 0, later words at the position of
            # the space before them (Are 0, tourists 4, enticed 13).
            initial = start.get(line, 0)
            start[line] = initial + olen + 1
            closing = token[len(token) - emarks:] if emarks else ""
            table.append({
                "word": token, "bare": bare, "item_id": item_id,
                "screen": screen, "line": line,
                "line_position": position[line], "serial": serial,
                "initial": initial, "olen": olen,
                "wlen": count_graphemes(bare),
                "punct": PUNCT_CODES.get(closing[:1], 0) if closing else 0,
                "omarks": omarks, "emarks": emarks, "wnum": wnum,
                "txfr": frequency[bare], "box": box,
            })
    return table


def landing(info, fx):
    """OBLP, WDLP, XPOS of a fixation at pixel x on this word."""
    x1, _, x2, _ = info["box"]
    if fx < x1:
        oblp = 0  # on the space before the word
    else:
        share = (fx - x1) / max(x2 - x1, 1)
        oblp = min(info["olen"], int(share * info["olen"]) + 1)
    wdlp = max(0, min(oblp - info["omarks"], info["wlen"]))
    return oblp, wdlp, info["initial"] + oblp


def row_text(values):
    word, *numbers = values
    return f"{word:<22}" + "".join(f"{n:>6}" for n in numbers)


def write_session(folder, screens, events):
    """screens: [(item_id, words, boxes)] in screen order.
    events: [(item_id, kind, duration_ms, fx, word_index)] in the order
    they happened; kind is "fix" or "blink"; word_index is None for a
    fixation on no word.

    Returns (paths written, {"ma1p"|"ma2p"|"wrdp": (header, rows)},
    per-word table from word_table)."""
    table = word_table(screens)
    by_word = {(row["item_id"], row["serial"] - 1): row for row in table}
    screen_of = {item_id: n for n, (item_id, _, _) in enumerate(screens, 1)}

    ma1, fixations_of = [], {}
    last_screen = None
    previous = None  # (line, xpos) of the last event if it was on a word
    for fxno, (item_id, kind, duration, fx, word) in enumerate(events, 1):
        screen = screen_of[item_id]
        duration = int(round(duration))
        first_on_screen = screen != last_screen
        last_screen = screen
        if kind == "blink" or word is None:
            label = "*Blink" if kind == "blink" else "*Offscreen"
            ma1.append([label, screen] + [MISSING] * 5 + [duration]
                       + [MISSING] * 4)
            previous = None
            continue
        info = by_word[(item_id, word)]
        oblp, wdlp, xpos = landing(info, fx)
        # LAUN as in Dundee: previous XPOS minus this XPOS (Are 1 ->
        # tourists 6 gives -5); 0 for the first fixation on a screen,
        # -99 after a blink or a change of line.
        if first_on_screen:
            launch = 0
        elif previous is None or previous[0] != info["line"]:
            launch = MISSING
        else:
            launch = previous[1] - xpos
        previous = (info["line"], xpos)
        ma1.append([info["word"], screen, info["line"], info["olen"],
                    info["wlen"], xpos, info["wnum"], duration, oblp, wdlp,
                    launch, info["txfr"]])
        fixations_of.setdefault(info["wnum"], []).append(
            [info["word"], screen, info["line"], info["olen"], info["wlen"],
             xpos, info["wnum"], duration, oblp, wdlp, fxno, info["txfr"]])

    ma2 = []
    for info in table:
        rows = fixations_of.get(info["wnum"])
        if rows:
            ma2.extend(rows)
        else:
            ma2.append([info["word"], MISSING, 0, 0, 0, 0, info["wnum"],
                        0, 0, 0, 0, 0])

    wrdp = [[info["word"], TEXT_FILE, info["screen"], info["line"],
             info["line_position"], info["serial"], info["initial"],
             info["olen"], info["wlen"], info["punct"], info["omarks"],
             info["emarks"], info["wnum"], info["txfr"]] for info in table]
    sheets = {"ma1p": (MA1_HEADER, ma1), "ma2p": (MA2_HEADER, ma2),
              "wrdp": (WRDP_HEADER, wrdp)}

    paths = {name: os.path.join(folder, f"dundee_{name}.dat")
             for name in sheets}
    for name, (header, rows) in sheets.items():
        with open(paths[name], "w", encoding="utf-8", newline="\n") as file:
            if name != "wrdp":  # Dundee word files have no header line
                file.write(row_text(header) + "\n")
            file.writelines(row_text(r) + "\n" for r in rows)
    return list(paths.values()), sheets, table

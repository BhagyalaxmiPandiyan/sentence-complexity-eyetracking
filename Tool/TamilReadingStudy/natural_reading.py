
"""
Natural (whole-sentence) reading with webcam gaze, for word-level
eye-tracking measures like the English Dundee data in phase 1.

The whole sentence is shown at once, each word in its own
box. The reader reads normally and presses ENTER when finished. Gaze
samples are logged; analyze_reading.py turns them into fixations and
FFD / FPD / RPD / TD per word.

Everything else (instructions, 9-point validation, drift check before
each sentence, comprehension questions, output files) is reused from
hybrid_spr.py, so the same calibration model and package_results.py work.

The "|" region marks in the items file are ignored: every word becomes
its own box. critical_region is converted to the word indices it covers.
Long sentences wrap onto several left-aligned lines (like the Dundee
texts), with wide line spacing so webcam gaze can tell the lines apart.

Default items: items\\tamil100_items.csv (the 100 Tamil news sentences,
in 4 blocks with a break and 9-point validation between blocks).

Usage (PowerShell):
    Real session (after gaze_calibration.py --label P01):
        .\\.venv\\Scripts\\python.exe natural_reading.py --participant P01 --list 0
    Try it with the mouse as "gaze":
        .\\.venv\\Scripts\\python.exe natural_reading.py --participant TEST --gaze mouse --show-gaze
    Automatic test with a simulated reader (keep --speed 1, so the
    30 Hz gaze stream has realistic fixation lengths):
        .\\.venv\\Scripts\\python.exe natural_reading.py --participant AUTO --gaze autopilot --windowed --data-dir data_simulated
"""

import argparse
import ctypes
import json
import os

import cv2
import numpy as np

import hybrid_spr
from hybrid_spr import (
    KEY_SPACE, KEYS_NO, KEYS_YES, MODEL_FILE, DATA_DIR, Session,
)
from tamil_render import count_graphemes

ITEMS_FILE = os.path.join("items", "tamil100_items.csv")
KEY_ENTER = 13          # ends the sentence (reader has finished it)
MARGIN_PX = 80          # left/right and top/bottom screen margin
LINE_SPACING = 3.0      # line pitch as a multiple of the text height

hybrid_spr.INSTRUCTIONS = [
    ("வாசிப்புச் சோதனை", "Reading task"),
    ("முழு வாக்கியமும் ஒரே நேரத்தில் தோன்றும். இயல்பாகப் படியுங்கள்.",
     "The whole sentence appears at once. Read it normally."),
    ("படித்து முடித்ததும் ENTER அழுத்தவும்.",
     "When you have finished reading, press ENTER."),
    ("புள்ளி தோன்றும்போது அதைப் பார்த்து SPACE அழுத்தவும்.",
     "When a dot appears, look at it and press SPACE."),
    ("தலையை அசைக்காமல் இருங்கள்.", "Keep your head still."),
    ("தொடங்க SPACE அழுத்தவும்.", "Press SPACE to start."),
]


def split_words(item):
    """Words of an item, and the word indices of its critical region."""
    words, critical = [], []
    for number, region in enumerate(item["regions"], start=1):
        for word in region.split():
            if number == item["critical_region"]:
                critical.append(len(words))
            words.append(word)
    return words, critical


def layout_words(words, renderer, screen_w, screen_h, gap_px,
                 line_spacing=LINE_SPACING):
    """Wrap words onto left-aligned lines, block centred vertically.

    Returns one (x1, y1, x2, y2) box per word.
    Raises ValueError if the lines do not fit on the screen.
    """
    sizes = [renderer.measure(word) for word in words]
    line_h = max(h for _, h in sizes)
    pitch = int(line_h * line_spacing)
    right = screen_w - MARGIN_PX

    lines, x = [[]], MARGIN_PX
    for index, (width, _) in enumerate(sizes):
        if width > right - MARGIN_PX:
            raise ValueError(f"Word {words[index]} is wider than the screen.")
        if lines[-1] and x + width > right:
            lines.append([])
            x = MARGIN_PX
        lines[-1].append((index, x, width))
        x += width + gap_px

    block_h = pitch * (len(lines) - 1) + line_h
    if block_h > screen_h - 2 * MARGIN_PX:
        raise ValueError(
            f"Sentence needs {len(lines)} lines ({block_h} px); screen "
            f"allows {screen_h - 2 * MARGIN_PX} px. Use a smaller "
            "--font-size or --line-spacing."
        )
    top = (screen_h - block_h) // 2
    boxes = [None] * len(words)
    for number, line in enumerate(lines):
        y = top + number * pitch
        for index, x, width in line:
            boxes[index] = (x, y, x + width, y + line_h)
    return boxes


def reading_plan(boxes, words, rng):
    """Simulated reader: one scripted fixation sequence for a sentence.

    Returns [(word_index, x, y, duration_s)]. Longer words get longer
    and more fixations, short words are sometimes skipped, and some
    words trigger a regression to an earlier word.
    """
    plan = []
    word = 0
    while word < len(words):
        x1, y1, x2, y2 = boxes[word]
        letters = count_graphemes(words[word])
        if letters <= 3 and word > 0 and rng.random() < 0.25:
            word += 1  # skip short word
            continue
        n_fix = 2 if letters >= 8 and rng.random() < 0.5 else 1
        for k in range(n_fix):
            # Landing slightly left of the word centre
            x = x1 + (x2 - x1) * (0.35 + 0.3 * k) + rng.gauss(0, 8)
            duration = 0.16 + 0.012 * letters + abs(rng.gauss(0, 0.04))
            plan.append((word, x, (y1 + y2) / 2, duration))
        if word >= 2 and rng.random() < 0.15:
            back = rng.randrange(0, word)
            bx1, by1, bx2, by2 = boxes[back]
            plan.append((back, (bx1 + bx2) / 2, (by1 + by2) / 2,
                         0.18 + abs(rng.gauss(0, 0.04))))
        word += 1
    return plan


class NaturalSession(Session):
    N_BLOCKS = 4

    def layout_trials(self):
        # One box per word, wrapped onto lines (called again with a
        # smaller font when a sentence does not fit).
        self.min_region_width = None
        for trial in self.practice + self.main:
            if "words" not in trial:
                trial["words"], trial["critical_words"] = split_words(trial)
                trial["regions"] = trial["words"]
                trial["critical_region"] = None
            try:
                trial["boxes"] = layout_words(
                    trial["words"], self.renderer, self.screen_w,
                    self.screen_h, self.args.gap, self.args.line_spacing)
            except ValueError as error:
                raise ValueError(f"Item {trial['item_id']}: {error}") from None

    def run_trial(self, trial, phase):
        self.trial_index = self.trials_run
        self.trials_run += 1
        self.phase = phase
        words, boxes = trial["words"], trial["boxes"]

        # Drift check: look at the dot where the first word starts
        self.offset = (0.0, 0.0)
        dot = (boxes[0][0] + 8, (boxes[0][1] + boxes[0][3]) // 2)
        canvas = self.blank()
        cv2.circle(canvas, dot, 10, hybrid_spr.DOT, -1)
        self.present(canvas, "drift_on")
        if self.autopilot:
            self.autopilot.schedule(self.now(), [
                (0.0, ("look", dot)), (0.7, ("key", KEY_SPACE)),
            ])
        _, t_space = self.wait_for({KEY_SPACE})
        self.log_event("drift_space", t=t_space)

        first = t_space - hybrid_spr.DRIFT_WINDOW_S * self.time_scale
        pts = np.array([(x, y) for t, x, y, valid in self.recent
                        if valid and first <= t <= t_space])
        if len(pts) >= hybrid_spr.MIN_DRIFT_SAMPLES:
            med = np.median(pts, axis=0)
            self.offset = (float(med[0] - dot[0]), float(med[1] - dot[1]))

        # Whole sentence at once
        canvas = self.blank()
        for text, (x1, y1, _, _) in zip(words, boxes):
            self.renderer.draw(canvas, text, x1, y1)
        self.present(canvas, "sentence_on")
        if self.autopilot:
            # Gaze jumps instantly, so each scripted fixation lasts
            # exactly until the next one; "dur" is in real seconds.
            steps, t = [], 0.0
            for word, x, y, duration in reading_plan(
                    boxes, words, self.autopilot.rng):
                steps.append((t, ("look", (x, y))))
                steps.append((t, ("truth", json.dumps(
                    {"word": word, "dur": duration / self.args.speed}))))
                t += duration
            steps.append((t, ("key", KEY_ENTER)))
            self.autopilot.schedule(self.now(), steps)
        _, t_key = self.wait_for({KEY_ENTER})
        self.log_event("sentence_off", t=t_key)
        self.log_event("trial_end")

        # Comprehension question
        response, correct, rt = "", "", ""
        if trial["question"]:
            canvas = self.blank()
            width, _ = self.renderer.measure(trial["question"])
            self.renderer.draw(canvas, trial["question"],
                               (self.screen_w - width) // 2,
                               self.screen_h // 2 - 80)
            width, _ = self.renderer.measure(hybrid_spr.TEXT_ANSWER)
            self.renderer.draw(canvas, hybrid_spr.TEXT_ANSWER,
                               (self.screen_w - width) // 2,
                               self.screen_h // 2 + 40, hybrid_spr.GREY)
            self.present(canvas, "question_on")
            t_on = self.now()
            if self.autopilot:
                right = trial["answer"] == "Y"
                if self.autopilot.rng.random() < 0.1:
                    right = not right
                self.autopilot.schedule(t_on, [
                    (0.8, ("key", ord("y") if right else ord("n"))),
                ])
            key, t_key = self.wait_for(KEYS_YES | KEYS_NO)
            response = "Y" if key in KEYS_YES else "N"
            correct = int(response == trial["answer"])
            rt = round((t_key - t_on) * 1000, 1)
            self.log_event("question_response", response, t=t_key)

        # Same columns as hybrid_spr.py; "regions" holds the words and
        # "critical_region" the critical word indices (0-based).
        self.writers["trials"].writerow([
            self.trial_index, phase, trial["item_id"], trial["type"],
            trial["condition"], len(words),
            " ".join(map(str, trial["critical_words"])),
            " | ".join(words), json.dumps(boxes),
            round(self.offset[0], 1), round(self.offset[1], 1), len(pts),
            trial["question"], trial["answer"], response, correct, rt,
        ])
        for file in self.files.values():
            file.flush()

    def run(self):
        super().run()
        print(f"Word measures: .\\.venv\\Scripts\\python.exe "
              f"analyze_reading.py {self.folder}")


def main():
    parser = argparse.ArgumentParser(description="Natural reading + gaze")
    parser.add_argument("--participant", required=True,
                        help="anonymous code, e.g. P01 (no names)")
    parser.add_argument("--list", type=int, default=0,
                        help="Latin-square list number (0, 1, ...)")
    parser.add_argument("--items", default=ITEMS_FILE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--gaze", choices=["webcam", "mouse", "autopilot"],
                        default="webcam")
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--any-model", action="store_true",
                        help="allow a gaze model calibrated under "
                             "another label")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--gap", type=int, default=40,
                        help="pixels between words (wider = easier "
                             "word assignment)")
    parser.add_argument("--line-spacing", type=float, default=LINE_SPACING,
                        help="line pitch as a multiple of the text height "
                             "(wider = easier line assignment)")
    parser.add_argument("--show-gaze", action="store_true",
                        help="draw the gaze point (testing only)")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="autopilot speed-up (keep 1 for analysis)")
    args = parser.parse_args()

    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
    try:
        NaturalSession(args).run()
    finally:
        winmm.timeEndPeriod(1)


if __name__ == "__main__":
    main()


"""
Hybrid cumulative self-paced reading with webcam gaze.
Design: docs/study_design.md

Each sentence is shown on one line in regions. SPACE reveals the next
region; earlier regions stay visible, so the reader can look back.
Key presses give region reading times; webcam gaze shows whether the
reader looks back at earlier regions.

Usage (PowerShell, from D:\\eye-tracker):
    Real session (needs calibration\\gaze_model_v2.joblib from gaze_calibration.py):
        .\\.venv\\Scripts\\python.exe hybrid_spr.py --participant P01 --list 0
    Try the experiment using the mouse as "gaze":
        .\\.venv\\Scripts\\python.exe hybrid_spr.py --participant TEST --gaze mouse --show-gaze
    Fully automatic test run (no person needed):
        .\\.venv\\Scripts\\python.exe hybrid_spr.py --participant AUTO --gaze autopilot --windowed

Keys: SPACE = next, Y/N = answer, C = continue after validation,
Q = quit at validation screen, ESC = abort (data so far is saved).
"""

import argparse
import csv
import ctypes
import json
import os
import random
import time
import zlib
from collections import deque

import cv2
import joblib
import numpy as np

from gaze_runtime import AutopilotGaze, MouseGaze, WebcamGaze
from tamil_render import TextRenderer, draw_sentence, layout_regions

# ---------- Configuration ----------
MODEL_FILE = os.path.join("calibration", "gaze_model_v2.joblib")
ITEMS_FILE = os.path.join("items", "demo_items.csv")
DATA_DIR = "data"

MIN_FONT_SIZE = 24          # smallest font tried on narrow screens
DRIFT_WINDOW_S = 0.5       # gaze before SPACE on the drift dot
MIN_DRIFT_SAMPLES = 5
LOOKBACK_DELAY_S = 0.6      # after last region appears
LOOKBACK_CUE_S = 1.5
VALIDATION_SHOW_S = 1.4
VALIDATION_SKIP_S = 0.5     # ignore the saccade to each point
VALIDATION_POINTS = [
    (x, y) for y in (0.15, 0.5, 0.85) for x in (0.15, 0.5, 0.85)
]

KEY_SPACE = 32
KEY_ESC = 27
KEYS_YES = {ord("y"), ord("Y")}
KEYS_NO = {ord("n"), ord("N")}

WHITE = (240, 240, 240)
GREY = (160, 160, 160)
DOT = (0, 220, 255)

INSTRUCTIONS = [
    ("வாசிப்புச் சோதனை", "Reading task"),
    ("ஒவ்வொரு வாக்கியமும் பகுதி பகுதியாகத் தோன்றும்.",
     "Each sentence appears part by part."),
    ("அடுத்த பகுதியைப் பார்க்க SPACE அழுத்தவும்.",
     "Press SPACE to see the next part."),
    ("தேவைப்பட்டால் முந்தைய பகுதிகளை மீண்டும் படிக்கலாம்.",
     "You may re-read earlier parts if needed."),
    ("புள்ளி தோன்றும்போது அதைப் பார்த்து SPACE அழுத்தவும்.",
     "When a dot appears, look at it and press SPACE."),
    ("ஒரு பகுதியைச் சுற்றி பெட்டி தோன்றினால், அது மறையும் வரை அதைப் பாருங்கள்.",
     "If a box appears around a part, look at it until it disappears."),
    ("தலையை அசைக்காமல் இருங்கள்.", "Keep your head still."),
    ("தொடங்க SPACE அழுத்தவும்.", "Press SPACE to start."),
]
TEXT_LOOK_AT_DOT = "புள்ளியைப் பாருங்கள்  /  Look at the dot"
TEXT_ANSWER = "Y = ஆம்        N = இல்லை"
TEXT_PRACTICE_DONE = ("பயிற்சி முடிந்தது. தொடங்க SPACE அழுத்தவும்.",
                      "Practice finished. Press SPACE to begin.")
TEXT_BREAK = ("இடைவேளை. ஓய்வெடுத்த பின் SPACE அழுத்தவும்.",
              "Break. Press SPACE when ready.")
TEXT_END = ("முடிந்தது. நன்றி!", "Finished. Thank you!")


class Aborted(Exception):
    pass


# ---------- ITEMS ----------
def load_items(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    items = []
    for row in rows:
        regions = [part.strip() for part in row["regions"].split("|")]
        if not all(regions):
            raise ValueError(f"Item {row['item_id']}: empty region")
        items.append({
            "item_id": row["item_id"].strip(),
            "type": row["type"].strip(),
            "condition": row.get("condition", "").strip(),
            "regions": regions,
            "critical_region": int(row["critical_region"])
            if row.get("critical_region", "").strip() else None,
            "question": row.get("question", "").strip(),
            "answer": row.get("answer", "").strip().upper(),
        })
    return items


def build_trials(items, list_number, seed):
    """Latin-square choice of one version per experimental item."""
    practice = [it for it in items if it["type"] == "practice"]

    versions = {}
    for it in items:
        if it["type"] == "experimental":
            versions.setdefault(it["item_id"], []).append(it)

    chosen = []
    for n, item_id in enumerate(sorted(versions)):
        options = sorted(versions[item_id], key=lambda it: it["condition"])
        chosen.append(options[(n + list_number) % len(options)])

    main = chosen + [
        it for it in items if it["type"] in ("filler", "lookback", "sentence")
    ]
    random.Random(seed).shuffle(main)
    return practice, main


# ---------- AUTOPILOT (automated testing only) ----------
class Autopilot:
    """Scripted reader: moves synthetic gaze and presses keys."""

    def __init__(self, gaze, speed, seed, log_event):
        self.gaze = gaze
        self.speed = speed
        self.rng = random.Random(seed)
        self.log_event = log_event
        self.plan = []

    def schedule(self, now, steps):
        self.plan = sorted(
            ((now + delay / self.speed, action) for delay, action in steps),
            key=lambda step: step[0],
        )

    def key(self, now):
        pressed = None
        while self.plan and self.plan[0][0] <= now:
            _, (kind, value) = self.plan.pop(0)
            if kind == "look":
                self.gaze.target = value
            elif kind == "key":
                pressed = value
            elif kind == "truth":
                self.log_event("autopilot_truth", value)
        return pressed

    def reading_steps(self, boxes, revealed, trial):
        newest = centre(boxes[revealed - 1])
        base = self.rng.uniform(0.5, 0.9)
        if (trial["condition"] == "relative_participle"
                and revealed == trial["critical_region"]):
            base *= 1.4  # simulated complexity effect
        steps = [(0.0, ("look", newest))]
        if revealed > 1 and self.rng.random() < 0.3:
            steps += [
                (0.25, ("truth", f"regression region={revealed}")),
                (0.25, ("look", centre(boxes[0]))),
                (0.85, ("look", newest)),
            ]
            base += 0.7
        steps.append((base, ("key", KEY_SPACE)))
        return steps


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ---------- SESSION ----------
class Session:
    N_BLOCKS = 2  # main trials are split by N_BLOCKS - 1 breaks

    def __init__(self, args):
        self.args = args
        self.t0 = time.perf_counter()
        self.phase = "setup"
        self.trial_index = -1       # -1 outside trials (logged)
        self.trials_run = 0         # unique trial numbering
        self.offset = (0.0, 0.0)
        self.recent = deque(maxlen=900)
        self.pending_keys = deque()
        self.last_gaze = None
        self.last_overlay = 0.0
        self.base_canvas = None
        self.validation = {}
        self.time_scale = 1.0 / args.speed if args.gaze == "autopilot" else 1.0

        # Screen size: from the gaze model (webcam) or Windows.
        self.bundle = None
        if args.gaze == "webcam":
            if not os.path.exists(args.model):
                raise FileNotFoundError(
                    f"{args.model} not found. Run gaze_calibration.py "
                    "first, or use --gaze mouse."
                )
            self.bundle = joblib.load(args.model)
            self.screen_w = self.bundle["screen_width"]
            self.screen_h = self.bundle["screen_height"]

            # Never use another person's calibration by accident.
            model_session = self.bundle["session_id"]
            if (not model_session.endswith("_" + args.participant)
                    and not args.any_model):
                raise ValueError(
                    f"The gaze model is from calibration session "
                    f"{model_session}, not participant {args.participant}. "
                    f"Calibrate first: gaze_calibration.py --label "
                    f"{args.participant}  (or pass --any-model)."
                )
        else:
            self.screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            self.screen_h = ctypes.windll.user32.GetSystemMetrics(1)

        # Trials, each laid out on one line before anything is shown.
        # On narrow screens the font shrinks; the size used is logged.
        seed = zlib.crc32(args.participant.encode("utf-8"))
        self.practice, self.main = build_trials(
            load_items(args.items), args.list, seed
        )
        self.font_size = args.font_size
        while True:
            self.renderer = TextRenderer(self.font_size)
            try:
                self.layout_trials()
                break
            except ValueError:
                self.renderer.close()
                if self.font_size <= MIN_FONT_SIZE:
                    raise
                self.font_size -= 2
        if self.font_size != args.font_size:
            print(f"Note: font reduced to {self.font_size} px so every "
                  f"sentence fits on this screen.")
        self.small = TextRenderer(max(18, self.font_size * 2 // 3))

        # Output folder and files
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(args.data_dir,
                                   f"{args.participant}_{stamp}")
        os.makedirs(self.folder, exist_ok=True)
        self.files = {}
        self.writers = {}
        for name, header in {
            "events": ["t", "phase", "trial_index", "event", "detail"],
            "gaze": ["t", "phase", "trial_index", "raw_x", "raw_y",
                     "x", "y", "valid", "reason"],
            "trials": ["trial_index", "phase", "item_id", "type",
                       "condition", "n_regions", "critical_region",
                       "regions", "boxes", "offset_x", "offset_y",
                       "drift_samples", "question", "answer", "response",
                       "correct", "question_rt_ms"],
            "validation": ["label", "point", "target_x", "target_y",
                           "samples", "median_x", "median_y", "error_x",
                           "error_y", "error"],
        }.items():
            self.files[name] = open(
                os.path.join(self.folder, f"{name}.csv"), "w",
                newline="", encoding="utf-8-sig",
            )
            self.writers[name] = csv.writer(self.files[name])
            self.writers[name].writerow(header)

        # Window and gaze source
        self.window = "Hybrid Reading"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(self.window, self.screen_w // 2,
                             self.screen_h // 2)
        else:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

        self.autopilot = None
        if args.gaze == "webcam":
            self.gaze = WebcamGaze(self.bundle, args.camera)
        elif args.gaze == "mouse":
            self.gaze = MouseGaze(self.window)
        else:
            self.gaze = AutopilotGaze(seed=seed)
            self.autopilot = Autopilot(self.gaze, args.speed, seed,
                                       self.log_event)

        self.write_session_info(completed=False)

    def layout_trials(self):
        """Region boxes for every trial; ValueError if one does not fit."""
        self.min_region_width = None
        for trial in self.practice + self.main:
            try:
                trial["boxes"], _ = layout_regions(
                    trial["regions"], self.renderer,
                    self.screen_w, self.screen_h, self.args.gap,
                )
            except ValueError as error:
                raise ValueError(f"Item {trial['item_id']} "
                                 f"{trial['condition']}: {error}") from None
            if trial["type"] == "experimental":
                narrowest = min(x2 - x1 for x1, _, x2, _ in trial["boxes"])
                self.min_region_width = min(
                    narrowest, self.min_region_width or narrowest
                )

    # ---------- logging ----------
    def now(self):
        return time.perf_counter() - self.t0

    def log_event(self, event, detail="", t=None):
        self.writers["events"].writerow([
            round(self.now() if t is None else t, 4), self.phase,
            self.trial_index, event, detail,
        ])

    def record(self, sample):
        t = sample.t - self.t0
        x = sample.x - self.offset[0]
        y = sample.y - self.offset[1]
        self.writers["gaze"].writerow([
            round(t, 4), self.phase, self.trial_index,
            round(sample.x, 1), round(sample.y, 1),
            round(x, 1), round(y, 1), int(sample.valid), sample.reason,
        ])
        self.recent.append((t, sample.x, sample.y, sample.valid))
        if sample.valid:
            self.last_gaze = (x, y)

    def write_session_info(self, completed, note=""):
        info = {
            "participant": self.args.participant,
            "list": self.args.list,
            "items_file": self.args.items,
            "gaze_source": self.args.gaze,
            "screen": [self.screen_w, self.screen_h],
            "font_size": self.font_size,
            "region_gap_px": self.args.gap,
            "line_spacing": getattr(self.args, "line_spacing", None),
            "min_experimental_region_width_px": self.min_region_width,
            "validation": self.validation,
            "completed": completed,
            "note": note,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.bundle is not None:
            info["gaze_model"] = {
                key: self.bundle[key] for key in (
                    "session_id", "feature_set", "model_kind",
                    "frame_width", "frame_height", "loto_median_error",
                    "loto_mae_x", "loto_mae_y",
                )
            }
        with open(os.path.join(self.folder, "session.json"), "w",
                  encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2)

    # ---------- display and input ----------
    def blank(self):
        return np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

    def present(self, canvas, event, detail=""):
        """Show a canvas and log the event when it is drawn."""
        self.base_canvas = canvas
        cv2.imshow(self.window, canvas)
        key = cv2.waitKey(1)  # lets the window paint
        if key != -1:
            self.pending_keys.append((key & 0xFF, self.now()))
        self.log_event(event, detail)

    def pump(self):
        """One loop step: gaze samples, overlay, keyboard."""
        for sample in self.gaze.poll():
            self.record(sample)

        if (self.args.show_gaze and self.last_gaze is not None
                and self.now() - self.last_overlay >= 1 / 30):
            frame = self.base_canvas.copy()
            gx, gy = self.last_gaze
            if np.isfinite(gx) and np.isfinite(gy):
                cv2.circle(frame, (int(gx), int(gy)), 9, (0, 0, 255), -1)
            cv2.imshow(self.window, frame)
            self.last_overlay = self.now()

        key = cv2.waitKey(1)
        t = self.now()
        if self.pending_keys:
            if key != -1:
                self.pending_keys.append((key & 0xFF, t))
            key, t = self.pending_keys.popleft()
        elif key != -1:
            key &= 0xFF
        else:
            key = None

        if self.autopilot is not None and key is None:
            key = self.autopilot.key(t)

        if key == KEY_ESC:
            raise Aborted("ESC pressed")
        return key, t

    def wait_for(self, keys, timeout=None):
        start = self.now()
        while True:
            key, t = self.pump()
            if key is not None and key in keys:
                return key, t
            if timeout is not None and t - start >= timeout:
                return None, t

    def text_screen(self, lines, event, keys=(KEY_SPACE,)):
        canvas = self.blank()
        y = 120
        for tamil, english in lines:
            self.renderer.draw(canvas, tamil, 100, y)
            y += int(self.font_size * 1.5)
            self.small.draw(canvas, english, 100, y, GREY)
            y += int(self.font_size * 1.6)
        self.present(canvas, event)
        if self.autopilot:
            self.autopilot.schedule(self.now(), [(0.4, ("key", KEY_SPACE))])
        return self.wait_for(set(keys))

    # ---------- validation ----------
    def run_validation(self, label):
        self.phase = f"validation_{label}"
        self.trial_index = -1
        self.offset = (0.0, 0.0)
        errors = []

        for index, (fx, fy) in enumerate(VALIDATION_POINTS):
            point = (int(fx * self.screen_w), int(fy * self.screen_h))
            canvas = self.blank()
            cv2.circle(canvas, point, 14, DOT, -1)
            cv2.circle(canvas, point, 24, WHITE, 2)
            self.small.draw(canvas, TEXT_LOOK_AT_DOT, 40, 30, GREY)
            self.present(canvas, "validation_point", index)
            start = self.now()
            if self.autopilot:
                self.autopilot.schedule(start, [(0, ("look", point))])
            self.wait_for(set(), VALIDATION_SHOW_S * self.time_scale)

            first = start + VALIDATION_SKIP_S * self.time_scale
            pts = np.array([(x, y) for t, x, y, valid in self.recent
                            if valid and t >= first])
            if len(pts):
                med = np.median(pts, axis=0)
                err = med - point
                errors.append(err)
                row = [label, index, point[0], point[1], len(pts),
                       round(med[0], 1), round(med[1], 1),
                       round(err[0], 1), round(err[1], 1),
                       round(float(np.hypot(*err)), 1)]
            else:
                row = [label, index, point[0], point[1], 0] + [""] * 5
            self.writers["validation"].writerow(row)

        summary = {"points_with_data": len(errors)}
        if errors:
            errors = np.array(errors)
            summary["median_abs_error_x"] = round(
                float(np.median(np.abs(errors[:, 0]))), 1)
            summary["median_abs_error_y"] = round(
                float(np.median(np.abs(errors[:, 1]))), 1)
            summary["median_error"] = round(
                float(np.median(np.hypot(errors[:, 0], errors[:, 1]))), 1)
        self.validation[label] = summary
        self.write_session_info(completed=False)
        self.validation_result_screen(label, summary)

    def validation_result_screen(self, label, summary):
        canvas = self.blank()
        lines = [f"Validation ({label}): {summary['points_with_data']}/9 "
                 "points had gaze data"]
        if "median_error" in summary:
            needed = 3 * summary["median_abs_error_x"]
            lines += [
                f"Median horizontal error: {summary['median_abs_error_x']} px"
                f"   |   vertical: {summary['median_abs_error_y']} px"
                f"   |   overall: {summary['median_error']} px",
                f"Narrowest experimental region: {self.min_region_width} px."
                f"  Design rule needs region >= 3 x horizontal error"
                f" = {needed:.0f} px",
                "Verdict: " + (
                    "OK for region-level gaze"
                    if self.min_region_width and needed <= self.min_region_width
                    else "TOO INACCURATE for these regions - recalibrate, "
                         "or treat gaze measures as unreliable"
                ),
            ]
        lines.append("C = continue        Q = quit (data saved)")
        for row, text in enumerate(lines):
            self.small.draw(canvas, text, 60, 80 + row * 55, WHITE)
        self.present(canvas, "validation_result", label)
        if self.autopilot:
            self.autopilot.schedule(self.now(), [(0.4, ("key", ord("c")))])
        key, _ = self.wait_for({ord("c"), ord("C"), ord("q"), ord("Q")})
        if key in (ord("q"), ord("Q")):
            raise Aborted(f"operator quit after validation ({label})")

    # ---------- trials ----------
    def run_trial(self, trial, phase):
        self.trial_index = self.trials_run
        self.trials_run += 1
        self.phase = phase
        regions, boxes = trial["regions"], trial["boxes"]
        line_y = (boxes[0][1] + boxes[0][3]) // 2

        # Drift check: look at the dot where region 1 starts
        self.offset = (0.0, 0.0)
        dot = (boxes[0][0] + 8, line_y)
        canvas = self.blank()
        cv2.circle(canvas, dot, 10, DOT, -1)
        self.present(canvas, "drift_on")
        if self.autopilot:
            self.autopilot.schedule(self.now(), [
                (0.0, ("look", dot)), (0.7, ("key", KEY_SPACE)),
            ])
        _, t_space = self.wait_for({KEY_SPACE})
        self.log_event("drift_space", t=t_space)

        first = t_space - DRIFT_WINDOW_S * self.time_scale
        pts = np.array([(x, y) for t, x, y, valid in self.recent
                        if valid and first <= t <= t_space])
        if len(pts) >= MIN_DRIFT_SAMPLES:
            med = np.median(pts, axis=0)
            self.offset = (float(med[0] - dot[0]), float(med[1] - dot[1]))

        # Cumulative reveal
        for revealed in range(1, len(regions) + 1):
            canvas = self.blank()
            draw_sentence(canvas, regions, boxes, revealed, self.renderer)
            self.present(canvas, "reveal", revealed)
            if self.autopilot:
                self.autopilot.schedule(
                    self.now(),
                    self.autopilot.reading_steps(boxes, revealed, trial),
                )

            if trial["type"] == "lookback" and revealed == len(regions):
                self.run_lookback_cue(canvas, regions, boxes, revealed)

            _, t_key = self.wait_for({KEY_SPACE})
            self.log_event("key", revealed, t=t_key)

        self.log_event("trial_end")

        # Comprehension question
        response, correct, rt = "", "", ""
        if trial["question"]:
            canvas = self.blank()
            width, _ = self.renderer.measure(trial["question"])
            self.renderer.draw(canvas, trial["question"],
                               (self.screen_w - width) // 2,
                               self.screen_h // 2 - 80)
            width, _ = self.renderer.measure(TEXT_ANSWER)
            self.renderer.draw(canvas, TEXT_ANSWER,
                               (self.screen_w - width) // 2,
                               self.screen_h // 2 + 40, GREY)
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

        self.writers["trials"].writerow([
            self.trial_index, phase, trial["item_id"], trial["type"],
            trial["condition"], len(regions), trial["critical_region"] or "",
            " | ".join(regions), json.dumps(boxes),
            round(self.offset[0], 1), round(self.offset[1], 1), len(pts),
            trial["question"], trial["answer"], response, correct, rt,
        ])
        for file in self.files.values():
            file.flush()

    def run_lookback_cue(self, canvas, regions, boxes, revealed):
        self.wait_for(set(), LOOKBACK_DELAY_S * self.time_scale)
        cue = self.blank()
        draw_sentence(cue, regions, boxes, revealed, self.renderer,
                      highlight=0)
        self.present(cue, "cue_on", 1)
        if self.autopilot:
            self.autopilot.schedule(self.now(), [
                (0.05, ("look", centre(boxes[0]))),
            ])
        self.wait_for(set(), LOOKBACK_CUE_S * self.time_scale)
        self.present(canvas, "cue_off", 1)
        if self.autopilot:
            self.autopilot.schedule(self.now(), [
                (0.0, ("look", centre(boxes[revealed - 1]))),
                (0.5, ("key", KEY_SPACE)),
            ])

    # ---------- whole session ----------
    def run(self):
        note = ""
        completed = False
        try:
            self.phase = "instructions"
            self.text_screen(INSTRUCTIONS, "instructions_on")

            if not self.args.skip_validation:
                self.run_validation("start")

            for trial in self.practice:
                self.run_trial(trial, "practice")
            if self.practice:
                self.phase = "instructions"
                self.text_screen([TEXT_PRACTICE_DONE], "practice_done")

            # A break (with validation) between each block of trials
            blocks = self.N_BLOCKS
            breaks = [len(self.main) * k // blocks for k in range(1, blocks)]
            for number, trial in enumerate(self.main):
                if number in breaks and number > 0:
                    if not self.args.skip_validation:
                        self.run_validation(
                            "break" if blocks == 2
                            else f"break{breaks.index(number) + 1}")
                    self.phase = "break"
                    self.text_screen([TEXT_BREAK], "break_on")
                self.run_trial(trial, "main")

            if not self.args.skip_validation:
                self.run_validation("end")

            self.phase = "end"
            canvas = self.blank()
            self.renderer.draw(canvas, TEXT_END[0], 100, 200)
            self.small.draw(canvas, TEXT_END[1], 100, 270, GREY)
            self.present(canvas, "end")
            self.wait_for(set(), 1.5 * self.time_scale)
            completed = True
        except Aborted as reason:
            note = str(reason)
            self.log_event("aborted", note)
        finally:
            self.write_session_info(completed, note)
            self.gaze.close()
            for file in self.files.values():
                file.close()
            self.renderer.close()
            self.small.close()
            cv2.destroyAllWindows()

        print(f"Session {'completed' if completed else 'stopped: ' + note}")
        print(f"Data saved in: {self.folder}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid SPR + gaze")
    parser.add_argument("--participant", required=True,
                        help="anonymous code, e.g. P01 (no names)")
    parser.add_argument("--list", type=int, default=0,
                        help="Latin-square list number (0, 1, ...)")
    parser.add_argument("--items", default=ITEMS_FILE)
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="output folder (use data_simulated for "
                             "autopilot runs)")
    parser.add_argument("--gaze", choices=["webcam", "mouse", "autopilot"],
                        default="webcam")
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--any-model", action="store_true",
                        help="allow a gaze model calibrated under "
                             "another label")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--gap", type=int, default=40,
                        help="pixels between regions")
    parser.add_argument("--show-gaze", action="store_true",
                        help="draw the gaze point (testing only)")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="autopilot speed-up factor")
    args = parser.parse_args()

    # 1 ms timer resolution so cv2.waitKey(1) polls keys every ~1 ms
    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
    try:
        Session(args).run()
    finally:
        winmm.timeEndPeriod(1)


if __name__ == "__main__":
    main()

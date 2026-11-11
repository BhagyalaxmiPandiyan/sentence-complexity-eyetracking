
"""
Gaze calibration: 25 targets, logs every sample, then trains the model.

Writes calibration\\calibration_samples.csv (appended per session) and
calibration\\gaze_model_v2.joblib, then shows the live gaze point.

Usage:
    .\\.venv\\Scripts\\python.exe gaze_calibration.py --label P01
"""

import argparse
import cv2
import numpy as np
import ctypes
import csv
import os
import sys
import time

from gaze_features import ALL_MEASURES, create_face_mesh, measure, vector
from train_gaze_model import describe, train
from validate_calibration import CALIBRATION_DIR, SAMPLES_CSV

# ---------- Configuration ----------
CAMERA_INDEX = 0
FRAMES_PER_TARGET = 75

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="",
                    help="tag added to the session id, e.g. still_640")
parser.add_argument("--camera-width", type=int, default=None,
                    help="requested webcam width (default: camera default)")
parser.add_argument("--camera-height", type=int, default=None)
args = parser.parse_args()

screen_w = ctypes.windll.user32.GetSystemMetrics(0)
screen_h = ctypes.windll.user32.GetSystemMetrics(1)

mesh = create_face_mesh()

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam. Close other camera apps.")

if args.camera_width and args.camera_height:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

ok, frame = camera.read()
if ok:
    # The camera may not support the requested size; this is the truth.
    print(f"Camera frame size: {frame.shape[1]} x {frame.shape[0]}")

window = "Gaze Calibration"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

session_id = time.strftime("%Y%m%d_%H%M%S")
if args.label:
    session_id += "_" + args.label


# 25 targets: 5 x 5 grid
X_FRACTIONS = (0.08, 0.29, 0.50, 0.71, 0.92)
Y_FRACTIONS = (0.08, 0.29, 0.50, 0.71, 0.92)

positions = [
    (col, row, x, y)
    for row, y in enumerate(Y_FRACTIONS)
    for col, x in enumerate(X_FRACTIONS)
]

def draw_screen(target=None, message=""):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    if target is not None:
        tx, ty = target
        cv2.circle(canvas, (tx, ty), 15, (0, 220, 255), -1)
        cv2.circle(canvas, (tx, ty), 27, (255, 255, 255), 2)

    if message:
        cv2.putText(
            canvas, message, (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2
        )

    cv2.imshow(window, canvas)


# Raw samples for offline validation (validate_calibration.py).
# Appends, so several sessions can share one file.
os.makedirs(CALIBRATION_DIR, exist_ok=True)
write_header = (
    not os.path.exists(SAMPLES_CSV) or os.path.getsize(SAMPLES_CSV) == 0
)
samples_file = open(SAMPLES_CSV, "a", newline="", encoding="utf-8")
samples_writer = csv.writer(samples_file)

if write_header:
    samples_writer.writerow([
        "session_id", "target_id", "grid_col", "grid_row",
        "target_x", "target_y", "sample_index", "attempt_index",
        "seconds_since_space", "screen_w", "screen_h",
        "frame_w", "frame_h",
    ] + ALL_MEASURES)

stopped = False

try:
    # ---------- Calibration ----------
    for index, (col, row, px, py) in enumerate(positions, start=1):
        target = (int(px * screen_w), int(py * screen_h))

        draw_screen(
            target,
            f"Target {index}/{len(positions)}: Look at the dot, then press SPACE"
        )

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam frame unavailable.")

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                raise KeyboardInterrupt

            if key == ord(" "):
                break

        collected = 0
        attempts = 0
        space_time = time.time()

        while collected < FRAMES_PER_TARGET:
            ok, frame = camera.read()
            if not ok:
                continue

            measures = measure(frame, mesh)
            attempts += 1

            if measures is not None:
                frame_h, frame_w = frame.shape[:2]
                samples_writer.writerow([
                    session_id, index - 1, col, row,
                    target[0], target[1], collected, attempts - 1,
                    round(time.time() - space_time, 4),
                    screen_w, screen_h, frame_w, frame_h,
                ] + [measures[name] for name in ALL_MEASURES])

                collected += 1

            draw_screen(
                target,
                f"Target {index}/{len(positions)}: Samples {collected}/"
                f"{FRAMES_PER_TARGET}"
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

            if attempts > FRAMES_PER_TARGET * 10:
                raise RuntimeError(
                    "Face detection is unstable. Improve lighting "
                    "and reposition your face, then restart."
                )

        samples_file.flush()
        time.sleep(0.2)

    samples_file.close()
    print(f"Calibration samples saved: {SAMPLES_CSV} "
          f"(session {session_id})")

    # ---------- Train gaze mapping ----------
    draw_screen(None, "Training gaze model...")
    cv2.waitKey(1)
    bundle = train(session=session_id)
    describe(bundle)
    model = bundle["model"]

    # ---------- Live gaze demonstration ----------
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        measures = measure(frame, mesh)

        if measures is not None:
            gx, gy = model.predict(
                vector(measures, bundle["features"])[None, :]
            )[0]
            gx = int(np.clip(gx, 0, screen_w - 1))
            gy = int(np.clip(gy, 0, screen_h - 1))

            draw_screen(
                (gx, gy),
                f"Estimated gaze point (held-out error ~"
                f"{bundle['loto_median_error']:.0f} px) | Q to quit"
            )
        else:
            draw_screen(None, "Face not detected - adjust position")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Calibration stopped.")
    stopped = True

finally:
    samples_file.close()
    camera.release()
    mesh.close()
    cv2.destroyAllWindows()

print("Program ended.")

# Non-zero exit lets run_session.bat stop instead of using an old model.
if stopped:
    sys.exit(1)

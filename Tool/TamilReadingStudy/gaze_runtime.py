
"""
Gaze sample sources for hybrid_spr.py.

All sources return GazeSample tuples timestamped with time.perf_counter():
    WebcamGaze    - live webcam + gaze_model_v2.joblib, in a background
                    thread so key presses are timed independently of
                    face-mesh processing
    MouseGaze     - mouse position as gaze (testing the experiment flow)
    AutopilotGaze - synthetic gaze steered by the runner's autopilot
                    (automated end-to-end test, no person needed)
"""

import queue
import threading
import time
from collections import deque, namedtuple

import cv2
import numpy as np

from gaze_features import create_face_mesh, measure, vector

GazeSample = namedtuple("GazeSample", "t x y valid reason")

SAMPLE_INTERVAL = 1 / 30  # mouse/autopilot sampling, like a webcam


class WebcamGaze:
    kind = "webcam"

    def __init__(self, bundle, camera_index=0, clock=time.perf_counter):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.blink_ratio = bundle["blink_ratio"]
        self.screen_w = bundle["screen_width"]
        self.screen_h = bundle["screen_height"]
        self.clock = clock
        self.samples = queue.Queue()
        self.stop_event = threading.Event()

        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot open webcam. Close other camera apps.")

        # Use the resolution the model was calibrated with.
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, bundle["frame_width"])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, bundle["frame_height"])
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Webcam frame unavailable.")
        self.frame_size = (frame.shape[1], frame.shape[0])
        if self.frame_size != (bundle["frame_width"], bundle["frame_height"]):
            print(f"WARNING: camera gives {self.frame_size}, model was "
                  f"calibrated at {bundle['frame_width']} x "
                  f"{bundle['frame_height']}")

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        mesh = create_face_mesh()
        openness_history = deque(maxlen=90)  # ~3 s at 30 fps

        try:
            while not self.stop_event.is_set():
                ok, frame = self.camera.read()
                t = self.clock()

                if not ok:
                    self.samples.put(GazeSample(t, np.nan, np.nan, False,
                                                "no_frame"))
                    continue

                measures = measure(frame, mesh)
                if measures is None:
                    self.samples.put(GazeSample(t, np.nan, np.nan, False,
                                                "no_face"))
                    continue

                openness = (measures["eye1_openness"]
                            + measures["eye2_openness"]) / 2
                openness_history.append(openness)
                if (len(openness_history) >= 15 and openness
                        < self.blink_ratio * np.median(openness_history)):
                    self.samples.put(GazeSample(t, np.nan, np.nan, False,
                                                "blink"))
                    continue

                x, y = self.model.predict(
                    vector(measures, self.features)[None, :]
                )[0]

                # Far outside the screen = not a usable estimate
                margin_x = 0.1 * self.screen_w
                margin_y = 0.1 * self.screen_h
                if (-margin_x <= x <= self.screen_w + margin_x
                        and -margin_y <= y <= self.screen_h + margin_y):
                    self.samples.put(GazeSample(t, x, y, True, "ok"))
                else:
                    self.samples.put(GazeSample(t, x, y, False, "offscreen"))
        finally:
            mesh.close()

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.samples.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.camera.release()


class MouseGaze:
    kind = "mouse"

    def __init__(self, window, clock=time.perf_counter):
        self.clock = clock
        self.position = None
        self.last_t = 0.0
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        self.position = (x, y)

    def poll(self):
        now = self.clock()
        if now - self.last_t < SAMPLE_INTERVAL:
            return []
        self.last_t = now
        if self.position is None:
            return [GazeSample(now, np.nan, np.nan, False, "no_mouse")]
        return [GazeSample(now, float(self.position[0]),
                           float(self.position[1]), True, "ok")]

    def close(self):
        pass


class AutopilotGaze:
    """Noisy, biased synthetic gaze that follows self.target."""

    kind = "autopilot"

    def __init__(self, noise_px=40.0, bias=None, seed=0,
                 clock=time.perf_counter):
        self.clock = clock
        self.noise_px = noise_px
        self.rng = np.random.default_rng(seed)
        # Each simulated reader gets its own calibration offset.
        self.bias = (self.rng.normal(0, 60, 2) if bias is None
                     else np.asarray(bias, dtype=float))
        self.target = None
        self.last_t = 0.0

    def poll(self):
        now = self.clock()
        if now - self.last_t < SAMPLE_INTERVAL:
            return []
        self.last_t = now
        if self.target is None:
            return [GazeSample(now, np.nan, np.nan, False, "no_target")]
        if self.rng.random() < 0.05:
            return [GazeSample(now, np.nan, np.nan, False, "blink")]
        x, y = (np.asarray(self.target, dtype=float) + self.bias
                + self.rng.normal(0, self.noise_px, 2))
        return [GazeSample(now, x, y, True, "ok")]

    def close(self):
        pass

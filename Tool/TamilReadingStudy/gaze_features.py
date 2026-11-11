
"""
Shared eye and face measurements for the gaze scripts.

measure() returns one dict of measurements per webcam frame.

LEGACY_FEATURES are numerically identical to the original
extract_features() in gaze_calibration.py, so existing
gaze_model.joblib files stay valid. The other fields are logged
for offline experiments (validate_calibration.py); no live model
uses them yet.
"""

import math

import cv2
import numpy as np

# Original four features: iris position inside each eye box.
# The vertical ratio is normalised by the eyelid gap, which moves
# with vertical gaze - kept unchanged as the baseline.
LEGACY_FEATURES = [
    "eye1_x_ratio", "eye1_y_ratio",
    "eye2_x_ratio", "eye2_y_ratio",
]

# Candidate replacement: iris offset from the midpoint of the eye
# corners, measured along/across the corner line and divided by the
# eye width. Does not use the eyelids, and is roll-invariant.
CORNER_FEATURES = [
    "eye1_h_corner", "eye1_v_corner",
    "eye2_h_corner", "eye2_v_corner",
]

# Head-position proxies from face landmarks (not a calibrated
# head-pose estimate).
HEAD_FEATURES = [
    "head_yaw", "head_pitch", "head_roll_deg",
    "head_scale", "nose_x", "nose_y",
]

# Detection-quality indicators (blinks, distance to camera).
QUALITY_FIELDS = [
    "eye1_openness", "eye2_openness", "inter_ocular_px",
]

ALL_MEASURES = (
    LEGACY_FEATURES + CORNER_FEATURES + HEAD_FEATURES + QUALITY_FIELDS
)

# Iris centres: 468-472 and 473-477
IRIS_LANDMARKS = [range(468, 473), range(473, 478)]

# Eye corners and upper/lower eyelid landmarks
EYE_GEOMETRY = [
    (33, 133, 159, 145),
    (362, 263, 386, 374),
]

NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
CHEEK_A = 234
CHEEK_B = 454
OUTER_CORNER_A = 33
OUTER_CORNER_B = 263


def create_face_mesh():
    import mediapipe as mp

    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def measure(frame, mesh):
    """Return a dict of ALL_MEASURES for one BGR frame, or None."""
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark

    # Normalised image coordinates (used by the legacy features)
    def xy(i):
        return np.array([lm[i].x, lm[i].y], dtype=float)

    # Pixel coordinates (geometry is not distorted by aspect ratio)
    def px(i):
        return np.array([lm[i].x * frame_w, lm[i].y * frame_h], dtype=float)

    out = {}

    for n, (iris_ids, (corner_a, corner_b, upper, lower)) in enumerate(
        zip(IRIS_LANDMARKS, EYE_GEOMETRY), start=1
    ):
        # ---------- Legacy features (unchanged) ----------
        iris = np.mean([xy(i) for i in iris_ids], axis=0)
        a = xy(corner_a)
        b = xy(corner_b)
        top = xy(upper)
        bottom = xy(lower)

        eye_width = abs(b[0] - a[0])
        eye_height = abs(bottom[1] - top[1])

        if eye_width < 1e-5 or eye_height < 1e-5:
            return None

        out[f"eye{n}_x_ratio"] = (iris[0] - min(a[0], b[0])) / eye_width
        out[f"eye{n}_y_ratio"] = (
            iris[1] - min(top[1], bottom[1])
        ) / eye_height

        # ---------- Corner-referenced features ----------
        iris_p = np.mean([px(i) for i in iris_ids], axis=0)
        a_p = px(corner_a)
        b_p = px(corner_b)
        axis = b_p - a_p
        width_p = float(np.linalg.norm(axis))

        if width_p < 1.0:
            return None

        along = axis / width_p
        across = np.array([-along[1], along[0]])  # points down the image
        offset = iris_p - (a_p + b_p) / 2

        out[f"eye{n}_h_corner"] = float(offset @ along) / width_p
        out[f"eye{n}_v_corner"] = float(offset @ across) / width_p

        # Eyelid gap relative to eye width; drops sharply in a blink.
        out[f"eye{n}_openness"] = (
            float(np.linalg.norm(px(lower) - px(upper))) / width_p
        )

    # ---------- Head-position proxies ----------
    nose = px(NOSE_TIP)
    cheek_a = px(CHEEK_A)
    cheek_b = px(CHEEK_B)
    face_axis = cheek_b - cheek_a
    face_width = float(np.linalg.norm(face_axis))

    outer_a = px(OUTER_CORNER_A)
    outer_b = px(OUTER_CORNER_B)
    eye_line_mid = (outer_a + outer_b) / 2
    inter_ocular = float(np.linalg.norm(outer_b - outer_a))

    face_vertical = px(CHIN) - px(FOREHEAD)
    face_height = float(np.linalg.norm(face_vertical))

    if face_width < 1.0 or face_height < 1.0:
        return None

    out["head_yaw"] = float(
        (nose - (cheek_a + cheek_b) / 2) @ (face_axis / face_width)
    ) / face_width
    out["head_pitch"] = float(
        (nose - eye_line_mid) @ (face_vertical / face_height)
    ) / face_height
    out["head_roll_deg"] = math.degrees(
        math.atan2(outer_b[1] - outer_a[1], outer_b[0] - outer_a[0])
    )
    out["head_scale"] = inter_ocular / frame_w
    out["nose_x"] = lm[NOSE_TIP].x
    out["nose_y"] = lm[NOSE_TIP].y
    out["inter_ocular_px"] = inter_ocular

    return out


def vector(measures, names):
    """Select named measures as a 1-D float array."""
    return np.array([measures[name] for name in names], dtype=float)

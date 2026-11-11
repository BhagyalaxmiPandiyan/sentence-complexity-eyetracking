
"""
Train the gaze model used by hybrid_spr.py from logged calibration data.

gaze_calibration.py calls train() automatically after a session. Run this
file directly to retrain from an earlier session or with other settings.

Reads calibration\\calibration_samples.csv, applies the same settling and
blink filtering as validate_calibration.py, reports leave-one-target-out
error, and saves calibration\\gaze_model_v2.joblib.

Usage:
    .\\.venv\\Scripts\\python.exe train_gaze_model.py
    .\\.venv\\Scripts\\python.exe train_gaze_model.py --session 20261023_101500_P01
    .\\.venv\\Scripts\\python.exe train_gaze_model.py --features corner
"""

import argparse
import os

import joblib
import numpy as np

from gaze_features import CORNER_FEATURES, HEAD_FEATURES, LEGACY_FEATURES
from validate_calibration import (
    CALIBRATION_DIR, SAMPLES_CSV, leave_one_target_out, load_session,
    make_model,
)

MODEL_FILE = os.path.join(CALIBRATION_DIR, "gaze_model_v2.joblib")

# Best validated combination on session 20261022_223351 is corner_head.
FEATURE_SETS = {
    "legacy": LEGACY_FEATURES,
    "corner": CORNER_FEATURES,
    "corner_head": CORNER_FEATURES + HEAD_FEATURES,
    "legacy_head": LEGACY_FEATURES + HEAD_FEATURES,
}


def train(samples=SAMPLES_CSV, session=None, feature_set="corner_head",
          model_kind="ridge_poly2", drop_first=10, blink_ratio=0.6,
          output=MODEL_FILE):
    """Fit, validate and save the gaze model; returns the saved bundle."""
    session, _, columns = load_session(samples, session)
    features = FEATURE_SETS[feature_set]

    # Same filtering as validate_calibration.py
    keep = columns["sample_index"] >= drop_first
    openness = (columns["eye1_openness"] + columns["eye2_openness"]) / 2
    target_median = {
        g: np.median(openness[columns["target_id"] == g])
        for g in np.unique(columns["target_id"])
    }
    keep &= openness >= blink_ratio * np.array(
        [target_median[g] for g in columns["target_id"]]
    )

    X = np.column_stack([columns[f][keep] for f in features])
    Y = np.column_stack([columns["target_x"][keep], columns["target_y"][keep]])
    groups = columns["target_id"][keep].astype(int)

    # Held-out estimate, reported and stored with the model
    predictions = leave_one_target_out(X, Y, groups, model_kind)
    errors = np.linalg.norm(predictions - Y, axis=1)

    bundle = {
        "model": make_model(model_kind).fit(X, Y),
        "features": features,
        "feature_set": feature_set,
        "model_kind": model_kind,
        "session_id": session,
        "screen_width": int(columns["screen_w"][0]),
        "screen_height": int(columns["screen_h"][0]),
        "frame_width": int(columns["frame_w"][0]),
        "frame_height": int(columns["frame_h"][0]),
        "blink_ratio": blink_ratio,
        "samples": len(Y),
        "loto_median_error": float(np.median(errors)),
        "loto_mae_x": float(np.mean(np.abs(predictions[:, 0] - Y[:, 0]))),
        "loto_mae_y": float(np.mean(np.abs(predictions[:, 1] - Y[:, 1]))),
    }
    joblib.dump(bundle, output)
    return bundle


def describe(bundle, output=MODEL_FILE):
    print(f"Session: {bundle['session_id']} | features: "
          f"{bundle['feature_set']} ({len(bundle['features'])}) | model: "
          f"{bundle['model_kind']} | samples: {bundle['samples']}")
    print(f"Held-out (leave-one-target-out) error: median "
          f"{bundle['loto_median_error']:.1f} px, MAE x "
          f"{bundle['loto_mae_x']:.1f} px, MAE y "
          f"{bundle['loto_mae_y']:.1f} px")
    print(f"Camera frame used in calibration: {bundle['frame_width']} x "
          f"{bundle['frame_height']}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=SAMPLES_CSV)
    parser.add_argument("--session", default=None)
    parser.add_argument("--features", choices=FEATURE_SETS,
                        default="corner_head")
    parser.add_argument("--model", choices=["ridge_poly2", "extratrees"],
                        default="ridge_poly2")
    parser.add_argument("--drop-first", type=int, default=10)
    parser.add_argument("--blink-ratio", type=float, default=0.6)
    parser.add_argument("--output", default=MODEL_FILE)
    args = parser.parse_args()

    bundle = train(args.samples, args.session, args.features, args.model,
                   args.drop_first, args.blink_ratio, args.output)
    describe(bundle, args.output)


if __name__ == "__main__":
    main()


"""
Offline, target-held-out validation of the gaze calibration.

Reads calibration\\calibration_samples.csv written by gaze_calibration.py. Every
evaluation model is trained only on targets excluded from its test
set, so the errors measure prediction at unseen screen positions,
not training error. Predictions are never clipped to the screen.

Usage (PowerShell, from D:\\eye-tracker):
    .\\.venv\\Scripts\\python.exe validate_calibration.py
    .\\.venv\\Scripts\\python.exe validate_calibration.py --blink-filter
    .\\.venv\\Scripts\\python.exe validate_calibration.py --session 20261022_191800
"""

import argparse
import csv
import math
import os
import time

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from gaze_features import (
    CORNER_FEATURES, HEAD_FEATURES, LEGACY_FEATURES, QUALITY_FIELDS,
)

# ---------- Configuration ----------
CALIBRATION_DIR = "calibration"
SAMPLES_CSV = os.path.join(CALIBRATION_DIR, "calibration_samples.csv")
PER_TARGET_CSV = os.path.join(CALIBRATION_DIR, "validation_per_target.csv")
EXPERIMENT_LOG = os.path.join(CALIBRATION_DIR, "experiments_log.csv")

# Line spacing of the current reading layout (reading_tracker.py).
# Gaze error must be well below half of it to assign lines reliably.
READING_LINE_SPACING_PX = 65

# Each experiment changes one factor relative to an earlier one:
# A -> B changes the model; B -> C changes the eye features;
# B -> D adds head-position features.
EXPERIMENTS = [
    ("A_legacy4_extratrees", LEGACY_FEATURES, "extratrees"),
    ("B_legacy4_ridge", LEGACY_FEATURES, "ridge_poly2"),
    ("C_corner4_ridge", CORNER_FEATURES, "ridge_poly2"),
    ("D_legacy4_head_ridge", LEGACY_FEATURES + HEAD_FEATURES, "ridge_poly2"),
    # C -> E adds head features to the corner features.
    ("E_corner4_head_ridge", CORNER_FEATURES + HEAD_FEATURES, "ridge_poly2"),
]


def make_model(kind):
    if kind == "extratrees":
        # Same settings as the live model in gaze_calibration.py
        return ExtraTreesRegressor(
            n_estimators=150, random_state=42,
            min_samples_leaf=2, n_jobs=-1,
        )
    return make_pipeline(
        StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0)
    )


# ---------- LOAD DATA ----------
def load_session(path, session):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run gaze_calibration.py first."
        )

    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"{path} contains no samples.")

    sessions = sorted({row["session_id"] for row in rows})
    if session is None:
        session = sessions[-1]
    elif session not in sessions:
        raise ValueError(
            f"Session {session} not found. Available: {sessions}"
        )

    rows = [row for row in rows if row["session_id"] == session]
    columns = {
        name: np.array([float(row[name]) for row in rows])
        for name in rows[0] if name != "session_id"
    }
    return session, sessions, columns


# ---------- EVALUATION ----------
def leave_one_target_out(X, Y, groups, kind):
    """Predict each target from a model trained on all other targets."""
    predictions = np.full(Y.shape, np.nan)

    for group in np.unique(groups):
        test = groups == group
        train = ~test
        assert group not in set(groups[train])

        model = make_model(kind).fit(X[train], Y[train])
        predictions[test] = model.predict(X[test])

    return predictions


def grouped_splits(X, Y, groups, kind, repeats, test_fraction):
    """Median error per split over repeated random target hold-outs."""
    splitter = GroupShuffleSplit(
        n_splits=repeats, test_size=test_fraction, random_state=0
    )
    split_medians = []

    for train, test in splitter.split(X, Y, groups):
        assert not set(groups[train]) & set(groups[test])

        model = make_model(kind).fit(X[train], Y[train])
        errors = np.linalg.norm(model.predict(X[test]) - Y[test], axis=1)
        split_medians.append(np.median(errors))

    return np.array(split_medians)


def per_target_rows(predictions, Y, groups, cols, rows):
    results = []

    for group in np.unique(groups):
        sel = groups == group
        pred = predictions[sel]
        target = Y[sel][0]
        errors = np.linalg.norm(pred - target, axis=1)
        mean_pred = pred.mean(axis=0)

        results.append({
            "target_id": int(group),
            "grid_col": int(cols[sel][0]),
            "grid_row": int(rows[sel][0]),
            "target_x": target[0],
            "target_y": target[1],
            "samples": int(sel.sum()),
            "mean_pred_x": round(mean_pred[0], 1),
            "mean_pred_y": round(mean_pred[1], 1),
            "mae_x": round(np.mean(np.abs(pred[:, 0] - target[0])), 1),
            "mae_y": round(np.mean(np.abs(pred[:, 1] - target[1])), 1),
            "mean_error": round(errors.mean(), 1),
            "median_error": round(np.median(errors), 1),
            # Bias: systematic offset of the average prediction
            "bias": round(np.linalg.norm(mean_pred - target), 1),
            # Precision: frame-to-frame spread while fixating
            "sd_x": round(pred[:, 0].std(), 1),
            "sd_y": round(pred[:, 1].std(), 1),
        })

    return results


def is_inner(row, n_cols, n_rows):
    return (
        0 < row["grid_col"] < n_cols - 1
        and 0 < row["grid_row"] < n_rows - 1
    )


def print_grid(results, key, n_cols, n_rows):
    grid = {(r["grid_row"], r["grid_col"]): r[key] for r in results}
    for row in range(n_rows):
        print("      " + " ".join(
            f"{grid.get((row, col), float('nan')):7.0f}"
            for col in range(n_cols)
        ))


# ---------- FEATURE STABILITY ----------
def feature_stability(columns, groups, names):
    """Within-target spread vs. separation between target means."""
    unique = np.unique(groups)
    target_x = np.array([columns["target_x"][groups == g][0] for g in unique])
    target_y = np.array([columns["target_y"][groups == g][0] for g in unique])

    print(f"\n{'feature':<16}{'within_sd':>11}{'between_sd':>12}"
          f"{'ratio':>8}{'r(x)':>8}{'r(y)':>8}")

    for name in names:
        values = columns[name]
        means = np.array([values[groups == g].mean() for g in unique])
        sds = np.array([values[groups == g].std() for g in unique])
        within = math.sqrt(np.mean(sds ** 2))
        between = means.std()
        ratio = between / within if within > 0 else float("inf")

        def corr(target):
            if means.std() == 0 or target.std() == 0:
                return float("nan")
            return np.corrcoef(means, target)[0, 1]

        print(f"{name:<16}{within:>11.4f}{between:>12.4f}"
              f"{ratio:>8.2f}{corr(target_x):>8.2f}{corr(target_y):>8.2f}")

    print("ratio = between-target / within-target spread. Below ~1 the "
          "feature cannot\nseparate targets. r(x), r(y) = correlation of "
          "per-target means with\ntarget X / Y: a gaze feature should "
          "correlate strongly with one of them.")


def to_degrees(pixels, screen_w_px, screen_width_cm, distance_cm):
    cm = pixels * screen_width_cm / screen_w_px
    return math.degrees(math.atan2(cm, distance_cm))


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--samples", default=SAMPLES_CSV)
    parser.add_argument("--session", default=None,
                        help="session_id (default: most recent)")
    parser.add_argument("--drop-first", type=int, default=10,
                        help="samples to drop after SPACE per target "
                             "(eye still settling); default 10")
    parser.add_argument("--no-blink-filter", dest="blink_filter",
                        action="store_false",
                        help="keep frames whose eye openness is below "
                             "--blink-ratio x that target's median")
    parser.add_argument("--blink-ratio", type=float, default=0.6)
    parser.add_argument("--repeats", type=int, default=30,
                        help="GroupShuffleSplit repeats")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--screen-width-cm", type=float, default=None)
    parser.add_argument("--distance-cm", type=float, default=None)
    args = parser.parse_args()

    session, sessions, columns = load_session(args.samples, args.session)

    screen_w = int(columns["screen_w"][0])
    screen_h = int(columns["screen_h"][0])
    n_cols = int(columns["grid_col"].max()) + 1
    n_rows = int(columns["grid_row"].max()) + 1

    print("=== CALIBRATION VALIDATION ===")
    print(f"Samples file: {args.samples} | sessions available: "
          f"{len(sessions)} | using: {session}")
    print(f"Screen: {screen_w} x {screen_h} | camera frame: "
          f"{int(columns['frame_w'][0])} x {int(columns['frame_h'][0])}")
    print(f"Targets: {len(np.unique(columns['target_id']))} "
          f"({n_cols} x {n_rows} grid) | raw samples: "
          f"{len(columns['target_id'])}")

    # ---------- Sample selection ----------
    keep = columns["sample_index"] >= args.drop_first

    # Eyelids lower when looking down, so the blink threshold is relative
    # to each target's own median openness, not the session median.
    openness = (columns["eye1_openness"] + columns["eye2_openness"]) / 2
    target_median = {
        g: np.median(openness[columns["target_id"] == g])
        for g in np.unique(columns["target_id"])
    }
    blink = openness < args.blink_ratio * np.array(
        [target_median[g] for g in columns["target_id"]]
    )
    print(f"Dropped as settling (first {args.drop_first} per target): "
          f"{int((~keep).sum())}")
    print(f"Possible blinks (openness < {args.blink_ratio} x target "
          f"median): {int(blink.sum())} ({100 * blink.mean():.1f}%) - "
          f"{'removed' if args.blink_filter else 'kept'}")

    if args.blink_filter:
        keep &= ~blink

    columns = {name: values[keep] for name, values in columns.items()}
    groups = columns["target_id"].astype(int)
    Y = np.column_stack([columns["target_x"], columns["target_y"]])

    missing = set(range(n_cols * n_rows)) - set(groups.tolist())
    if missing:
        print(f"WARNING: targets with no samples after filtering: "
              f"{sorted(missing)}")

    detection_gaps = columns["attempt_index"] - columns["sample_index"]
    print(f"Frames without a face before the last kept sample "
          f"(per target, max): {int(detection_gaps.max())}")

    # ---------- Feature stability ----------
    print("\n--- FEATURE STABILITY (all logged features) ---")
    feature_stability(
        columns, groups,
        LEGACY_FEATURES + CORNER_FEATURES + HEAD_FEATURES + QUALITY_FIELDS,
    )

    # ---------- Experiments ----------
    degrees = args.screen_width_cm and args.distance_cm
    run_time = time.strftime("%Y-%m-%d %H:%M:%S")
    summaries = []
    all_target_rows = []

    for name, features, kind in EXPERIMENTS:
        X = np.column_stack([columns[f] for f in features])

        predictions = leave_one_target_out(X, Y, groups, kind)
        errors = np.linalg.norm(predictions - Y, axis=1)
        results = per_target_rows(
            predictions, Y, groups,
            columns["grid_col"], columns["grid_row"],
        )
        inner = [r for r in results if is_inner(r, n_cols, n_rows)]
        outer = [r for r in results if not is_inner(r, n_cols, n_rows)]
        split_medians = grouped_splits(
            X, Y, groups, kind, args.repeats, args.test_fraction
        )

        summary = {
            "run_time": run_time,
            "session_id": session,
            "experiment": name,
            "model": kind,
            "features": "+".join(features),
            "drop_first": args.drop_first,
            "blink_filter": args.blink_filter,
            "samples": len(errors),
            "loto_mae_x": round(np.mean(np.abs(predictions[:, 0] - Y[:, 0])), 1),
            "loto_mae_y": round(np.mean(np.abs(predictions[:, 1] - Y[:, 1])), 1),
            "loto_mean_error": round(errors.mean(), 1),
            "loto_median_error": round(np.median(errors), 1),
            "loto_p90_error": round(np.percentile(errors, 90), 1),
            "inner_median_of_target_medians": round(
                np.median([r["median_error"] for r in inner]), 1
            ) if inner else "",
            "outer_median_of_target_medians": round(
                np.median([r["median_error"] for r in outer]), 1
            ) if outer else "",
            "mean_sd_x": round(np.mean([r["sd_x"] for r in results]), 1),
            "mean_sd_y": round(np.mean([r["sd_y"] for r in results]), 1),
            "gss_median": round(np.median(split_medians), 1),
            "gss_q25": round(np.percentile(split_medians, 25), 1),
            "gss_q75": round(np.percentile(split_medians, 75), 1),
        }
        summaries.append(summary)

        for r in results:
            all_target_rows.append(
                {"run_time": run_time, "session_id": session,
                 "experiment": name, **r}
            )

        print(f"\n--- {name} ---")
        print(f"Leave-one-target-out ({len(results)} folds): "
              f"MAE x={summary['loto_mae_x']} px, "
              f"MAE y={summary['loto_mae_y']} px, "
              f"Euclidean mean={summary['loto_mean_error']} / "
              f"median={summary['loto_median_error']} / "
              f"p90={summary['loto_p90_error']} px")
        print(f"Inner targets (interpolation) median: "
              f"{summary['inner_median_of_target_medians']} px | "
              f"edge targets (extrapolation) median: "
              f"{summary['outer_median_of_target_medians']} px")
        print(f"Precision while fixating: mean SD x={summary['mean_sd_x']}, "
              f"y={summary['mean_sd_y']} px")
        print(f"GroupShuffleSplit x{args.repeats} "
              f"({int(args.test_fraction * 100)}% targets held out): "
              f"median {summary['gss_median']} px "
              f"[IQR {summary['gss_q25']}-{summary['gss_q75']}]")
        if degrees:
            print(f"Median error in visual angle: "
                  f"{to_degrees(summary['loto_median_error'], screen_w, args.screen_width_cm, args.distance_cm):.2f} deg")
        print("Per-target median error (px), screen layout:")
        print_grid(results, "median_error", n_cols, n_rows)
        print("Per-target bias = |mean prediction - target| (px):")
        print_grid(results, "bias", n_cols, n_rows)

    # ---------- Comparison ----------
    print("\n=== SUMMARY (leave-one-target-out, held-out targets only) ===")
    print(f"{'experiment':<24}{'median':>8}{'MAE x':>8}{'MAE y':>8}"
          f"{'inner':>8}{'SD x':>7}{'SD y':>7}{'GSS':>7}")
    for s in summaries:
        print(f"{s['experiment']:<24}{s['loto_median_error']:>8}"
              f"{s['loto_mae_x']:>8}{s['loto_mae_y']:>8}"
              f"{s['inner_median_of_target_medians']:>8}"
              f"{s['mean_sd_x']:>7}{s['mean_sd_y']:>7}{s['gss_median']:>7}")

    half_line = READING_LINE_SPACING_PX / 2
    print(f"\nReference: half the current reading line spacing = "
          f"{half_line:.0f} px. Line-level assignment needs median "
          f"error well below this.")
    print("These are calibration-session errors at held-out targets. A "
          "separate test session\nis still required before claiming "
          "real-world accuracy.")

    # ---------- Save ----------
    for path, rows in ((PER_TARGET_CSV, all_target_rows),
                       (EXPERIMENT_LOG, summaries)):
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved: {PER_TARGET_CSV}, {EXPERIMENT_LOG} (appended)")


if __name__ == "__main__":
    main()

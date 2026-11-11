
"""
Pack one participant's data into a zip to send back to the researcher.

Includes the participant's session folders from data\\ and only their own
rows of the calibration samples. No images or video are recorded.

Usage:
    .\\.venv\\Scripts\\python.exe package_results.py P07
"""

import csv
import glob
import io
import os
import subprocess
import sys
import time
import zipfile

from validate_calibration import SAMPLES_CSV

RESULTS_DIR = "results"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: package_results.py <participant code>")
    participant = sys.argv[1]

    sessions = sorted(
        folder for folder in glob.glob(os.path.join("data", participant + "_*"))
        if os.path.isdir(folder)
    )

    calibration_rows = []
    if os.path.exists(SAMPLES_CSV):
        with open(SAMPLES_CSV, newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        calibration_rows = [rows[0]] + [
            row for row in rows[1:] if row[0].endswith("_" + participant)
        ]

    if not sessions and len(calibration_rows) <= 1:
        print(f"No data found for participant {participant}.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = os.path.join(RESULTS_DIR, f"{participant}_results_{stamp}.zip")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in sessions:
            for path in sorted(glob.glob(os.path.join(folder, "*"))):
                archive.write(path, "/".join(
                    ["data", os.path.basename(folder), os.path.basename(path)]
                ))
        if len(calibration_rows) > 1:
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerows(calibration_rows)
            archive.writestr(
                f"calibration/{participant}_calibration_samples.csv",
                buffer.getvalue(),
            )

    print("=" * 60)
    print("Results file (send this to the researcher):")
    print(os.path.abspath(output))
    print("=" * 60)

    # Open the folder with the file selected, for non-technical users.
    if "--no-open" not in sys.argv:
        subprocess.run(["explorer", "/select,", os.path.abspath(output)])


if __name__ == "__main__":
    main()

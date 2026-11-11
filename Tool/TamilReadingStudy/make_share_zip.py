
"""
Build TamilReadingStudy.zip to send to participants (researcher only).

Packs the scripts, items and instructions into a TamilReadingStudy\\
folder inside the zip. Leaves out everything made on this computer:
.venv, calibration (your own gaze model and samples), data, results,
caches and earlier zips.

Usage:
    .\\.venv\\Scripts\\python.exe make_share_zip.py
    .\\.venv\\Scripts\\python.exe make_share_zip.py --out E:\\share
"""

import argparse
import glob
import os
import zipfile

FOLDER = "TamilReadingStudy"
FILES = [
    "*.py", "*.bat", "requirements.txt",
    "Instructions_Tamil_Reading_Study.txt", "items/*.csv",
]
RESEARCHER_ONLY = {"make_share_zip.py", "build_tamil100_items.py",
                   "build_tamil_corpus.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="..",
                        help="folder for the zip (default: next to this "
                             "folder)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    paths = sorted({
        os.path.relpath(path, here)
        for pattern in FILES
        for path in glob.glob(os.path.join(here, pattern))
    })
    paths = [p for p in paths if os.path.basename(p) not in RESEARCHER_ONLY]

    output = os.path.abspath(os.path.join(args.out, FOLDER + ".zip"))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(os.path.join(here, path),
                          "/".join([FOLDER] + path.split(os.sep)))

    print(f"{len(paths)} files packed:")
    for path in paths:
        print("  " + path)
    print(f"\nShare this file: {output}")
    print("Send it together with Instructions_Tamil_Reading_Study.txt "
          "and the consent form.")


if __name__ == "__main__":
    main()

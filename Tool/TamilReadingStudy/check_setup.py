
"""
Check that this computer can run the reading study.

Usage:
    .\\.venv\\Scripts\\python.exe check_setup.py
    .\\.venv\\Scripts\\python.exe check_setup.py --no-camera
"""

import argparse
import ctypes
import platform
import sys
import time

problems = []


def report(ok, message):
    print(("  OK       " if ok else "  PROBLEM  ") + message)
    if not ok:
        problems.append(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()

    print("Checking this computer for the reading study...\n")

    report(platform.system() == "Windows", f"Windows ({platform.platform()})")
    report(sys.version_info[:2] == (3, 12),
           f"Python {platform.python_version()} (3.12 required)")

    try:
        import cv2
        import mediapipe
        import sklearn
        report(True, f"Libraries: mediapipe {mediapipe.__version__}, "
                     f"opencv {cv2.__version__}, scikit-learn "
                     f"{sklearn.__version__}")
    except ImportError as error:
        report(False, f"Library missing: {error}. Run setup.bat again.")
        return

    from hybrid_spr import MIN_FONT_SIZE, load_items
    from natural_reading import ITEMS_FILE, layout_words, split_words
    from tamil_render import TextRenderer

    width, _ = TextRenderer(32).measure("க")  # Tamil letter KA
    report(width > 0, "Tamil font (Nirmala UI)")

    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    items = load_items(ITEMS_FILE)
    fitted = None
    for size in range(32, MIN_FONT_SIZE - 1, -2):
        renderer = TextRenderer(size)
        try:
            lines = 0
            for item in items:
                boxes = layout_words(split_words(item)[0], renderer,
                                     screen_w, screen_h, 40)
                lines = max(lines, len({box[1] for box in boxes}))
            fitted = size
            break
        except ValueError:
            continue
        finally:
            renderer.close()
    report(fitted is not None,
           f"Screen {screen_w} x {screen_h}: {len(items)} sentences fit at "
           f"font size {fitted} px (up to {lines} lines)" if fitted else
           f"Screen {screen_w} x {screen_h}: sentences do not fit on the "
           "screen. Use a larger screen or lower display scaling.")

    if not args.no_camera:
        from gaze_features import create_face_mesh, measure

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            report(False, "Webcam could not be opened. Close other apps "
                          "using the camera (Zoom, Teams, Camera).")
        else:
            print("\n  Sit in front of the webcam for 4 seconds...")
            mesh = create_face_mesh()
            frames = faces = 0
            size = None
            end = time.time() + 4
            while time.time() < end:
                ok, frame = camera.read()
                if not ok:
                    continue
                size = (frame.shape[1], frame.shape[0])
                frames += 1
                faces += measure(frame, mesh) is not None
            camera.release()
            mesh.close()
            rate = 100 * faces / frames if frames else 0
            report(frames > 0, f"Webcam {size}: {frames / 4:.0f} frames/s")
            report(rate >= 90, f"Face and eyes detected in {rate:.0f}% of "
                               "frames (need 90%; improve lighting on "
                               "your face)")

    print()
    if problems:
        print(f"{len(problems)} problem(s) found. Fix them, then run "
              "check_setup again.")
        sys.exit(1)
    print("All checks passed. You can run run_session.bat.")


if __name__ == "__main__":
    main()

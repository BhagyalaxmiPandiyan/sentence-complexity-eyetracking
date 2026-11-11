
"""
Tamil text rendering and region layout for the reading experiment.

Text is drawn with the Windows GDI text engine through ctypes, which
shapes Tamil correctly (vowel-sign reordering, two-part vowels, pulli).
Pillow without libraqm and cv2.putText cannot do this. Canvases are
NumPy BGR images, so they can be shown with cv2.imshow.

Run this file directly to save a layout check image:
    .\\.venv\\Scripts\\python.exe tamil_render.py
"""

import ctypes
import unicodedata
from ctypes import wintypes

import numpy as np

DEFAULT_FONT = "Nirmala UI"

gdi32 = ctypes.windll.gdi32

# Handles must be pointer-sized; ctypes defaults to 32-bit int.
gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
gdi32.CreateDIBSection.restype = ctypes.c_void_p
gdi32.CreateDIBSection.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, wintypes.DWORD,
]
gdi32.CreateFontW.restype = ctypes.c_void_p
gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [
    wintypes.LPCWSTR
]
gdi32.SelectObject.restype = ctypes.c_void_p
gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
gdi32.SetBkMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
gdi32.SetTextColor.argtypes = [ctypes.c_void_p, wintypes.DWORD]
gdi32.TextOutW.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, wintypes.LPCWSTR,
    ctypes.c_int,
]
gdi32.GetTextExtentPoint32W.argtypes = [
    ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_int,
    ctypes.POINTER(wintypes.SIZE),
]

TRANSPARENT = 1
ANTIALIASED_QUALITY = 4
DEFAULT_CHARSET = 1


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class TextRenderer:
    """Draws shaped Unicode text onto BGR NumPy canvases."""

    def __init__(self, size_px=32, font_name=DEFAULT_FONT, bold=False):
        self.size_px = size_px
        self.hdc = gdi32.CreateCompatibleDC(None)
        self.font = gdi32.CreateFontW(
            -size_px, 0, 0, 0, 700 if bold else 400, 0, 0, 0,
            DEFAULT_CHARSET, 0, 0, ANTIALIASED_QUALITY, 0, font_name,
        )
        if not self.hdc or not self.font:
            raise RuntimeError(f"Could not create GDI font {font_name}")
        gdi32.SelectObject(self.hdc, self.font)
        gdi32.SetBkMode(self.hdc, TRANSPARENT)

    def measure(self, text):
        """Return (width, height) in pixels of the shaped text."""
        size = wintypes.SIZE()
        gdi32.GetTextExtentPoint32W(
            self.hdc, text, len(text), ctypes.byref(size)
        )
        return size.cx, size.cy

    def draw(self, canvas, text, x, y, color=(240, 240, 240)):
        """Draw text with its top-left corner at (x, y). color is BGR."""
        width, height = self.measure(text)
        x0, y0 = max(0, x), max(0, y)
        x1 = min(canvas.shape[1], x + width)
        y1 = min(canvas.shape[0], y + height)
        if x1 <= x0 or y1 <= y0:
            return

        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height  # top-down rows
        header.biPlanes = 1
        header.biBitCount = 32

        bits = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(
            self.hdc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0
        )
        if not bitmap:
            raise RuntimeError("CreateDIBSection failed")

        try:
            previous = gdi32.SelectObject(self.hdc, bitmap)
            buffer = np.ctypeslib.as_array(
                (ctypes.c_ubyte * (width * height * 4)).from_address(
                    bits.value
                )
            ).reshape(height, width, 4)

            # Start from the existing canvas so anti-aliasing blends in.
            buffer[:, :, :3] = 0
            buffer[y0 - y:y1 - y, x0 - x:x1 - x, :3] = canvas[y0:y1, x0:x1]

            b, g, r = color
            gdi32.SetTextColor(self.hdc, r | (g << 8) | (b << 16))
            gdi32.TextOutW(self.hdc, 0, 0, text, len(text))
            gdi32.GdiFlush()

            canvas[y0:y1, x0:x1] = buffer[y0 - y:y1 - y, x0 - x:x1 - x, :3]
            gdi32.SelectObject(self.hdc, previous)
        finally:
            gdi32.DeleteObject(bitmap)

    def close(self):
        if self.font:
            gdi32.DeleteObject(self.font)
            self.font = None
        if self.hdc:
            gdi32.DeleteDC(self.hdc)
            self.hdc = None


def count_graphemes(text):
    """Count visible letters: a base character plus its combining marks.

    Tamil vowel signs and pulli are combining marks, so கொ and க்
    each count as one letter, as a reader would count them.
    """
    count = 0
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char) in ("Mn", "Mc") and count:
            continue
        count += 1
    return count


def layout_regions(regions, renderer, screen_w, screen_h, gap_px):
    """Place regions on one horizontal line, centred on the screen.

    Returns a list of (x1, y1, x2, y2) boxes and the text top y.
    Raises ValueError if the sentence does not fit on one line.
    """
    sizes = [renderer.measure(text) for text in regions]
    total = sum(w for w, _ in sizes) + gap_px * (len(regions) - 1)
    if total > screen_w - 80:
        raise ValueError(
            f"Sentence is {total} px wide; screen allows {screen_w - 80}. "
            "Use fewer words or a smaller --font-size."
        )

    line_h = max(h for _, h in sizes)
    top = (screen_h - line_h) // 2
    x = (screen_w - total) // 2
    boxes = []
    for width, _ in sizes:
        boxes.append((x, top, x + width, top + line_h))
        x += width + gap_px
    return boxes, top


def draw_sentence(canvas, regions, boxes, revealed, renderer,
                  highlight=None):
    """Draw revealed regions; masked regions are shown as a grey bar.

    revealed: number of regions visible (cumulative presentation).
    highlight: optional region index to outline (look-back cue).
    """
    import cv2

    for index, (text, (x1, y1, x2, y2)) in enumerate(zip(regions, boxes)):
        if index < revealed:
            renderer.draw(canvas, text, x1, y1)
        else:
            bar_y = y2 - max(4, (y2 - y1) // 6)
            cv2.rectangle(canvas, (x1, bar_y), (x2, bar_y + 3),
                          (110, 110, 110), -1)

    if highlight is not None:
        x1, y1, x2, y2 = boxes[highlight]
        cv2.rectangle(canvas, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12),
                      (0, 200, 255), 3)


if __name__ == "__main__":
    import cv2

    renderer = TextRenderer(32)
    regions = ["கடைக்குச் சென்ற", "ராமன் ஒரு புத்தகம்", "வாங்கினான்,",
               "பிறகு வீடு திரும்பினான்."]
    canvas = np.zeros((960, 1536, 3), dtype=np.uint8)
    boxes, _ = layout_regions(regions, renderer, 1536, 960, 40)
    draw_sentence(canvas, regions, boxes, revealed=3, renderer=renderer)
    for box in boxes:
        cv2.rectangle(canvas, box[:2], box[2:], (0, 120, 0), 1)
    renderer.draw(canvas, "கொண்டு பள்ளியில் தேநீர் வைத்தாள் க்ஷ ஸ்ரீ",
                  80, 200)
    cv2.imwrite("layout_check.png", canvas)
    print("Saved layout_check.png  boxes:", boxes)
    print("Letter count of the first region:", count_graphemes(regions[0]))
    renderer.close()

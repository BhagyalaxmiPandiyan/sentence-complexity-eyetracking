"""
Step 0: download the Tamil Nadu state Tamil-language textbooks (Class 3-12)
that serve as graded readability ground truth.

Source: Google Drive mirrors linked from tntextbooks.net / tamilaruvi.in
(the official textbooksonline.tn.nic.in was unreachable). Mirror pages also
link unrelated books, so every candidate PDF is checked: its first pages must
name the expected class (e.g. "ஆறாம் வகுப்பு"), otherwise it is discarded.

Output: Dataset/graded_textbooks/class{N}.pdf  (+ manifest.csv)
Run:    python src/step0_fetch_textbooks.py
"""
import os, re, sys, csv, urllib.request
import pypdfium2 as pdfium
from tamil_text import repair_pdf_text

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "Dataset", "graded_textbooks")

# class -> (edition, candidate Google Drive file IDs, mirror page they came from)
CANDIDATES = {
    3:  ("Term 1", ["1LK0e6FLB3KdaVCnMwhiMYKhCqa5O6YBs", "1dWJfNE0Y2zcP8ZuCy3-J9BHnxEpLTf5a"],
         "https://www.tntextbooks.net/2022/12/class-3-term-1-tamil-and-english.html"),
    4:  ("Term 1", ["11Vn_7f2VK0nRFOfuVqJERf-70IOfckZ9"],
         "https://www.tntextbooks.net/2022/12/class-4-term-1-tamil-and-english.html"),
    5:  ("Term 1", ["10P94MT91L4wy3xAOLiZiHFKjnw5pX7Fg"],
         "https://www.tntextbooks.net/2022/12/class-5-term-1-tamil-and-english.html"),
    6:  ("Term 1", ["1VoooQkxhq6uqpOYr0oLVbwbiVmohONLn"],
         "https://www.tntextbooks.net/2022/12/class-6-term-1-tamil-and-english.html"),
    7:  ("Term 1", ["1QgQ9RMNoNMXfhUxJ7oFPe-28MyBUQ0oX", "1wne7-24cxV83PoTWW1s9YNksEZZELSvH"],
         "https://www.tntextbooks.net/2022/12/class-7-term-1-tamil-and-english.html"),
    8:  ("Full year", ["1GNQVi6tCf4AR8FVBIQdol6B4e_ALtMGH"],
         "https://www.tntextbooks.net/2025/08/class-8-tamil-full-textbook-2025-2026.html"),
    9:  ("Full year", ["1KbiKhIqZXpkUPz_kfP8LxrhmMkVtyhVD"],
         "https://www.tntextbooks.net/2022/12/class-9-tamil-full-textbook-2024.html"),
    10: ("Full year", ["1jYqCbgtRIWgULnm4lleRsTo9eb7Xn_8D"],
         "https://www.tamilaruvi.in/2022/12/class-10-tamil-full-textbook-2022-2023.html"),
    11: ("Full year", ["1OYhlp64q1Mx5NkQjdbvTWpLQ8CXonh6B", "1jvIf0MWnSIvWyi2LGfsx0V_cW7xJ0e0Y"],
         "https://www.tntextbooks.net/2022/12/class-11-tamil-full-textbook-2023.html"),
    12: ("Full year", ["1syhbydqEPDllh66-RvSuvbqAfJmCQncx", "1u1GnfbHXvW8xdQs49y9q2knrW10m1i4X"],
         "https://www.tntextbooks.net/2022/12/class-12-tamil-full-textbook-2023.html"),
}

# Title-page markers: "மூன்றாம் வகுப்பு" = third class
ORDINAL = {3: ["மூன்றாம் வகுப்பு"], 4: ["நான்காம் வகுப்பு"], 5: ["ஐந்தாம் வகுப்பு"],
           6: ["ஆறாம் வகுப்பு"], 7: ["ஏழாம் வகுப்பு"], 8: ["எட்டாம் வகுப்பு"],
           9: ["ஒன்பதாம் வகுப்பு"], 10: ["பத்தாம் வகுப்பு"],
           # higher secondary books say "மேல்நிலை முதலாம்/இரண்டாம் ஆண்டு" + "XI/XII Std"
           11: ["முதலாம் ஆண்டு", "XI Std"], 12: ["இரண்டாம் ஆண்டு", "XII Std"]}


def download(file_id, path):
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def names_class(path, n, pages=6):
    """Does the PDF's front matter name class n? Repair font artifacts first."""
    pdf = pdfium.PdfDocument(path)
    text = repair_pdf_text("".join(pdf[i].get_textpage().get_text_range() for i in range(min(pages, len(pdf)))))
    npages = len(pdf)
    pdf.close()                      # Windows: must close before the file can be moved/deleted
    return any(m in text for m in ORDINAL[n]), npages


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for n, (edition, ids, page) in CANDIDATES.items():
        final = os.path.join(OUT_DIR, f"class{n}.pdf")
        if os.path.exists(final):
            print(f"class {n}: already downloaded"); rows.append((n, edition, "", page, "existing")); continue
        for fid in ids:
            tmp = final + ".part"
            try:
                ok = download(fid, tmp)
            except Exception as e:
                print(f"class {n}: {fid} download failed ({e})"); continue
            if not ok:
                print(f"class {n}: {fid} is not a PDF"); os.remove(tmp); continue
            match, npages = names_class(tmp, n)
            print(f"class {n}: {fid} {npages} pages, names class {n}: {match}")
            if match:
                os.replace(tmp, final); rows.append((n, edition, fid, page, npages)); break
            os.remove(tmp)
        else:
            print(f"class {n}: NO MATCHING PDF FOUND")
    with open(os.path.join(OUT_DIR, "manifest.csv"), "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()

"""
Step 4: a second graded Tamil dataset with cleaner labels -- StoryWeaver reading levels.

Why: the textbook grade labels turned out to be too noisy at sentence level
(even Class 3 vs Class 12 sentences were only ~52% separable). StoryWeaver
(Pratham Books, CC BY 4.0) gives every children's book a reading level 1-4,
set by the publisher/editors for the book's language.

Full story text needs a login, but the public search API returns each book's
short description (blurb, usually 1-3 sentences, written at the book's level).
We use those blurbs as level-labelled Tamil text.

Output: output/storyweaver_sentences.csv   one row per sentence:
        book_id, level, title, sentence

Run: python src/step4_storyweaver_data.py      (~1 min, needs internet)
"""
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "output", "storyweaver_books.json")
OUT = os.path.join(BASE, "output", "storyweaver_sentences.csv")

API = "https://storyweaver.org.in/api/v1/books-search?languages%5B%5D=Tamil&per_page=100&page={}"
TAMIL = re.compile("[஀-௿]")
SPLIT = re.compile(r"(?<=[.?!])\s+")        # sentence boundary = . ? ! followed by space
MIN_WORDS = 3


def download():
    books, page, total = [], 1, None
    while total is None or page <= total:
        req = urllib.request.Request(API.format(page), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:          # default Python agent gets 403
            d = json.load(r)
        total = d["metadata"]["totalPages"]
        books += [{"book_id": b["id"], "level": b["level"], "title": b["title"],
                   "description": b.get("description") or ""} for b in d["data"]]
        print(f"  page {page}/{total}  ({len(books)} books)", flush=True)
        page += 1
        time.sleep(0.5)                         # be polite to the server
    with open(RAW, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False)
    return books


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(RAW):
        with open(RAW, encoding="utf-8") as f:
            books = json.load(f)
    else:
        books = download()

    rows, seen = [], set()
    for b in books:
        if not str(b["level"]).isdigit():
            continue
        text = re.sub(r"\s+", " ", b["description"]).strip()
        for s in SPLIT.split(text):
            s = s.strip()
            if len(s.split()) < MIN_WORDS or not TAMIL.search(s) or s in seen:
                continue
            seen.add(s)                          # same blurb reused by several editions
            rows.append({"book_id": b["book_id"], "level": int(b["level"]),
                         "title": b["title"], "sentence": s})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n{len(books)} books -> {len(out)} unique sentences ({out.book_id.nunique()} books)")
    print(out.groupby("level").agg(sentences=("sentence", "size"), books=("book_id", "nunique"),
                                   mean_words=("sentence", lambda x: x.str.split().str.len().mean()))
          .round(1).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

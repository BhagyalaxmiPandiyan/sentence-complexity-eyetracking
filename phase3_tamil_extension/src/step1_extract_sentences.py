"""
Step 1: extract graded PROSE sentences from the Tamil textbooks.

For each Dataset/graded_textbooks/class{N}.pdf:
  1. Repair font artifacts (tamil_text.repair_pdf_text).
  2. Keep only prose sections. Lessons open with a fixed heading:
        கவிதைப்பேழை   poetry         -> skip (classical/poetic Tamil would distort the test)
        உரைநடை உலகம்  prose          -> KEEP
        விரிவானம்      extra reading  -> KEEP (stories, drama, letters)
        கற்கண்டு       grammar        -> skip
     and the prose ends at the exercises (மதிப்பீடு / கற்பவை கற்றபின்)
     or an author note box (நூல் வெளி).
  3. Split into sentences and drop extraction garbage, judged by its SYMPTOMS
     (replacement characters, letter-spaced fragments, words starting with a
     bare vowel sign), NOT by word frequency: filtering out rare words would
     remove exactly the hard sentences and bias the evaluation.

Output: output/graded_sentences.csv  (class, page, section, sentence, n_words)
Run:    python src/step1_extract_sentences.py
"""
import os, re, sys, glob
import pandas as pd
import pypdfium2 as pdfium
from wordfreq import zipf_frequency
from tamil_text import repair_pdf_text, tamil_words, merge_split_words

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS = os.path.join(BASE, "Dataset", "graded_textbooks")
OUT = os.path.join(BASE, "output", "graded_sentences.csv")

SECTION_START = {"கவிதைப்பேழை": "poetry", "உரைநடை உலகம்": "prose",
                 "விரிவானம்": "extra_reading", "கற்கண்டு": "grammar"}
KEEP = {"prose", "extra_reading"}
SECTION_END = ["மதிப்பீடு", "கற்பவை கற்றபின்", "நூல் வெளி", "நூல்வெளி"]

MIN_WORDS, MAX_WORDS = 4, 40
SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
BARE_SIGN_START = re.compile(r"(^|\s)[ா-்]")     # word beginning with ா ி ு ் ... = broken glyph
PAGE_NUMBER = re.compile(r"^\d{1,3}\s+")                   # "61 எங்களுக்குள்..." (page number glued on)
SPEAKER = re.compile(r"^[஀-௿\s.]{1,25}\s*:\s*")  # "பாத்திமா : ..." (drama speaker label)
EXERCISE = re.compile(r"_{3,}|\.{4,}|\(?[அஆஇஈ]\)|\+")       # blanks, (அ) (ஆ) options, "வழி + தடம்"
# instruction-style sentence endings: imperative -க/-உக (எழுதுக, கூறுக, கண்டுபிடிக்க)
# and 1st-person activity verbs (மகிழ்வோம், எழுதுவேன்)
INSTRUCTION_END = re.compile(r"(க|வோம்|வேன்)\s*[.?!]?$")
# Sequences that cannot occur in valid Tamil, but do in text from the broken
# legacy font some pages use (e.g. "குழநரதைகபைாடு": ை followed by ா)
INVALID_SEQ = re.compile("[ா-ௌ][ா-்]|்[ா-ௌ]")
# Genuine Tamil words of <= 2 characters (monosyllables + a few particles)
SHORT_WORDS = set("நீ என வா போ பூ கை மை பை தீ ஐ ஆ ஓ ஈ ஏ ஊ தா மா பா வை தை கோ சீ நா கா கூ தே பே ஒ".split())
# Tamil words never start with a pulli consonant (ன்று) and native words never
# start with ண ழ ள ற ன ங, so such a token is the tail of a split word ("புதுமைக்கெ ன்று")
BAD_WORD_START = re.compile("[க-ஹ]்|[ணழளறனங]")
MIN_PAGE_KNOWN = 0.6      # page-level: skip pages where <60% of words are real (broken font)
TOC = "பொருளடக்கம்"       # table of contents: everything before it is preface/anthems


def is_verse(raw):
    """Prose wraps in long lines; verse/lists are short lines. The first and last
    line of a sentence may be fragments (it can start/end mid-line), so check the
    middle lines, and require at least one full-width line."""
    lens = [len(l.split()) for l in raw.split("\n") if l.strip()]
    if len(lens) < 2:
        return False
    return max(lens) < 5 or any(n < 4 for n in lens[1:-1])


def normalise(raw):
    s = re.sub(r"\s*\n\s*", " ", raw).strip(" \t\"'“”‘’")
    s = PAGE_NUMBER.sub("", s).lstrip("•●◆▪- ")
    s = SPEAKER.sub("", s).strip(" \"'“”‘’")
    return merge_split_words(s)          # "கிரா மத்தில்" -> "கிராமத்தில்"


def is_clean(sent):
    words = sent.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return False
    if "�" in sent or BARE_SIGN_START.search(sent) or INVALID_SEQ.search(sent):
        return False
    if EXERCISE.search(sent) or INSTRUCTION_END.search(sent):
        return False
    # leftover letter-spacing splits ("க ண் டு பி டி ப் பு"): a 1-2 character token
    # that is not one of Tamil's few genuine short words is a fragment. (Word
    # frequency can't decide this: fragments like பி score as high as நீ.)
    if any((len(w) <= 2 and w not in SHORT_WORDS) or BAD_WORD_START.match(w) for w in tamil_words(sent)):
        return False
    if len(re.findall(r"[A-Za-z]", sent)) > 0.1 * len(sent):      # mostly English
        return False
    return len(tamil_words(sent)) >= 0.8 * len(words)


def page_known(text):
    words = tamil_words(text)
    return sum(zipf_frequency(w, "ta") > 0 for w in words) / len(words) if len(words) >= 20 else 0.0


def book_sections(path):
    """Yield (page, section, text) chunks, switching section at each heading.

    Primary books (Class 3-5) have no prose headings; for those every page is
    yielded as section "all" and only the sentence filters apply.
    Pages before the table of contents, and pages in the broken legacy font,
    are skipped (page-level, so no bias against individual hard sentences)."""
    pdf = pdfium.PdfDocument(path)
    pages = [repair_pdf_text(pdf[i].get_textpage().get_text_range()).replace("\r", "")
             for i in range(len(pdf))]
    pdf.close()
    toc = next((i for i, t in enumerate(pages) if TOC in t), -1)
    usable = [(i, t) for i, t in enumerate(pages) if i > toc and page_known(t) >= MIN_PAGE_KNOWN]
    if sum(t.count("உரைநடை உலகம்") for t in pages) < 3:
        for i, text in usable:
            yield i, "all", text
        return
    section = None
    for i, text in usable:
        # find every heading on this page, in order, and cut the page there
        marks = sorted((m.start(), h) for h in list(SECTION_START) + SECTION_END
                       for m in re.finditer(re.escape(h), text))
        pos = 0
        for start, h in marks:
            yield i, section, text[pos:start]
            section = SECTION_START.get(h)          # an END marker -> None
            pos = start + len(h)
        yield i, section, text[pos:]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(BOOKS, "class*.pdf"))):
        cls = int(re.search(r"class(\d+)\.pdf", path).group(1))
        n_kept = 0
        for page, section, chunk in book_sections(path):
            if section not in KEEP and section != "all":
                continue
            for raw in SENT_SPLIT.split(chunk.strip()):
                s = normalise(raw)
                if not is_verse(raw.strip()) and is_clean(s):
                    rows.append((cls, page, section, s, len(s.split())))
                    n_kept += 1
        print(f"class {cls:2d}: {n_kept} prose sentences")
    df = pd.DataFrame(rows, columns=["class", "page", "section", "sentence", "n_words"])
    df = df.drop_duplicates(subset="sentence")
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {len(df)} sentences -> {OUT}")
    print(df.groupby("class").n_words.agg(["count", "mean"]).round(1).to_string())


if __name__ == "__main__":
    main()

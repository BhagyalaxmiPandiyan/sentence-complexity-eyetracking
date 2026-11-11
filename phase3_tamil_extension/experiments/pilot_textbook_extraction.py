"""
Pilot: can Tamil Nadu state textbooks (graded Class 1-12) be used as
readability ground truth? Extracts Tamil text from a textbook PDF, repairs the
font artifacts (doubled pulli / vowel signs / consonants), and measures the
share of extracted words that are real Tamil words (known to wordfreq).

Run: python experiments/pilot_textbook_extraction.py <textbook.pdf>
PDFs go in Dataset/graded_textbooks/ (not committed: government copyright, ~100MB each)
"""
import sys, re, pypdfium2 as pdfium
from wordfreq import zipf_frequency
sys.stdout.reconfigure(encoding="utf-8")
TA = re.compile(r"[஀-௿]+")
C = "[க-ஹ]"            # Tamil consonants
V = "[ா-்ௗ]"      # vowel signs + pulli
def clean(t):
    t = re.sub(rf"்்({C})\1", r"்\1", t)   # க்்கக -> க்க   (doubled pulli + doubled consonant)
    t = re.sub(r"்்", "்", t)                # ்் -> ்
    t = re.sub(rf"({V})\1", r"\1", t)                       # ொொ, ோோ -> single
    return t
pdf = pdfium.PdfDocument(sys.argv[1])
raw = "".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
cl = clean(raw)
known = lambda ws: sum(zipf_frequency(x, "ta") > 0 for x in ws) / len(ws)
w = TA.findall(cl)
print("known after cleaning %.1f%% (%d words)" % (100*known(w), len(w)))
i = cl.find("முகவுரை"); print(cl[i:i+330])
unk = [x for x in w if zipf_frequency(x,"ta")==0]
import collections; print("most common unknown:", collections.Counter(unk).most_common(25))

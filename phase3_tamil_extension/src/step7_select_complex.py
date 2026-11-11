"""
Step 7 (Phase B input): pick the complex Tamil sentences an LLM will simplify,
and write ready-to-paste prompts for a free web chat (no API key needed).

Input : output/sentence_features.csv     (step 3, textbooks)
        output/storyweaver_features.csv  (step 5)
Output: Dataset/complex_sentences/original.csv          id, source, level, sentence
        Dataset/complex_sentences/prompt_batch{1,2,3}.txt  20 sentences each

Selection: N_PER_SOURCE random sentences from the hardest level of each dataset
(textbook Class 11-12 prose, StoryWeaver level 4) that are long enough to be
worth simplifying (MIN_WORDS..MAX_WORDS words) and are clean Tamil (no digits,
no Latin letters, end with a full stop). Random, NOT ranked by our metric, so the
test set is not biased towards either metric.

How to use the prompts: see Dataset/complex_sentences/HOW_TO_SIMPLIFY.md.

Run: python src/step7_select_complex.py
"""
import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "Dataset", "complex_sentences")

N_PER_SOURCE = 30
MIN_WORDS, MAX_WORDS = 10, 30
BATCH = 20
SEED = 42

PROMPT = """You are an expert Tamil teacher. Rewrite each Tamil sentence below so that a child in Class 3-5 (about 8-10 years old) can read and understand it easily.

Rules:
1. Keep the same meaning. Do not add new information and do not drop important information.
2. Use common, everyday Tamil words instead of rare or literary words.
3. Use shorter words: avoid long words with many suffixes joined together where a simpler form exists.
4. You may split one long sentence into two or three short sentences.
5. Write natural, correct written Tamil (not colloquial spoken Tamil, not English, no transliteration).
6. If a sentence is already simple, still write the simplest version you can.

Output format: exactly one line per input sentence, in the same order, like this:
S01 | <simplified Tamil text>
S02 | <simplified Tamil text>
Do not write anything else: no explanations, no headings, no numbering other than the IDs.

Sentences:
"""


# Letters a native Tamil word never starts with, and common suffixes that the
# PDF extraction cut off from their word ('மேடை களிலும்', 'மரு தனை').
NO_WORD_START = tuple("யலளழறனணட")
SUFFIX_FRAGMENTS = {"கிறது", "கின்றன", "கள்", "களை", "களில்", "களிலும்", "தனை", "திலும்", "தில்"}
# Found by reading the selection: split words the rules miss, and verb-less fragments.
EXCLUDE = ["உண்ணா தவர்", "பா தை", "பாய்ந்ததான்", "நீட்டிப் படுத்து.", "வெள்ளைக் குதிரை.", "கருவை மரங்கள்"]
DOUBLED_LETTER = r"த்தத[ாேை]|ப்பப"     # font artifact: 'பார்த்ததேன்' for 'பார்த்தேன்'


def broken_word(sentence):
    """True if the PDF split a word in two (e.g. 'இருந்தனா லும்', 'முறை யாக').
    wordfreq cannot detect this: its web data contains the fragments too."""
    w = re.findall("[஀-௿]+", sentence)
    return any(b.startswith(NO_WORD_START) or b in SUFFIX_FRAGMENTS for b in w[1:])


def clean(s):
    return (s.str.endswith(".") & ~s.str[:-1].str.contains(r"[.?!]")    # one sentence only
            & ~s.str.contains(DOUBLED_LETTER)
            & ~s.str.contains(r"[=()]")                                   # garbled / bracketed text
            & ~s.str.contains("|".join(map(re.escape, EXCLUDE)))
            & ~s.str.contains(r"[0-9A-Za-z\u200b-\u200d]")      # digits, Latin, zero-width chars
            & ~s.str.contains(r"[\"“”'‘’]")                              # skip quoted dialogue
            & ~s.map(broken_word))


def main():
    t = pd.read_csv(os.path.join(BASE, "output", "sentence_features.csv"))
    s = pd.read_csv(os.path.join(BASE, "output", "storyweaver_features.csv"))
    t = t[(t["class"] >= 11)].assign(source="textbook", level=lambda d: d["class"])
    s = s[(s.level == 4)].assign(source="storyweaver")
    picked = []
    for d in (t, s):
        d = d.assign(sentence=d.sentence.str.replace(r"\s+([.,;])", r"\1", regex=True))  # "word ." -> "word."
        d = d[d.n_words.between(MIN_WORDS, MAX_WORDS) & clean(d.sentence)]
        d = d.drop_duplicates("sentence")
        picked.append(d.sample(N_PER_SOURCE, random_state=SEED)[["source", "level", "sentence"]])
    o = pd.concat(picked)
    o = o.sample(frac=1, random_state=SEED).reset_index(drop=True)
    o.insert(0, "id", [f"S{i + 1:02d}" for i in range(len(o))])

    os.makedirs(OUT_DIR, exist_ok=True)
    o.to_csv(os.path.join(OUT_DIR, "original.csv"), index=False, encoding="utf-8")
    for b in range(0, len(o), BATCH):
        part = o.iloc[b:b + BATCH]
        lines = "\n".join(f"{r.id} | {r.sentence}" for r in part.itertuples())
        path = os.path.join(OUT_DIR, f"prompt_batch{b // BATCH + 1}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(PROMPT + lines + "\n")
    print(f"wrote {len(o)} sentences ({o.source.value_counts().to_dict()}) "
          f"and {-(-len(o) // BATCH)} prompt files to {OUT_DIR}")


if __name__ == "__main__":
    main()

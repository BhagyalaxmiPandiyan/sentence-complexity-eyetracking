"""
Prepare Dundee sentences for input to the left-corner parser.
Reads lexical_features.csv (which has text_id, wnum, sent_id, word)
and outputs:
  1. A text file with one sentence per line (tokenized words, space-separated)
  2. A mapping file: line_number -> (text_id, sent_id, list of wnums)
"""
import os
import pandas as pd
import json

OUT = "/mnt/e/Project/Final One/output"
LEX_CSV = os.path.join(OUT, "lexical_features.csv")
SENT_FILE = os.path.join(OUT, "dundee_sentences.txt")
MAP_FILE  = os.path.join(OUT, "dundee_sent_map.json")

def main():
    df = pd.read_csv(LEX_CSV)
    print(f"Loaded {len(df)} rows from lexical_features.csv")

    # Group by (text_id, sent_id) to get ordered words per sentence
    # sent_id=0 means sentence boundary markers — skip those
    df = df[df["sent_id"] > 0].copy()
    df = df.sort_values(["text_id", "sent_id", "wnum"])

    sentences = []  # list of word lists
    sent_map  = []  # list of {"text_id":..., "sent_id":..., "wnums":[...]}

    for (text_id, sent_id), grp in df.groupby(["text_id", "sent_id"]):
        words = grp["word"].tolist()
        wnums = grp["wnum"].tolist()
        sentences.append(words)
        sent_map.append({
            "text_id": int(text_id),
            "sent_id": int(sent_id),
            "wnums": [int(w) for w in wnums]
        })

    print(f"Total sentences: {len(sentences)}")
    avg_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
    print(f"Average sentence length: {avg_len:.1f} words")

    # Write sentences file: one per line, words space-separated
    # Replace unknown/special tokens for parser compatibility
    def clean_word(w):
        if not isinstance(w, str):
            return "<unk>"
        w = w.strip()
        if not w:
            return "<unk>"
        return w.lower()

    with open(SENT_FILE, "w", encoding="utf-8") as f:
        for words in sentences:
            line = " ".join(clean_word(w) for w in words)
            f.write(line + "\n")

    # Write mapping file
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(sent_map, f)

    print(f"Wrote {SENT_FILE}")
    print(f"Wrote {MAP_FILE}")

if __name__ == "__main__":
    main()

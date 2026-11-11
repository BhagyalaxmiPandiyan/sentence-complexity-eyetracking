
"""
Build items\\tamil100_items.csv: the 100 Tamil news sentences of the
phase 3 study (UD Tamil-TTB / Dinamani, picked by
phase3_tamil_extension\\motr_study\\select_sentences.py), plus its
3 practice sentences and 20 "was this word in the sentence?" probes.

Researcher only; the shared zip already contains the CSV.

Usage:
    .\\.venv\\Scripts\\python.exe build_tamil100_items.py
"""

import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "..", "..", "phase3_tamil_extension", "motr_study")
SENTENCES = os.path.join(STUDY, "stimuli", "sentences.csv")
STIMULI_JS = os.path.join(STUDY, "web", "stimuli.js")
OUTPUT = os.path.join(HERE, "items", "tamil100_items.csv")

QUESTION = "{word}   —   இந்தச் சொல் வாக்கியத்தில் இருந்ததா?"


def practice_sentences():
    """PRACTICE list from stimuli.js (never among the 100)."""
    with open(STIMULI_JS, encoding="utf-8") as file:
        text = file.read()
    block = re.search(r"const PRACTICE = (\[.*?\]);", text, re.S).group(1)
    return json.loads(block)


def main():
    with open(SENTENCES, newline="", encoding="utf-8-sig") as file:
        sentences = list(csv.DictReader(file))

    rows = []
    for number, item in enumerate(practice_sentences()):
        question = answer = ""
        if number == 1:  # one demo question so the first real one is no surprise
            longest = max(item["text"].split(), key=len).strip(".,")
            question, answer = QUESTION.format(word=longest), "Y"
        rows.append([item["sid"], "practice", "", item["text"], "",
                     question, answer])

    for item in sentences:
        question = answer = ""
        if item["probe_word"]:
            question = QUESTION.format(word=item["probe_word"])
            answer = "Y" if item["probe_answer"] == "yes" else "N"
        rows.append([item["sid"], "sentence", item["band"], item["text"],
                     "", question, answer])

    with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "type", "condition", "regions",
                         "critical_region", "question", "answer"])
        writer.writerows(rows)
    print(f"{len(rows)} items ({len(sentences)} sentences) -> {OUTPUT}")


if __name__ == "__main__":
    main()

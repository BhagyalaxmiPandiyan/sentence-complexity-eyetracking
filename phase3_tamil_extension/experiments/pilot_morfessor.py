"""
Pilot check: can an unsupervised segmenter (Morfessor) give surface morpheme
splits for EVERY Tamil word, including the ones ThamizhiMorph's lexicon misses?

1. Streams ~N articles of Tamil Wikipedia (HF wikimedia/wikipedia, 20231101.ta)
2. Trains Morfessor Baseline on word counts
3. Segments every word of Dataset/raw_corpus/wiki_sample_20.txt

Run on Windows:  python experiments/pilot_morfessor.py
"""
import os, re, sys, collections
import morfessor
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_ARTICLES = 5000
MODEL = os.path.join(BASE, "output", "morfessor_ta_wiki.bin")
OUT = os.path.join(BASE, "output", "pilot_morfessor_segments.tsv")
TA = re.compile(r"[஀-௿]+")

counts = collections.Counter()
ds = load_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)
for i, art in enumerate(ds):
    if i >= N_ARTICLES:
        break
    counts.update(TA.findall(art["text"]))
print(f"trained on {N_ARTICLES} articles: {sum(counts.values()):,} tokens, {len(counts):,} word types")

io = morfessor.MorfessorIO()
model = morfessor.BaselineModel()
model.load_data([(c, w) for w, c in counts.items() if c >= 2])
model.train_batch()
io.write_binary_model_file(MODEL, model)

words = []
for line in open(os.path.join(BASE, "Dataset", "raw_corpus", "wiki_sample_20.txt"), encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3:
        words += TA.findall(parts[2])
with open(OUT, "w", encoding="utf-8") as f:
    f.write("word\tsegments\tn_morph\n")
    for w in dict.fromkeys(words):
        seg = model.viterbi_segment(w)[0]
        f.write(f"{w}\t{' + '.join(seg)}\t{len(seg)}\n")
print("wrote", OUT)

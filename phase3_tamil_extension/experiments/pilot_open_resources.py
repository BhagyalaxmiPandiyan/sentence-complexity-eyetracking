"""
Pilot check: which EXISTING open Tamil resources can replace home-made parts?

  1. IndicNLP morph segmenter (pretrained Morfessor model for Tamil,
     trained by IIT Bombay/AI4Bharat on a large Indic corpus)
  2. Stanza Tamil pipeline (lemma + UD morphological features, trained on UD Tamil-TTB)
  3. UD Tamil-TTB treebank: gold, hand-annotated morphology -> free evaluation set
  4. wordfreq: Tamil word frequencies (replaces Phase 1's wiki_freq feature)

Run on Windows:  python experiments/pilot_open_resources.py
"""
import os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(BASE, "tools")
TA = re.compile(r"[஀-௿]+")

WORDS = ["அழகுக்குப்", "இடைக்காலத்தில்", "அலங்காரங்களும்", "அளவுவிகிதங்களைக்",
         "கட்டமைப்புகளை", "ஒருங்கினைப்பையும்", "விருப்பத்துக்குரியது", "கட்டிடங்கள்",
         "அமைந்திருக்கிறது", "தத்துவத்தின்", "வடிவமைத்தல்", "கலைஞனதும்"]

# --- 1. IndicNLP pretrained Tamil segmenter
from indicnlp import common
common.set_resources_path(os.path.join(TOOLS, "indic_nlp_resources"))
from indicnlp.morph import unsupervised_morph
seg = unsupervised_morph.UnsupervisedMorphAnalyzer("ta")

# --- 2. Stanza Tamil
import stanza
stanza.download("ta", processors="tokenize,pos,lemma", verbose=False)
nlp = stanza.Pipeline("ta", processors="tokenize,pos,lemma", verbose=False, tokenize_pretokenized=True)

# --- 4. wordfreq
from wordfreq import zipf_frequency

print(f"{'word':22s} {'IndicNLP segments':38s} {'Stanza lemma':16s} {'Stanza features':45s} zipf")
doc = nlp([WORDS])
for w, tok in zip(WORDS, doc.sentences[0].words):
    print(f"{w:22s} {' + '.join(seg.morph_analyze(w)):38s} {tok.lemma or '':16s} {(tok.upos+'|'+(tok.feats or ''))[:45]:45s} {zipf_frequency(w,'ta'):.2f}")

# --- 3. UD Tamil-TTB gold set: how much can we reuse for evaluation?
n_sent = n_tok = n_mwt = 0
feats = collections.Counter()
for split in ["train", "dev", "test"]:
    for line in open(os.path.join(TOOLS, "UD_Tamil-TTB", f"ta_ttb-ud-{split}.conllu"), encoding="utf-8"):
        if line.startswith("# sent_id"):
            n_sent += 1
        elif line and line[0].isdigit():
            cols = line.split("\t")
            if "-" in cols[0]:
                n_mwt += 1
            else:
                n_tok += 1
                feats[len(cols[5].split("|")) if cols[5] != "_" else 0] += 1
print(f"\nUD Tamil-TTB gold: {n_sent} sentences, {n_tok} syntactic words, "
      f"{n_mwt} surface tokens split into several words (clitics/postpositions)")
print("features per word:", dict(sorted(feats.items())))

# --- 1b. IndicNLP segmenter on the full 20-sentence pilot sample
words = []
for line in open(os.path.join(BASE, "Dataset", "raw_corpus", "wiki_sample_20.txt"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 3:
        words += TA.findall(p[2])
n_morph = [len(seg.morph_analyze(w)) for w in words]
print(f"\nIndicNLP on pilot sample: mean {sum(n_morph)/len(n_morph):.2f} morphs/word, "
      f"{sum(n==1 for n in n_morph)/len(n_morph):.0%} left unsplit")

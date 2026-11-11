"""
Step 3: compute every complexity feature for every graded textbook sentence.

Input : ../phase3_tamil_extension/output/graded_sentences.csv  (2,351 sentences, Class 3-12)
Output: output/sentence_features.csv                          (one row per sentence)

Three feature groups:

  A. WORD-LEVEL (the English-style baseline, as in Singh et al. 2016 / our Phase 2)
     n_words          words in the sentence
     word_len         mean characters per word
     word_zipf        mean word-form frequency (wordfreq, Zipf scale: higher = more common)
     word_surprisal   mean GPT-2 surprisal per word, in bits

  B. MORPHOLOGY-AWARE (new)
     morphs_per_word  mean morphemes per word (IndicNLP pretrained Tamil Morfessor)
     stem_surprisal   mean surprisal of the first morpheme (the stem) of each word
     suffix_surprisal mean surprisal of everything after the stem (0 for unsplit words)
     feats_per_word   mean number of *marked* UD features per word (Stanza), e.g.
                      Case=Loc, Tense=Past, Number=Plur, VerbForm=Conv
     lemma_zipf       mean frequency of the lemma (dictionary form) instead of the word form
     nonfinite_verbs  words that are participles / converbs / infinitives / verbal nouns:
                      how Tamil builds subordinate clauses inside one sentence

  C. SYNTAX (Stanza dependency parse; replaces the paper's embedding-depth features)
     tree_depth       depth of the dependency tree
     dep_distance     mean distance (in words) between a word and its head

Surprisal: `abinayam/gpt-2-tamil` uses English byte BPE (~7 tokens per Tamil word),
so tokens never line up with morphemes. Each token's surprisal is credited to the
character it ends on, and a word's / morpheme's surprisal is the sum over its
character span.

Run: python src/step3_textbook_features.py        (CPU, ~10 min)
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3 = os.path.join(os.path.dirname(BASE), "phase3_tamil_extension")
IN = os.path.join(PHASE3, "output", "graded_sentences.csv")
OUT = os.path.join(BASE, "output", "sentence_features.csv")

LM_NAME = "abinayam/gpt-2-tamil"
WORD = re.compile(r"[^\s.,?!;:'\"“”‘’()\[\]\-–—…]+")    # a word = run of non-space, non-punctuation
TAMIL = re.compile("[஀-௿]+")

# UD feature values that are the unmarked default: they add no morphology on the surface
UNMARKED = {"Case=Nom", "Number=Sing", "Person=3", "Polarity=Pos", "Mood=Ind",
            "VerbForm=Fin", "Voice=Act"}
IGNORED = ("Gender=", "PronType=", "NumType=", "PunctType=", "Polite=", "Animacy=",
           "AdpType=", "NumForm=", "Reflex=")
NONFINITE = {"VerbForm=Part", "VerbForm=Ger", "VerbForm=Conv", "VerbForm=Inf", "VerbForm=Vnoun"}


# ---------------------------------------------------------------- tools
def load_tools():
    from indicnlp import common
    common.set_resources_path(os.path.join(PHASE3, "tools", "indic_nlp_resources"))
    from indicnlp.morph import unsupervised_morph
    import stanza
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seg = unsupervised_morph.UnsupervisedMorphAnalyzer("ta")
    nlp = stanza.Pipeline("ta", processors="tokenize,mwt,pos,lemma,depparse",
                          tokenize_no_ssplit=True, verbose=False, download_method=None)
    tok = AutoTokenizer.from_pretrained(LM_NAME)
    lm = AutoModelForCausalLM.from_pretrained(LM_NAME).eval()
    return seg, nlp, tok, lm


# ---------------------------------------------------------------- surprisal
@torch.no_grad()
def char_surprisal(sentence, tok, lm):
    """Surprisal (bits) credited to each character: token surprisal goes to the
    last character the token covers."""
    enc = tok(sentence, return_offsets_mapping=True)
    # start-of-text: the MODEL's bos id (50256). The tokenizer's own <|endoftext|> is 50265,
    # outside the model's 50257-row embedding; 50256 also gave the lowest surprisal on a sample.
    ids = [lm.config.bos_token_id] + enc.input_ids
    logits = lm(torch.tensor([ids])).logits[0, :-1]
    logp = torch.log_softmax(logits, -1)
    surp = -logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])] / math.log(2)
    per_char = np.zeros(len(sentence))
    for s, (_, end) in zip(surp.tolist(), enc.offset_mapping):
        per_char[max(end - 1, 0)] += s
    return per_char


# ---------------------------------------------------------------- stanza features
def parse_features(doc):
    words = [w for s in doc.sentences for w in s.words if w.upos != "PUNCT"]
    n = max(len(words), 1)
    marked = nonfinite = 0
    lemma_z = []
    from wordfreq import zipf_frequency
    for w in words:
        feats = (w.feats or "").split("|") if w.feats else []
        marked += sum(1 for f in feats if f not in UNMARKED and not f.startswith(IGNORED))
        if NONFINITE & set(feats):        # any POS: Stanza tags relative participles (தோன்றிய) as ADJ
            nonfinite += 1
        if w.lemma and TAMIL.search(w.lemma):
            lemma_z.append(zipf_frequency(w.lemma, "ta"))

    # syntax: tree depth and mean dependency distance (per Stanza sentence, then combined)
    depth, dist = 0, []
    for s in doc.sentences:
        head = {w.id: w.head for w in s.words}
        for w in s.words:
            d, h = 0, w.id
            while h and d < 100:
                h = head.get(h, 0); d += 1
            depth = max(depth, d)
            if w.head and w.upos != "PUNCT":
                dist.append(abs(w.id - w.head))
    return {
        "feats_per_word": marked / n,
        "lemma_zipf": float(np.mean(lemma_z)) if lemma_z else 0.0,
        "nonfinite_verbs": nonfinite,
        "tree_depth": depth,
        "dep_distance": float(np.mean(dist)) if dist else 0.0,
    }


# ---------------------------------------------------------------- one sentence
def sentence_features(sentence, seg, nlp, tok, lm):
    from wordfreq import zipf_frequency
    per_char = char_surprisal(sentence, tok, lm)

    lens, zipfs, w_surp, st_surp, sf_surp, n_morph = [], [], [], [], [], []
    for m in WORD.finditer(sentence):
        w, start = m.group(), m.start()
        lens.append(len(w))
        w_surp.append(per_char[start:m.end()].sum())
        if TAMIL.fullmatch(w):
            zipfs.append(zipf_frequency(w, "ta"))
            morphs = seg.morph_analyze(w)
            if "".join(morphs) != w:                      # segmenter changed the string: treat as 1 morph
                morphs = [w]
        else:
            morphs = [w]                                  # numbers, mixed tokens like 1920ஆம்
        n_morph.append(len(morphs))
        stem_end = start + len(morphs[0])
        st_surp.append(per_char[start:stem_end].sum())
        sf_surp.append(per_char[stem_end:m.end()].sum())

    f = {
        "n_words": len(lens),
        "word_len": float(np.mean(lens)),
        "word_zipf": float(np.mean(zipfs)) if zipfs else 0.0,
        "word_surprisal": float(np.mean(w_surp)),
        "morphs_per_word": float(np.mean(n_morph)),
        "stem_surprisal": float(np.mean(st_surp)),
        "suffix_surprisal": float(np.mean(sf_surp)),
    }
    f.update(parse_features(nlp(sentence)))
    return f


def main():
    s = pd.read_csv(IN)
    seg, nlp, tok, lm = load_tools()
    rows = []
    for i, sent in enumerate(s.sentence):
        rows.append(sentence_features(sent, seg, nlp, tok, lm))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(s)}", flush=True)
    out = pd.concat([s[["class", "page", "section", "sentence"]], pd.DataFrame(rows)], axis=1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}  ({len(out)} sentences)")
    print(out.drop(columns=["page", "section", "sentence"]).groupby("class").mean().round(2).to_string())


if __name__ == "__main__":
    main()

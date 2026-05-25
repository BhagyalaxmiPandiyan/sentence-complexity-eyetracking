"""
General-purpose syntactic feature extractor.
Works on any input sentence — not tied to the Dundee corpus.

Uses:
  - Penn Treebank WSJ POS bigrams for syntactic surprisal
  - spaCy en_core_web_sm for POS tagging + dependency parsing on novel text
  - AoA dict and wordfreq for lexical features

Exports:
  build_ptb_pos_lm()
  SyntacticExtractor(ptb_pos_lm, aoa_dict, wiki_freq_fn, nlp=None)
    .extract(sentence) -> list of feature dicts (one per word)
"""

import os
import re
import math
import pickle
import tarfile
from collections import defaultdict

import numpy as np

DATASET = "e:/Project/Final One/Dataset"
OUT     = "e:/Project/Final One/output"

PTB_POS_TAGS = frozenset([
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
    "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB", "#", "$", "''", "``", ",", ".", ":",
    "-LRB-", "-RRB-",
])

# Regex that matches (POS_TAG word) leaf nodes in PTB bracketed trees.
# POS tags are all-caps, may contain $, #, `, ', ", :, -, .
_LEAF_RE = re.compile(r'\(([A-Z$#`\'".:,\-]+)\s+([^()\s]+)\)')


def extract_pos_sequence_from_tree(tree_str: str) -> list:
    """Extract left-to-right POS sequence from one bracketed PTB tree string."""
    return [m.group(1) for m in _LEAF_RE.finditer(tree_str)
            if m.group(1) in PTB_POS_TAGS]


def build_ptb_pos_lm() -> tuple:
    """
    Build POS unigram + bigram counts from Penn Treebank WSJ .mrg files.
    Cached to output/ptb_pos_lm.pkl after the first run (~60 s).

    Returns (pos_uni: dict[str,int], pos_bi: dict[str, dict[str,int]]).
    """
    cache = os.path.join(OUT, "ptb_pos_lm.pkl")
    if os.path.exists(cache):
        print("  Loading PTB POS LM from cache...")
        with open(cache, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded: {sum(data[0].values()):,} unigrams, "
              f"{sum(sum(v.values()) for v in data[1].values()):,} bigrams")
        return data

    print("  Building PTB POS LM from Penn Treebank WSJ (first run — cached afterward)...")
    ptb_path = os.path.join(DATASET, "penn_treebank_3.tar.bz2")
    pos_uni = defaultdict(int)
    pos_bi  = defaultdict(lambda: defaultdict(int))
    n_sents = 0

    with tarfile.open(ptb_path, "r:bz2") as tf:
        wsj_mrg = [m for m in tf.getmembers()
                   if "parsed/mrg/wsj" in m.name and m.name.endswith(".mrg")]
        print(f"  Found {len(wsj_mrg)} WSJ .mrg files.")
        for m in wsj_mrg:
            try:
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                content = fobj.read().decode("utf-8", errors="ignore")
                # Split into individual top-level tree brackets
                depth, start = 0, 0
                for i, ch in enumerate(content):
                    if ch == "(":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            seq = extract_pos_sequence_from_tree(content[start:i + 1])
                            if seq:
                                n_sents += 1
                                for j, pos in enumerate(seq):
                                    pos_uni[pos] += 1
                                    if j > 0:
                                        pos_bi[seq[j - 1]][pos] += 1
            except Exception:
                continue

    # Freeze defaultdicts
    pos_bi = {k: dict(v) for k, v in pos_bi.items()}
    result = (dict(pos_uni), pos_bi)
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    print(f"  Built PTB POS LM from {n_sents:,} sentences.  "
          f"({sum(pos_uni.values()):,} tag tokens, {len(pos_uni)} unique tags)")
    return result


# ── Low-level probability helpers ────────────────────────────────────────────

def _pos_surp(pos_k: str, pos_prev, pos_uni: dict, pos_bi: dict) -> float:
    """Compute -log P(pos_k | pos_prev) with add-1 smoothing (nats)."""
    V = max(len(pos_uni), 1)
    if pos_prev is None:
        total = sum(pos_uni.values())
        p = (pos_uni.get(pos_k, 0) + 1) / (total + V)
    else:
        bi = pos_bi.get(pos_prev, {})
        p = (bi.get(pos_k, 0) + 1) / (sum(bi.values()) + V)
    return -math.log(max(p, 1e-12))


def _pos_entropy(pos_k: str, pos_bi: dict, pos_uni: dict) -> float:
    """Compute H(next_POS | pos_k) using PTB bigram model (nats)."""
    bi = pos_bi.get(pos_k, {})
    total = sum(bi.values())
    if total == 0:
        return math.log(max(len(pos_uni), 2))
    H = 0.0
    for cnt in bi.values():
        p = cnt / total
        H -= p * math.log(p)
    return H


# ── Main extractor class ─────────────────────────────────────────────────────

class SyntacticExtractor:
    """
    Feature extractor that works on any input sentence.

    Parameters
    ----------
    ptb_pos_lm : (pos_uni, pos_bi) from build_ptb_pos_lm()
    aoa_dict   : {word_lower: (mean_aoa, std_aoa)}
    wiki_freq_fn : callable(word_str) -> log_frequency  (negative float)
    nlp        : spaCy Language model (optional; falls back to "NN" tags + depth=0)
    """

    def __init__(self, ptb_pos_lm, aoa_dict, wiki_freq_fn, nlp=None):
        self.pos_uni, self.pos_bi = ptb_pos_lm
        self.aoa_dict      = aoa_dict
        self.wiki_freq_fn  = wiki_freq_fn
        self.nlp           = nlp

        vals = list(aoa_dict.values())
        self._default_aoa_mean = float(np.mean([v[0] for v in vals])) if vals else 6.0
        self._default_aoa_std  = float(np.mean([v[1] for v in vals])) if vals else 1.0

    def extract(self, sentence: str) -> list:
        """
        Return a list of feature dicts, one per whitespace-tokenized word.

        Feature keys match those used by System 1 (step4_system1.py):
          wlen, sent_len, wiki_freq, aoa_mean, aoa_std,
          total_surprisal, lex_surprisal, syn_surprisal,
          entropy_red, emb_depth, emb_diff,
          h1..h8
        """
        words = sentence.strip().split()
        if not words:
            return []

        sent_len  = len(words)
        pos_tags  = self._get_pos_tags(sentence, sent_len)
        dep_depths = self._get_dep_depths(sentence, sent_len)

        features  = []
        prev_pos  = None
        prev_H    = None
        prev_depth = None

        for i, word in enumerate(words):
            w    = re.sub(r"[^a-zA-Z]", "", word).lower()
            wlen = len(re.sub(r"[^a-zA-Z]", "", word))

            aoa_mean, aoa_std = self.aoa_dict.get(
                w, (self._default_aoa_mean, self._default_aoa_std))
            wfreq = self.wiki_freq_fn(w)   # log frequency (≤ 0)

            pos_k = pos_tags[i]
            depth = dep_depths[i]

            # lex_surprisal: -log P(word) via wordfreq (unigram; good proxy for novel text)
            lex_surp = -wfreq   # wfreq is log prob → -log prob = surprisal

            # syn_surprisal: -log P(POS_k | POS_{k-1}) from Penn Treebank
            syn_surp = _pos_surp(pos_k, prev_pos, self.pos_uni, self.pos_bi)

            total_surp = lex_surp + syn_surp

            # Entropy reduction from POS bigrams
            curr_H  = _pos_entropy(pos_k, self.pos_bi, self.pos_uni)
            ent_red = max(0.0, prev_H - curr_H) if prev_H is not None else 0.0

            # Embedding depth features
            emb_diff = depth - prev_depth if prev_depth is not None else 0

            # Hierarchical depth-transition features (h1-h8)
            if prev_depth is None:
                h1 = h2 = h3 = h4 = h6 = h7 = h8 = 0
            else:
                h1 = int(depth != prev_depth)
                h2 = int(depth > prev_depth)
                h3 = int(depth < prev_depth)
                h4 = int(depth == 0)
                h6 = int(depth - prev_depth > 1)
                h7 = int(prev_depth - depth > 1)
                h8 = int(depth == prev_depth)
            h5 = 0   # requires child-list computation; approximated as 0

            features.append({
                "wlen":            wlen,
                "sent_len":        sent_len,
                "wiki_freq":       round(wfreq, 4),
                "aoa_mean":        round(aoa_mean, 4),
                "aoa_std":         round(aoa_std, 4),
                "total_surprisal": round(total_surp, 4),
                "lex_surprisal":   round(lex_surp, 4),
                "syn_surprisal":   round(syn_surp, 4),
                "entropy_red":     round(ent_red, 4),
                "emb_depth":       depth,
                "emb_diff":        emb_diff,
                "h1": h1, "h2": h2, "h3": h3, "h4": h4,
                "h5": h5, "h6": h6, "h7": h7, "h8": h8,
            })

            prev_pos   = pos_k
            prev_H     = curr_H
            prev_depth = depth

        return features

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_pos_tags(self, sentence: str, n: int) -> list:
        """Return list of PTB POS tags (length n). Falls back to 'NN'."""
        if self.nlp is None:
            return ["NN"] * n
        try:
            doc = self.nlp(sentence)
            tags = [t.tag_ for t in doc]
            if len(tags) == n:
                return tags
            # Tokenization mismatch: align by word index as best-effort
            words = sentence.strip().split()
            result = []
            doc_iter = iter(doc)
            for word in words:
                t = next(doc_iter, None)
                result.append(t.tag_ if t is not None else "NN")
            return result
        except Exception:
            return ["NN"] * n

    def _get_dep_depths(self, sentence: str, n: int) -> list:
        """Return list of dependency depths (length n). Falls back to 0."""
        if self.nlp is None:
            return [0] * n
        try:
            doc = self.nlp(sentence)
            depths = [len(list(t.ancestors)) for t in doc]
            if len(depths) == n:
                return depths
            words = sentence.strip().split()
            result = []
            doc_iter = iter(doc)
            for word in words:
                t = next(doc_iter, None)
                result.append(len(list(t.ancestors)) if t is not None else 0)
            return result
        except Exception:
            return [0] * n


# ── Convenience: load spaCy ───────────────────────────────────────────────────

def load_spacy():
    """Load the best available spaCy English model."""
    import spacy
    for model in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            return spacy.load(model)
        except Exception:
            continue
    raise RuntimeError(
        "No spaCy English model found. Install one with:\n"
        "  python -m spacy download en_core_web_sm"
    )


# ── GPT-2 surprisal (general-purpose, works on any sentence) ─────────────────

def load_gpt2(model_name: str = "gpt2"):
    """
    Load GPT-2 tokenizer + model.  Returns (tokenizer, model).
    Works with transformers >= 4.x.  Model is set to eval mode.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tok, model


def gpt2_word_surprisals(sentence: str, gpt2_tok, gpt2_model) -> list:
    """
    Compute per-word GPT-2 surprisal (bits) for a sentence.

    GPT-2 tokenizes with BPE sub-words; tokens that start with 'Ġ' (U+0120)
    mark word boundaries.  We sum sub-word surprisals within each word.

    Returns a list of float surprisals, one per whitespace-tokenised word.
    If tokenization alignment fails, returns zeros.
    """
    import math
    import torch

    words = sentence.strip().split()
    if not words:
        return []

    try:
        # Prepend a space so the first word is treated as an interior word
        enc  = gpt2_tok(" " + sentence.strip(),
                        return_tensors="pt", add_special_tokens=False)
        ids  = enc.input_ids[0]
        toks = gpt2_tok.convert_ids_to_tokens(ids.tolist())

        if len(ids) < 2:
            return [0.0] * len(words)

        # Forward pass: input = tokens[0..T-2], predict tokens[1..T-1]
        with torch.no_grad():
            logits = gpt2_model(
                input_ids=ids[:-1].unsqueeze(0)
            ).logits[0]                         # (T-1, vocab)
        log_p = torch.log_softmax(logits, dim=-1)  # (T-1, vocab)

        # Sub-word surprisal for token at position k+1 given 0..k
        subword_surps = []
        for k in range(len(ids) - 1):
            lp = log_p[k, ids[k + 1]].item()
            subword_surps.append((-lp / math.log(2), toks[k + 1]))

        # Aggregate sub-words to words using 'Ġ' word-boundary marker
        word_surps   = []
        curr_surp    = 0.0
        words_so_far = 0

        for surp, tok_str in subword_surps:
            is_new_word = tok_str.startswith("Ġ") or tok_str.startswith("Ċ")
            if is_new_word and words_so_far > 0:
                word_surps.append(curr_surp)
                curr_surp = 0.0
            curr_surp += surp
            if is_new_word:
                words_so_far += 1

        if words_so_far > 0 or curr_surp > 0:
            word_surps.append(curr_surp)

        # Pad / truncate to match whitespace word count
        while len(word_surps) < len(words):
            word_surps.append(0.0)
        return word_surps[:len(words)]

    except Exception:
        return [0.0] * len(words)

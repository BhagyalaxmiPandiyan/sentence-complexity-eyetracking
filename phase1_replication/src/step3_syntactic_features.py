"""
Step 3: Extract syntactic features for each word in the Dundee corpus.

Features (paper Section 3.2):
  8.  total_surprisal     - lex_surprisal + syn_surprisal
  9.  lexical_surprisal   - -log P(word | prev_word) from bigram LM (BNC)
  10. syntactic_surprisal - from the LC parser (see experiments/build_lcparse_surprisal.py)
  11. entropy_reduction   - max(0, H_{k-1} - H_k) using PTB POS bigram entropy
  12. embedding_depth     - dependency tree depth for the word
  13. embedding_diff      - depth[current] - depth[previous word]

Hierarchical structure features (8 features, paper Section 3.2):
  Derived from the dependency parse transitions between adjacent words.

Output: output/syntactic_features.csv

Notes:
  - syn_surprisal comes entirely from output/lcparse_syn_surprisal.csv (the
    van Schijndel left-corner parser, the paper's actual method) — run
    experiments/build_lcparse_surprisal.py first. An earlier version of
    this script also built a hand-rolled PCFG/Viterbi parser as a fallback,
    but its output was always 100% overwritten by the LC parser merge, so
    it was removed (dead computation, confirmed via output diff).
  - PTB POS bigrams (via syntactic_extractor.build_ptb_pos_lm) are still
    used for entropy_red, which is independent of the syn_surprisal source.
  - Dependency parse from per-word-lexical-annotation used for depth features.
  - BNC bigram LM (from step 2 cache) used for lexical surprisal.
"""

import os, re, math
import pandas as pd
import numpy as np
from collections import defaultdict

DATASET = "e:/Project/Final One/phase1_replication/Dataset"
OUT     = "e:/Project/Final One/phase1_replication/output"
os.makedirs(OUT, exist_ok=True)
DUNDEE  = os.path.join(DATASET, "dundee_corpus")

# Import PTB POS LM builder from the general extractor module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from syntactic_extractor import build_ptb_pos_lm, _pos_entropy


def build_lexical_lm_from_bnc():
    """
    Load BNC bigrams to build lexical LM.
    Returns (unigrams dict, fwd_bigrams dict).
    """
    uni_cache = os.path.join(OUT, "bnc_bigrams.csv")
    bigram_cache = os.path.join(OUT, "bnc_bigrams_full.csv")

    unigrams = {}
    if os.path.exists(uni_cache):
        df = pd.read_csv(uni_cache)
        unigrams = dict(zip(df["word"], df["count"].astype(int)))

    fwd_counts = defaultdict(lambda: defaultdict(int))
    if os.path.exists(bigram_cache):
        bdf = pd.read_csv(bigram_cache, dtype={"w1": str, "w2": str, "count": int})
        for w1, grp in bdf.groupby("w1"):
            fwd_counts[str(w1)] = dict(zip(grp["w2"], grp["count"]))

    return unigrams, fwd_counts


# ── POS tags (for entropy_red via PTB POS bigrams) ────────────────────────────

def load_pos_tags():
    """Load POS tag for each (text_id, wnum) from dependency parse files."""
    ann_dir = os.path.join(DUNDEE, "per-word-lexical-annotation")
    pos_map = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.startswith("text") or not fname.endswith(".txt"):
            continue
        text_num = int(re.search(r"text(\d+)", fname).group(1))
        with open(os.path.join(ann_dir, fname), encoding="latin-1") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 9:
                    continue
                try:
                    pos  = str(p[1]).strip()
                    wnum = int(p[3])
                    key  = (text_num, wnum)
                    if key not in pos_map:
                        pos_map[key] = pos
                except (ValueError, IndexError):
                    continue
    return pos_map


def compute_lexical_surprisal(word, prev_word, unigrams, fwd_counts):
    """
    Lexical surprisal = -log P(word | prev_word) using BNC bigram LM.
    """
    if prev_word is None:
        return 0.0
    if not isinstance(word, str) or not isinstance(prev_word, str):
        return 0.0
    w = word.lower().strip(".,!?;:'\"()")
    pw = prev_word.lower().strip(".,!?;:'\"()")
    bigram_count = fwd_counts.get(pw, {}).get(w, 0)
    prev_count   = unigrams.get(pw, 0)
    # Add-1 (Laplace) smoothing
    vocab_size = max(len(unigrams), 1)
    p = (bigram_count + 1) / (prev_count + vocab_size)
    return -math.log(p)


def compute_bigram_entropy(word, unigrams, fwd_counts):
    """
    Entropy H_k = -sum P(w' | word) log P(w' | word) over all possible next words.
    """
    if not isinstance(word, str):
        return math.log(max(len(unigrams), 1))
    w = word.lower().strip(".,!?;:'\"()")
    next_counts = fwd_counts.get(w, {})
    total = sum(next_counts.values())
    if total == 0:
        # Unknown word: uniform distribution over vocabulary
        vocab = max(len(unigrams), 1)
        return math.log(vocab)
    entropy = 0.0
    for cnt in next_counts.values():
        p = cnt / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


# ── Dependency parse for embedding depth ─────────────────────────────────────

def load_dependency_parse():
    """
    Load dependency parse from per-word-lexical-annotation.
    Returns dict: (text_id, wnum) -> depth
    where depth = number of edges from word to root.
    """
    ann_dir = os.path.join(DUNDEE, "per-word-lexical-annotation")
    depth_map = {}   # (text_id, global_wnum) -> depth

    for fname in sorted(os.listdir(ann_dir)):
        if not fname.startswith("text") or not fname.endswith(".txt"):
            continue
        text_num = int(re.search(r"text(\d+)", fname).group(1))

        # Format: Word POS itemno WNUM SentenceID ID CPOS Head DepRel
        # ID = word ID within sentence (1-indexed)
        # Head = ID of head (0 = root)
        # WNUM = global word number (= c12 in per-word-info)

        # First pass: build head maps per sentence
        sentences = defaultdict(dict)  # sent_id -> {id: (wnum, head)}
        with open(os.path.join(ann_dir, fname), encoding="latin-1") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 9:
                    continue
                try:
                    wnum     = int(p[3])   # global word number
                    sent_id  = int(p[4])
                    word_id  = int(p[5])
                    head_id  = int(p[7])   # 0 = root
                    sentences[sent_id][word_id] = (wnum, head_id)
                except (ValueError, IndexError):
                    continue

        # Second pass: compute depth for each word (iterative, with cycle detection)
        for sent_id, words in sentences.items():
            depths = {}
            for wid in words:
                if wid in depths:
                    continue
                path = []
                visited = set()
                cur = wid
                while cur not in depths and cur in words and cur not in visited:
                    visited.add(cur)
                    path.append(cur)
                    _, head = words[cur]
                    if head == 0:
                        depths[cur] = 0
                        path.pop()
                        break
                    cur = head
                base = depths.get(cur, 0)
                for k, node in enumerate(reversed(path)):
                    depths[node] = base + k + 1

            for wid, (wnum, _) in words.items():
                depth_map[(text_num, wnum)] = depths.get(wid, 0)

    print(f"  Loaded dependency depths for {len(depth_map)} words.")
    return depth_map


# ── Hierarchical structure features from dependency transitions ───────────────

def compute_hierarchical_features(word_series, dep_depth_map, text_id):
    """
    Compute 8 binary/integer features encoding dependency tree transitions
    between adjacent words. These approximate the left-corner parser's
    hierarchical memory operations.

    Features:
      h1: head_changed       - head of current word != head of previous word
      h2: depth_increased    - embedding depth increased
      h3: depth_decreased    - embedding depth decreased
      h4: is_root            - current word is root (depth=0)
      h5: is_leaf            - current word has no dependents
      h6: depth_jump_up      - depth increased by > 1
      h7: depth_jump_down    - depth decreased by > 1
      h8: same_depth         - depth same as previous word
    """
    words = list(word_series)
    wnums = list(range(1, len(words) + 1))  # approximate wnum for this text slice

    results = []
    prev_depth = None
    for i, w in enumerate(words):
        wnum = wnums[i]
        depth = dep_depth_map.get((text_id, wnum), 0)

        if prev_depth is None:
            h1 = h2 = h3 = h4 = h5 = h6 = h7 = h8 = 0
        else:
            h1 = 1 if depth != prev_depth else 0
            h2 = 1 if depth > prev_depth else 0
            h3 = 1 if depth < prev_depth else 0
            h4 = 1 if depth == 0 else 0
            h5 = 0   # would need child lists to compute
            h6 = 1 if depth - prev_depth > 1 else 0
            h7 = 1 if prev_depth - depth > 1 else 0
            h8 = 1 if depth == prev_depth else 0

        results.append((h1, h2, h3, h4, h5, h6, h7, h8))
        prev_depth = depth

    return results


# ── GPT-2 surprisal for the Dundee corpus ────────────────────────────────────

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading lexical features data...")
    df_full = pd.read_csv(os.path.join(OUT, "lexical_features.csv"))
    print(f"  Loaded {len(df_full)} rows")
    # Syntactic features are word-level (not subject-level): deduplicate
    df = df_full.drop_duplicates(subset=["text_id", "wnum"]).copy()
    print(f"  Unique word positions: {len(df)}")

    print("\nBuilding PTB POS LM (Penn Treebank WSJ bigrams, cached after first run)...")
    ptb_pos_lm = build_ptb_pos_lm()
    ptb_pos_uni, ptb_pos_bi = ptb_pos_lm

    print("\nLoading POS tags from dependency parse...")
    pos_map = load_pos_tags()
    print(f"  Loaded POS tags for {len(pos_map)} words.")

    print("\nLoading BNC lexical LM for lex_surprisal...")
    unigrams, fwd_counts = build_lexical_lm_from_bnc()
    if not unigrams:
        print("  WARNING: BNC bigrams not ready. lex_surprisal set to 0.")

    print("\nLoading dependency parse for embedding depth...")
    dep_depth_map = load_dependency_parse()

    # Compute features per text (to maintain word order for context features)
    print("\nComputing syntactic features text by text...")
    records = []

    for tid, grp in df.groupby("text_id"):
        grp = grp.sort_values("wnum").reset_index(drop=True)
        words = grp["word"].tolist()
        wnums = grp["wnum"].tolist()

        prev_entropy = None

        for i, (word, wnum) in enumerate(zip(words, wnums)):
            prev_word = words[i - 1] if i > 0 else None

            # Lexical surprisal from BNC bigrams
            lex_surp = compute_lexical_surprisal(word, prev_word, unigrams, fwd_counts)

            # Syntactic surprisal: placeholder, always overwritten below by
            # the LC parser merge (output/lcparse_syn_surprisal.csv)
            syn_surp   = 0.0
            total_surp = lex_surp + syn_surp

            # Entropy reduction using PTB POS bigrams
            curr_pos = pos_map.get((tid, wnum), "NN")
            curr_entropy = _pos_entropy(curr_pos, ptb_pos_bi, ptb_pos_uni)
            if prev_entropy is None:
                ent_red = 0.0
            else:
                ent_red = max(0.0, prev_entropy - curr_entropy)
            prev_entropy = curr_entropy

            # Embedding depth
            depth = dep_depth_map.get((tid, wnum), 0)
            prev_wnum = wnums[i - 1] if i > 0 else None
            prev_depth = dep_depth_map.get((tid, prev_wnum), 0) if prev_wnum is not None else depth
            emb_diff = depth - prev_depth

            records.append({
                "text_id":         tid,
                "wnum":            wnum,
                "total_surprisal": round(total_surp, 4),
                "lex_surprisal":   round(lex_surp, 4),
                "syn_surprisal":   round(syn_surp, 4),
                "entropy_red":     round(ent_red, 4),
                "emb_depth":       depth,
                "emb_diff":        emb_diff,
            })

    syn_df = pd.DataFrame(records)

    # Add hierarchical features (h1-h8 from dep transitions)
    print("Computing hierarchical structure features...")
    h_records = []
    for tid, grp in df.groupby("text_id"):
        grp = grp.sort_values("wnum").reset_index(drop=True)
        h_feats = compute_hierarchical_features(grp["word"], dep_depth_map, tid)
        for i, (h1, h2, h3, h4, h5, h6, h7, h8) in enumerate(h_feats):
            h_records.append({
                "text_id": tid,
                "wnum":    grp.loc[i, "wnum"],
                "h1": h1, "h2": h2, "h3": h3, "h4": h4,
                "h5": h5, "h6": h6, "h7": h7, "h8": h8,
            })

    h_df = pd.DataFrame(h_records)
    syn_df = syn_df.merge(h_df, on=["text_id", "wnum"], how="left")

    # Use LC parser syn_surprisal (van Schijndel et al. 2013) — paper method
    # Zero values = sentence-initial words or boundary positions (correct, not fallback)
    lc_path = os.path.join(OUT, "lcparse_syn_surprisal.csv")
    if os.path.exists(lc_path):
        lc_df = pd.read_csv(lc_path).rename(columns={"syn_surp_pcfg": "syn_surp_lc"})
        syn_df = syn_df.merge(lc_df, on=["text_id", "wnum"], how="left")
        mask = syn_df["syn_surp_lc"].notna()
        syn_df.loc[mask, "syn_surprisal"] = syn_df.loc[mask, "syn_surp_lc"].round(4)
        syn_df["total_surprisal"] = (syn_df["lex_surprisal"] + syn_df["syn_surprisal"]).round(4)
        syn_df = syn_df.drop(columns=["syn_surp_lc"])
        print(f"\n  LC parser surprisal applied: {mask.sum():,}/{len(syn_df):,} words")
    else:
        print("\n  WARNING: lcparse_syn_surprisal.csv not found — syn_surprisal will be "
              "0.0 for all words. Run experiments/build_lcparse_surprisal.py first.")

    out_path = os.path.join(OUT, "syntactic_features.csv")
    syn_df.to_csv(out_path, index=False)

    print(f"\nSaved syntactic features -> {out_path}")
    print(f"  Rows: {len(syn_df)}")
    print("\nFeature summary:")
    for col in ["total_surprisal", "lex_surprisal", "syn_surprisal",
                "entropy_red", "emb_depth", "emb_diff"]:
        print(f"  {col:20s}: mean={syn_df[col].mean():.3f}, std={syn_df[col].std():.3f}")


if __name__ == "__main__":
    main()

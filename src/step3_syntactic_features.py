"""
Step 3: Extract syntactic features for each word in the Dundee corpus.

Features (paper Section 3.2):
  8.  total_surprisal     - -log P(word | context) from PCFG + bigram LM
  9.  lexical_surprisal   - -log P(word | prev_word) from bigram LM (BNC)
  10. syntactic_surprisal - total_surprisal - lexical_surprisal
  11. entropy_reduction   - max(0, H_{k-1} - H_k) using bigram entropy
  12. embedding_depth     - dependency tree depth for the word
  13. embedding_diff      - depth[current] - depth[previous word]

Hierarchical structure features (8 features, paper Section 3.2):
  Derived from the dependency parse transitions between adjacent words.

Output: output/syntactic_features.csv

Notes:
  - Penn Treebank .mrg files used to build PCFG for total surprisal.
  - Dependency parse from per-word-lexical-annotation used for depth features.
  - BNC bigram LM (from step 2 cache) used for lexical surprisal.
"""

import os, re, math, tarfile, io
import pandas as pd
import numpy as np
from collections import defaultdict

DATASET = "e:/Project/Final One/Dataset"
OUT     = "e:/Project/Final One/output"
os.makedirs(OUT, exist_ok=True)
DUNDEE  = os.path.join(DATASET, "dundee_corpus")

# Import PTB POS LM builder from the general extractor module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from syntactic_extractor import build_ptb_pos_lm, PTB_POS_TAGS, _pos_surp, _pos_entropy

PTB_POS_TAGS = frozenset([
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
    'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
    'WDT', 'WP', 'WP$', 'WRB', '#', '$', "''", '``', ',', '.', ':',
    '-LRB-', '-RRB-',
])
PTB_PHRASE_TAGS = frozenset([
    'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'NP', 'VP', 'PP', 'ADJP', 'ADVP',
    'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NX', 'PRN', 'PRT', 'QP', 'RRC',
    'UCP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X', 'TOP', 'ROOT',
])
PTB_ALL_NTS = PTB_POS_TAGS | PTB_PHRASE_TAGS


# ── Penn Treebank PCFG for total surprisal ───────────────────────────────────

def extract_productions_from_mrg(content):
    """
    Extract PCFG production rules from a bracketed parse tree string.
    Example: (S (NP (DT The) (NN cat)) (VP (VBD sat)))
    Returns list of (lhs, rhs_tuple) strings.
    """
    productions = []
    # Use a simple stack-based parser for bracketed trees
    stack = []
    i = 0
    while i < len(content):
        if content[i] == '(':
            # Start of a constituent
            # Read the label
            j = i + 1
            while j < len(content) and content[j] not in ' ()':
                j += 1
            label = content[i+1:j].strip()
            if label:
                stack.append((label, []))
            i = j
        elif content[i] == ')':
            if stack:
                lhs, children = stack.pop()
                if children:
                    rhs = tuple(children)
                    productions.append((lhs, rhs))
                if stack:
                    stack[-1][1].append(lhs)
            i += 1
        elif content[i] == ' ' or content[i] == '\n':
            i += 1
        else:
            # Read a terminal symbol
            j = i
            while j < len(content) and content[j] not in ' ()':
                j += 1
            terminal = content[i:j].strip()
            if terminal and stack:
                stack[-1][1].append(terminal)
            i = j
    return productions


def build_pcfg_from_treebank():
    """
    Extract WSJ .mrg files from Penn Treebank, build a PCFG.
    Cache the rule probabilities to output/pcfg_rules.csv.
    Returns: dict {(lhs, rhs): log_prob}
    """
    cache = os.path.join(OUT, "pcfg_rules.csv")
    if os.path.exists(cache):
        print("  Loading PCFG from cache...")
        df = pd.read_csv(cache)
        rules = {(lhs, rhs): float(lp)
                 for lhs, rhs, lp in zip(df["lhs"], df["rhs"], df["log_prob"])}
        print(f"  Loaded {len(rules)} PCFG rules.")
        return rules

    print("  Extracting Penn Treebank WSJ .mrg files...")
    ptb_path = os.path.join(DATASET, "penn_treebank_3.tar.bz2")

    lhs_counts  = defaultdict(int)          # lhs -> total count
    rule_counts = defaultdict(int)          # (lhs, rhs_str) -> count

    with tarfile.open(ptb_path, "r:bz2") as tf:
        members = tf.getmembers()
        wsj_mrg = [m for m in members if "parsed/mrg/wsj" in m.name and m.name.endswith(".mrg")]
        print(f"  Found {len(wsj_mrg)} WSJ .mrg files. Parsing...")

        for m in wsj_mrg:
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                content = f.read().decode("utf-8", errors="ignore")
                prods = extract_productions_from_mrg(content)
                for lhs, rhs in prods:
                    rhs_str = " ".join(rhs)
                    lhs_counts[lhs] += 1
                    rule_counts[(lhs, rhs_str)] += 1
            except Exception:
                continue

    print(f"  Total unique rules: {len(rule_counts)}")

    # Compute log probabilities: log P(rhs | lhs)
    rows = []
    rules = {}
    for (lhs, rhs_str), count in rule_counts.items():
        lhs_total = lhs_counts[lhs]
        log_p = math.log(count / lhs_total)
        rules[(lhs, rhs_str)] = log_p
        rows.append({"lhs": lhs, "rhs": rhs_str, "log_prob": log_p})

    pd.DataFrame(rows).to_csv(cache, index=False)
    print(f"  PCFG saved to cache ({len(rules)} rules).")
    return rules


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


# ── POS tags + PCFG surprisal ─────────────────────────────────────────────────

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


def _strip_func_tag(s):
    """Strip PTB functional tags: NP-SBJ -> NP, VP-TPC-1 -> VP, NNP -> NNP."""
    if s in PTB_ALL_NTS:
        return s
    for sep in ("-", "="):
        idx = s.find(sep)
        if idx > 0:
            base = s[:idx]
            if base in PTB_ALL_NTS:
                return base
    return None  # unknown symbol


def build_pos_level_pcfg_parser(pcfg_rules, min_log_prob=-5.0):
    """
    Build an NLTK ViterbiParser over POS-tag sequences.

    Strips PTB functional tags (NP-SBJ -> NP, VP-TPC -> VP) so that
    core phrase-structure rules like S -> NP VP are captured.
    All symbols become non-terminals; each POS tag gets a lexical rule
    POS -> <POS> (placeholder terminal). ViterbiParser (CYK) requires CNF.
    """
    try:
        from nltk.grammar import ProbabilisticProduction, Nonterminal, PCFG
        from nltk.parse import ViterbiParser
    except ImportError:
        print("  NLTK not available — skipping PCFG surprisal.")
        return None

    rules_by_lhs = defaultdict(list)
    for (lhs, rhs_str), log_prob in pcfg_rules.items():
        if log_prob < min_log_prob:
            continue
        lhs_base = _strip_func_tag(lhs)
        if lhs_base is None or lhs_base not in PTB_PHRASE_TAGS:
            continue
        rhs_parts = rhs_str.split()
        rhs_base = [_strip_func_tag(p) for p in rhs_parts]
        # Skip rules with unknown symbols or trace nodes (-NONE-)
        if any(b is None or b == "-NONE-" for b in rhs_base):
            continue
        rules_by_lhs[lhs_base].append((rhs_base, math.exp(log_prob)))

    if not rules_by_lhs:
        print("  No phrase structure rules found — skipping PCFG surprisal.")
        return None

    productions = []
    synth_id = [0]
    used_pos = set()

    def add_binarized(lhs_nt, rhs_nts, prob):
        if len(rhs_nts) <= 2:
            productions.append(ProbabilisticProduction(lhs_nt, rhs_nts, prob=max(prob, 1e-12)))
        else:
            synth_id[0] += 1
            synth_nt = Nonterminal(f"@{lhs_nt.symbol()}{synth_id[0]}")
            productions.append(
                ProbabilisticProduction(lhs_nt, [rhs_nts[0], synth_nt], prob=max(prob, 1e-12))
            )
            add_binarized(synth_nt, rhs_nts[1:], prob=1.0)

    for lhs_str, rhs_list in rules_by_lhs.items():
        lhs_nt = Nonterminal(lhs_str)
        total = sum(p for _, p in rhs_list)
        if total <= 0:
            continue
        for rhs_parts, prob in rhs_list:
            normalized = prob / total
            rhs_nts = [Nonterminal(s) for s in rhs_parts]
            add_binarized(lhs_nt, rhs_nts, normalized)
            for s in rhs_parts:
                if s in PTB_POS_TAGS:
                    used_pos.add(s)

    for pos in used_pos:
        productions.append(
            ProbabilisticProduction(Nonterminal(pos), [f"<{pos}>"], prob=1.0)
        )

    try:
        grammar = PCFG(Nonterminal("S"), productions)
        parser = ViterbiParser(grammar)
        print(f"  PCFG parser: {len(productions)} productions, {len(used_pos)} POS terminals.")
        return parser
    except Exception as e:
        print(f"  Failed to build PCFG parser: {e}")
        return None


def compute_pcfg_surprisals(df, ptb_pos_lm, pos_map):
    """
    Compute structural (syntactic) surprisal for each Dundee word using
    Penn Treebank POS bigrams.

        syn_surp_k = -log P(POS_k | POS_{k-1})

    POS tags come from the Dundee dependency-parse annotation files (accurate
    for the corpus words). The bigram model is estimated from the full PTB WSJ
    (3,348 files, ~50 K sentences) with add-1 smoothing — far more reliable
    than a model built on the 2,378 Dundee sentences.

    Returns dict: (text_id, wnum) -> structural_surprisal (nats).
    """
    pos_uni, pos_bi = ptb_pos_lm

    pos_df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in pos_map.items()],
        columns=["text_id", "wnum", "pos"]
    )
    merged = df.merge(pos_df, on=["text_id", "wnum"], how="left")
    merged["pos"] = merged["pos"].fillna("NN")

    syn_surp_map = {}

    for sid, grp in merged[merged["sent_id"] > 0].groupby("sent_id"):
        grp_s = grp.sort_values("wnum")
        tids  = grp_s["text_id"].astype(int).tolist()
        wnums = grp_s["wnum"].astype(int).tolist()
        seq   = grp_s["pos"].tolist()

        prev_pos = None
        for tid, wnum, pos in zip(tids, wnums, seq):
            syn_surp_map[(tid, wnum)] = _pos_surp(pos, prev_pos, pos_uni, pos_bi)
            prev_pos = pos

    print(f"  PTB POS bigram syn_surprisal computed for {len(syn_surp_map)} words.")
    return syn_surp_map


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

    print("\nComputing syntactic surprisal using PTB POS bigrams...")
    syn_surp_map = compute_pcfg_surprisals(df, ptb_pos_lm, pos_map)

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

            # Syntactic surprisal from PTB POS bigrams
            syn_surp   = syn_surp_map.get((tid, wnum), 0.0)
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

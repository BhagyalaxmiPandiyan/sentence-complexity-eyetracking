  """
Step 2: Extract lexical features for each word in the Dundee corpus.

Features (paper Section 3.2):
  1. word_length     - number of characters (from Dundee, already in dundee_rt.csv)
  2. sent_length     - number of words in the sentence
  3. wiki_freq       - log(unigram count + 1) from Wikipedia (or Brown corpus fallback)
  4. aoa_mean        - mean age of acquisition (years)
  5. aoa_std         - std dev of age of acquisition
  6. fwd_prob        - forward transition probability P(w | prev_w) from BNC
  7. bwd_prob        - backward transition probability P(w | next_w) from BNC

Output: output/lexical_features.csv

Note: BNC parsing is slow on first run (~20-40 min for 4049 files).
      Result is cached to output/bnc_bigrams.csv for reuse.
"""

import os, re, math, gzip
import pandas as pd
import numpy as np
from collections import defaultdict

DATASET = "e:/Project/Final One/Dataset"
OUT     = "e:/Project/Final One/output"
os.makedirs(OUT, exist_ok=True)


# ── Feature 3: Wikipedia word frequency ──────────────────────────────────────

def load_wiki_freq():
    """
    Load Wikipedia-based word frequencies via the wordfreq library.
    Returns dict: word_lower -> log(word_frequency + 1e-9)
    """
    from wordfreq import word_frequency
    rt = pd.read_csv(os.path.join(OUT, "dundee_rt.csv"))
    unique_words = rt["word"].dropna().str.lower().str.strip(".,!?;:'\"()").unique()
    result = {w: math.log(word_frequency(w, 'en') + 1e-9) for w in unique_words}
    print(f"  Loaded wordfreq frequencies for {len(result)} unique Dundee words.")
    return result


# ── Feature 4 & 5: Age of Acquisition ────────────────────────────────────────

def load_aoa():
    """
    Load AoA data from AoA.csv.
    Returns dict: word_lower -> (mean_aoa, std_aoa)
    """
    aoa_path = os.path.join(DATASET, "AoA.csv")
    df = pd.read_csv(aoa_path, encoding="latin-1")

    # Use AoA_Kup as mean (Kuperman et al. 2012)
    # Use AoA_Bristol_lem and AoA_Kup_lem to estimate spread (std)
    # If multiple measures available, take mean as best estimate of std
    aoa_dict = {}
    for _, row in df.iterrows():
        word = str(row["Word"]).lower().strip()
        mean_aoa = row.get("AoA_Kup", np.nan)
        if pd.isna(mean_aoa):
            mean_aoa = row.get("AoA_Kup_lem", np.nan)

        # Estimate std: use spread between different AoA measures
        vals = []
        for col in ["AoA_Kup", "AoA_Bird_lem", "AoA_Bristol_lem", "AoA_Cort_lem"]:
            v = row.get(col, np.nan)
            if not pd.isna(v):
                vals.append(v)
        std_aoa = np.std(vals) if len(vals) > 1 else 0.0

        if not pd.isna(mean_aoa):
            aoa_dict[word] = (float(mean_aoa), float(std_aoa))

    print(f"  Loaded AoA for {len(aoa_dict)} words.")
    return aoa_dict


# ── Feature 6 & 7: BNC Bigram Probabilities ──────────────────────────────────

def extract_bnc_words_from_file(path):
    """Extract ordered list of lowercase words from one BNC XML file."""
    words = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Extract <w ...>word</w> tags (word tokens)
        for m in re.finditer(r"<w [^>]*>([^<]+)</w>", content):
            w = m.group(1).strip().lower()
            if w and w.isalpha():
                words.append(w)
    except Exception:
        pass
    return words


def build_bnc_bigrams():
    """
    Walk all BNC XML files, collect all word tokens in order,
    count bigrams for forward and backward transition probabilities.
    Returns (unigram_counts, bigram_fwd_counts, bigram_bwd_counts).
    Caches result to output/bnc_bigrams.csv.
    """
    cache = os.path.join(OUT, "bnc_bigrams.csv")
    if os.path.exists(cache):
        print("  Loading BNC bigrams from cache...")
        df = pd.read_csv(cache)
        unigrams  = dict(zip(df["word"], df["count"]))
        # rebuild fwd and bwd from cached data
        fwd = defaultdict(lambda: defaultdict(int))
        bwd = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            w = row["word"]
            fwd_str = str(row.get("fwd_pairs", ""))
            bwd_str = str(row.get("bwd_pairs", ""))
            # simple storage: not caching bigrams, recompute probs from counts
        # If only unigrams cached, bigrams not available → recompute
        # Check if bigram file exists separately
        bigram_cache = os.path.join(OUT, "bnc_bigrams_full.csv")
        if os.path.exists(bigram_cache):
            bdf = pd.read_csv(bigram_cache)
            fwd_counts = defaultdict(lambda: defaultdict(int))
            bwd_counts = defaultdict(lambda: defaultdict(int))
            for _, row in bdf.iterrows():
                w1, w2, cnt = row["w1"], row["w2"], int(row["count"])
                fwd_counts[w1][w2] = cnt
                bwd_counts[w2][w1] = cnt
            return unigrams, fwd_counts, bwd_counts
        return unigrams, None, None

    bnc_root = os.path.join(DATASET, "BNC", "download", "Texts")
    all_words = []
    file_count = 0
    print("  Scanning BNC XML files (this takes 20-40 minutes on first run)...")

    for root, dirs, files in os.walk(bnc_root):
        for fname in files:
            if not fname.endswith(".xml"):
                continue
            file_count += 1
            if file_count % 500 == 0:
                print(f"    Processed {file_count} files, {len(all_words):,} words so far...")
            path = os.path.join(root, fname)
            all_words.extend(extract_bnc_words_from_file(path))

    print(f"  Total BNC words extracted: {len(all_words):,} from {file_count} files")

    # Unigram counts
    unigrams = defaultdict(int)
    for w in all_words:
        unigrams[w] += 1

    # Bigram counts
    fwd_counts = defaultdict(lambda: defaultdict(int))  # fwd[w1][w2] = count(w2 follows w1)
    bwd_counts = defaultdict(lambda: defaultdict(int))  # bwd[w2][w1] = count(w1 precedes w2)
    for i in range(len(all_words) - 1):
        w1, w2 = all_words[i], all_words[i + 1]
        fwd_counts[w1][w2] += 1
        bwd_counts[w2][w1] += 1

    # Save unigrams to cache
    rows = [{"word": w, "count": c} for w, c in unigrams.items()]
    pd.DataFrame(rows).to_csv(cache, index=False)

    # Save bigrams to separate cache
    bigram_rows = []
    for w1, nexts in fwd_counts.items():
        for w2, cnt in nexts.items():
            bigram_rows.append({"w1": w1, "w2": w2, "count": cnt})
    pd.DataFrame(bigram_rows).to_csv(os.path.join(OUT, "bnc_bigrams_full.csv"), index=False)

    print(f"  Saved BNC caches.")
    return dict(unigrams), fwd_counts, bwd_counts


def compute_transition_probs(word_list, fwd_counts, bwd_counts, unigrams):
    """
    Compute forward and backward transition probabilities for a list of words.
    P(w_k | w_{k-1}) = count(w_{k-1}, w_k) / count(w_{k-1})  [with +1 smoothing]
    P(w_k | w_{k+1}) = count(w_{k+1}, w_k) / count(w_{k+1})
    Returns: list of (fwd_prob, bwd_prob) tuples, same length as word_list.
    """
    result = []
    for i, raw_w in enumerate(word_list):
        w = str(raw_w).lower().strip(".,!?;:'\"()") if isinstance(raw_w, str) else ""

        # Forward: P(w | prev)
        if i > 0 and fwd_counts is not None:
            raw_p = word_list[i - 1]
            prev = str(raw_p).lower().strip(".,!?;:'\"()") if isinstance(raw_p, str) else ""
            fwd_count = fwd_counts.get(prev, {}).get(w, 0)
            prev_total = unigrams.get(prev, 0)
            fwd_p = (fwd_count + 1) / (prev_total + len(unigrams))
        else:
            fwd_p = 1.0 / max(len(unigrams), 1)

        # Backward: P(w | next)
        if i < len(word_list) - 1 and bwd_counts is not None:
            raw_n = word_list[i + 1]
            nxt = str(raw_n).lower().strip(".,!?;:'\"()") if isinstance(raw_n, str) else ""
            bwd_count = bwd_counts.get(w, {}).get(nxt, 0)
            nxt_total = unigrams.get(nxt, 0)
            bwd_p = (bwd_count + 1) / (nxt_total + len(unigrams))
        else:
            bwd_p = 1.0 / max(len(unigrams), 1)

        result.append((math.log(fwd_p), math.log(bwd_p)))
    return result


# ── Feature 2: Sentence length ────────────────────────────────────────────────

def compute_sentence_lengths(rt_df):
    """Add sent_len column: number of words in each sentence."""
    sent_sizes = rt_df.groupby("sent_id").size().rename("sent_len")
    return rt_df.merge(sent_sizes.reset_index(), on="sent_id", how="left")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading Dundee RT data...")
    rt_df = pd.read_csv(os.path.join(OUT, "dundee_rt.csv"))
    print(f"  Loaded {len(rt_df)} per-subject observations")

    # Feature 1 (word_length) is already in rt_df as 'wlen'

    # Work on unique (text_id, wnum) pairs for word-level features,
    # then merge back to the full per-subject table.
    word_df = rt_df.drop_duplicates(subset=["text_id", "wnum"]).copy()
    print(f"  Unique word positions: {len(word_df)}")

    # Feature 2: sentence length (computed on unique words per sentence)
    print("Computing sentence lengths...")
    word_df = compute_sentence_lengths(word_df)

    # Feature 3: Wikipedia frequency
    print("Loading word frequency (Wikipedia / fallback)...")
    wiki_freq = load_wiki_freq()
    word_df["wiki_freq"] = word_df["word"].str.lower().str.strip(".,!?;:'\"()").map(
        lambda w: wiki_freq.get(w, 0.0)
    )

    # Feature 4 & 5: Age of Acquisition
    print("Loading Age of Acquisition data...")
    aoa_dict = load_aoa()
    default_mean = np.mean([v[0] for v in aoa_dict.values()])
    default_std  = np.mean([v[1] for v in aoa_dict.values()])

    def get_aoa(word):
        if not isinstance(word, str):
            return (default_mean, default_std)
        w = word.lower().strip(".,!?;:'\"()")
        return aoa_dict.get(w, (default_mean, default_std))

    word_df["aoa_mean"] = word_df["word"].apply(lambda w: get_aoa(w)[0])
    word_df["aoa_std"]  = word_df["word"].apply(lambda w: get_aoa(w)[1])

    # Feature 6 & 7: BNC bigram probabilities (computed per text on ordered word sequence)
    print("Loading BNC bigram probabilities (cached or computing fresh)...")
    unigrams, fwd_counts, bwd_counts = build_bnc_bigrams()

    if fwd_counts is not None:
        print("  Computing per-word transition probabilities...")
        fwd_probs = {}
        bwd_probs = {}
        for tid, grp in word_df.groupby("text_id"):
            grp_sorted = grp.sort_values("wnum")
            words = grp_sorted["word"].tolist()
            probs = compute_transition_probs(words, fwd_counts, bwd_counts, unigrams)
            for (_, row), (fp, bp) in zip(grp_sorted.iterrows(), probs):
                fwd_probs[(int(tid), int(row["wnum"]))] = fp
                bwd_probs[(int(tid), int(row["wnum"]))] = bp

        word_df["fwd_prob"] = word_df.apply(
            lambda r: fwd_probs.get((int(r["text_id"]), int(r["wnum"])), math.log(1e-10)), axis=1)
        word_df["bwd_prob"] = word_df.apply(
            lambda r: bwd_probs.get((int(r["text_id"]), int(r["wnum"])), math.log(1e-10)), axis=1)
    else:
        print("  BNC bigrams not available. Setting transition probs to 0.")
        word_df["fwd_prob"] = 0.0
        word_df["bwd_prob"] = 0.0

    # Merge word-level features back to per-subject table
    feat_cols = ["text_id", "wnum", "sent_len", "wiki_freq", "aoa_mean", "aoa_std",
                 "fwd_prob", "bwd_prob"]
    rt_df = rt_df.merge(word_df[feat_cols], on=["text_id", "wnum"], how="left")

    # Select and save output (include subj_id for per-subject data)
    has_subj = "subj_id" in rt_df.columns
    id_cols = ["text_id", "wnum"] + (["subj_id"] if has_subj else []) + ["sent_id", "word"]
    out_cols = id_cols + ["wlen", "sent_len", "wiki_freq",
                          "aoa_mean", "aoa_std", "fwd_prob", "bwd_prob",
                          "ffd", "fpd", "rpd", "td"]

    out_df = rt_df[out_cols]
    out_path = os.path.join(OUT, "lexical_features.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved lexical features -> {out_path}")
    print(f"  Rows: {len(out_df)}")
    print("\nFeature summary:")
    for col in ["wlen", "sent_len", "wiki_freq", "aoa_mean", "aoa_std", "fwd_prob", "bwd_prob"]:
        print(f"  {col:12s}: mean={out_df[col].mean():.3f}, std={out_df[col].std():.3f}")


if __name__ == "__main__":
    main()

"""
Step 1: Parse Dundee eye-tracking corpus and compute 4 RT measures per word.

File structure discovered:
  - sa01ma2p.dat = subject 'a', article 01, type 2 (word-ordered fixations)
  - WNUM in RT files = global word index (same as c12 in per-word-info)
  - TEXT in RT files = page/screen number (not article number)
  - TEXT=-99 means word was skipped (not fixated)

Measures (paper Section 3.1):
  FFD - First Fixation Duration  : FDUR of the fixation with smallest FXNO
  FPD - First Pass Duration      : sum of FDUR during first continuous stay on word
  RPD - Regression Path Duration : sum of FDUR on word before eye first moves past it
  TD  - Total fixation Duration  : sum of all FDUR including re-fixations

Output: output/dundee_rt.csv  (one row per word token, averaged across subjects)
"""

import os, re
import pandas as pd

DUNDEE = "e:/Project/Final One/phase1_replication/Dataset/dundee_corpus"
OUT    = "e:/Project/Final One/phase1_replication/output"
os.makedirs(OUT, exist_ok=True)


def parse_rt2p_file(path):
    """
    Read one RT_dataset2P file.
    Returns DataFrame with columns: wnum, fdur, fxno
    Only keeps rows with fdur > 0 (fixated words).
    """
    rows = []
    try:
        with open(path, encoding="latin-1") as f:   
            for i, line in enumerate(f):
                if i == 0:
                    continue
                p = line.split()
                if len(p) < 11:
                    continue
                wnum = int(p[6])
                fdur = int(p[7])
                fxno = int(p[10])
                if wnum > 0 and fdur > 0 and fxno > 0:
                    rows.append((wnum, fdur, fxno))
    except Exception:
        return None
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["wnum", "fdur", "fxno"])


def compute_measures_one_article(df):
    """
    Given all fixations for one subject reading one article (from RT_dataset2P),
    compute FFD, FPD, RPD, TD for each word.

    df: DataFrame with columns [wnum, fdur, fxno], sorted by fxno ascending.

    Returns dict: wnum -> {ffd, fpd, rpd, td}
    """
    if df is None or len(df) == 0:
        return {}

    df = df.sort_values("fxno").reset_index(drop=True)

    # Build global timeline: list of (fxno, wnum, fdur) in chronological order
    timeline = list(df.itertuples(index=False))  # (wnum, fdur, fxno) → use .wnum .fdur .fxno

    # Group fixations per word
    word_fixations = {}   # wnum -> sorted list of (fxno, fdur)
    for row in timeline:
        wn = row.wnum
        word_fixations.setdefault(wn, []).append((row.fxno, row.fdur))

    result = {}
    for wn, fixes in word_fixations.items():
        fixes.sort()  # sort by fxno
        fxnos = [f[0] for f in fixes]
        fdurs = [f[1] for f in fixes]

        ffd = fdurs[0]   # first fixation duration
        td  = sum(fdurs)  # total fixation duration

        # FPD: consecutive fixations from the start (no other word between them)
        # Two fixations on the same word are "consecutive" if no other word appears
        # between their FXNO values in the global timeline.
        fpd = fdurs[0]
        for k in range(1, len(fxnos)):
            prev_fxno = fxnos[k - 1]
            curr_fxno = fxnos[k]
            # Check if any fixation on a DIFFERENT word happened between prev and curr
            other_between = df[(df["fxno"] > prev_fxno) &
                               (df["fxno"] < curr_fxno) &
                               (df["wnum"] != wn)]
            if len(other_between) == 0:
                fpd += fdurs[k]
            else:
                break  # left word → FPD window over

        # RPD: sum of fixations on this word that happened before the eye first
        # moved to a word with wnum > current word.
        first_fxno = fxnos[0]
        # Find first forward move: smallest fxno in timeline where wnum > wn AND fxno > first_fxno
        forward = df[(df["wnum"] > wn) & (df["fxno"] > first_fxno)]
        if len(forward) == 0:
            rpd = td  # never moved forward → all fixations count
        else:
            fxno_forward = forward["fxno"].min()
            rpd = sum(fd for fx, fd in fixes if fx < fxno_forward)

        result[wn] = {"ffd": ffd, "fpd": fpd, "rpd": rpd, "td": td}

    return result


def get_article_num(filename):
    """Extract article number from filename like 'sa01ma2p.dat' -> 1."""
    m = re.match(r"s[a-z](\d+)ma", filename)
    return int(m.group(1)) if m else None


def load_word_index():
    """
    Load canonical word list from per-word-info.
    Returns DataFrame: [text_id, c12, word, wlen, page, disp_line]
    """
    records = []
    info_dir = os.path.join(DUNDEE, "per-word-info")
    for fname in sorted(os.listdir(info_dir)):
        if not fname.endswith(".dat"):
            continue
        text_num = int(re.search(r"tx(\d+)", fname).group(1))
        with open(os.path.join(info_dir, fname), encoding="latin-1") as f:
            for line in f:
                p = line.split()
                if len(p) < 14:
                    continue
                word  = p[0]
                page  = int(p[2])
                dline = int(p[3])
                wlen  = int(p[7])
                c12   = int(p[12])   # global word index (= WNUM in RT files)
                records.append((text_num, c12, word, wlen, page, dline))
    return pd.DataFrame(records, columns=["text_id", "wnum", "word", "wlen", "page", "disp_line"])


def build_sentence_map():
    """
    Map (text_id, wnum) to sentence index using the dependency parse files.
    These files have WNUM and SentenceID columns that align with the RT corpus.
    Format: Word POS text_id WNUM SentenceID local_id CPOS Head DepRel
    Note: the Sentences/*.txt files use a different tokenization (punctuation
    as separate tokens) so they have a different word count than the RT files.
    """
    dep_dir = os.path.join(DUNDEE, "per-word-lexical-annotation")
    mapping = {}
    for fname in sorted(os.listdir(dep_dir)):
        if not fname.startswith("text") or not fname.endswith(".txt"):
            continue
        text_num = int(re.search(r"text(\d+)", fname).group(1))
        with open(os.path.join(dep_dir, fname), encoding="latin-1") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 5:
                    p = line.strip().split()
                if len(p) < 5:
                    continue
                try:
                    wnum    = int(p[3])
                    sent_id = int(p[4])
                    key = (text_num, wnum)
                    # First assignment wins (punctuation shares WNUM with prev word)
                    if key not in mapping:
                        mapping[key] = text_num * 10000 + sent_id
                except (ValueError, IndexError):
                    continue
    return mapping


def main():
    print("Loading canonical word list from per-word-info...")
    word_df = load_word_index()
    print(f"  Total word tokens: {len(word_df)}")

    print("Building sentence map...")
    sent_map = build_sentence_map()

    # Collect per-subject observations: {(text_id, wnum, subj_id): {ffd,fpd,rpd,td}}
    rt_per_subj = {}   # (text_id, wnum, subj_id) -> measure dict

    rt2p_dir = os.path.join(DUNDEE, "RT_dataset2P")
    files = sorted(os.listdir(rt2p_dir))
    print(f"Processing {len(files)} RT_dataset2P files...")

    subj_ids = {}  # filename prefix -> subject index 0-9
    for fname in files:
        if not fname.endswith(".dat"):
            continue
        art_num = get_article_num(fname)
        if art_num is None or art_num < 1 or art_num > 20:
            continue
        # Extract subject letter (s[a-j] prefix)
        subj_letter = fname[1]  # 'a'..'j' from sa01ma2p.dat -> 'a'
        if subj_letter not in subj_ids:
            subj_ids[subj_letter] = len(subj_ids)
        subj_idx = subj_ids[subj_letter]

        df = parse_rt2p_file(os.path.join(rt2p_dir, fname))
        measures = compute_measures_one_article(df)

        for wnum, m in measures.items():
            key = (art_num, wnum, subj_idx)
            rt_per_subj[key] = m

    print(f"  Per-subject fixation observations: {len(rt_per_subj)}")
    print(f"  Subjects found: {sorted(subj_ids.keys())}")

    # Build word metadata lookup
    word_meta = {}  # (text_id, wnum) -> {word, wlen, sent_id}
    for _, wr in word_df.iterrows():
        key = (int(wr["text_id"]), int(wr["wnum"]))
        sent_id = sent_map.get(key, -1)
        word_meta[key] = {
            "word": wr["word"],
            "wlen": int(wr["wlen"]),
            "sent_id": sent_id,
        }

    # Assemble final table: one row per (text_id, wnum, subj_id) observation
    # Paper: "RTs for all subjects were pooled into one set"
    # Only include fixated words (ffd > 0)
    print("Assembling per-subject RT table...")
    rows = []
    for (text_id, wnum, subj_idx), m in sorted(rt_per_subj.items()):
        if m["ffd"] <= 0:
            continue
        meta = word_meta.get((text_id, wnum), {})
        if not meta:
            continue
        rows.append({
            "text_id":  text_id,
            "wnum":     wnum,
            "subj_id":  subj_idx,
            "sent_id":  meta["sent_id"],
            "word":     meta["word"],
            "wlen":     meta["wlen"],
            "ffd":      m["ffd"],
            "fpd":      m["fpd"],
            "rpd":      m["rpd"],
            "td":       m["td"],
        })

    rt_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "dundee_rt.csv")
    rt_df.to_csv(out_path, index=False)

    fixated = rt_df[rt_df["ffd"] > 0]
    unique_words = rt_df.groupby(["text_id", "wnum"]).ngroups
    unique_sents = rt_df[rt_df["sent_id"] > 0]["sent_id"].nunique()
    print(f"\nSaved {len(rt_df)} per-subject observations -> {out_path}")
    print(f"  Unique words: {unique_words}, unique sentences: {unique_sents}")
    print(f"  Average observations per word: {len(rt_df)/max(unique_words,1):.1f}")
    print("\nMean RT values:")
    print(f"  FFD: {fixated['ffd'].mean():.1f} ms")
    print(f"  FPD: {fixated['fpd'].mean():.1f} ms")
    print(f"  RPD: {fixated['rpd'].mean():.1f} ms")
    print(f"  TD:  {fixated['td'].mean():.1f} ms")
    print("\nSample (first 5 rows):")
    print(fixated.head().to_string(index=False))


if __name__ == "__main__":
    main()

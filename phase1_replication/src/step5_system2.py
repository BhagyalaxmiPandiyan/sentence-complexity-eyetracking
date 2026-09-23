"""
Step 5: Train System 2 - Readability Assessment using predicted reading times.

Replicates paper Section 4:
  - Task: given two sentences, identify which is standard Wikipedia (harder) vs Simple Wikipedia
  - Features: predicted FFD (from System 1) at each word position in each sentence
  - Extended model adds sentence-level features (length, sum RT, surprisal sum, log parse prob)
  - Classifiers: LogisticRegression and LinearSVC (SVMrank approximation)
  - Evaluation: 10-fold cross-validation

Expected results (paper Table 5):
  Base RT + Pairwise Classification:     73.82%
  Extended RT + Pairwise Classification: 75.21%
"""

import os, re, pickle
import platform
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

if platform.system() == "Linux":
    OUT     = "/mnt/e/Project/Final One/output"
    DATASET = "/mnt/e/Project/Final One/Dataset"
else:
    OUT     = "e:/Project/Final One/output"
    DATASET = "e:/Project/Final One/Dataset"
os.makedirs(OUT, exist_ok=True)




# ── Load System 1 model ───────────────────────────────────────────────────────

def load_system1():
    """Load trained System 1 models and feature info."""
    model_path = os.path.join(OUT, "system1_models.pkl")
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    return models


# ── Predict FFD for a sentence ────────────────────────────────────────────────

def predict_ffd_for_sentence(sentence, model_info, lex_features_fn):
    """
    Given a sentence (string), predict FFD for each word using System 1.
    Returns list of predicted FFD values (one per word).
    """
    words = sentence.strip().split()
    if not words:
        return []

    # Build feature row for each word in the sentence
    rows = []
    for i, word in enumerate(words):
        row = lex_features_fn(word, words, i)
        rows.append(row)

    if not rows:
        return []

    feature_cols = model_info["features"]
    X = pd.DataFrame(rows)[feature_cols].fillna(0)
    X_scaled = model_info["scaler"].transform(X)
    preds = model_info["model"].predict(X_scaled)
    thr = model_info["threshold"]
    preds[preds < thr] = 0.0
    return list(preds)


# ── Build per-word features for novel sentences ───────────────────────────────

def build_sentence_feature_extractor(aoa_dict, wiki_freq_fn, syn_extractor=None):
    """
    Returns a function: (word, sentence_words, position) -> feature_dict.

    If syn_extractor (a SyntacticExtractor) is provided, all syntactic features
    (syn_surprisal, emb_depth, h1-h8, …) are computed via spaCy + PTB POS LM.
    Otherwise falls back to zeros (preserving the original behaviour).

    The extractor caches the last sentence so repeated calls for different
    word positions within the same sentence don't re-run spaCy.
    """
    # Precompute AoA defaults once (not per-word — would be O(n_aoa) per call)
    _default_aoa_mean = float(np.mean([v[0] for v in aoa_dict.values()])) if aoa_dict else 6.0
    _default_aoa_std  = float(np.mean([v[1] for v in aoa_dict.values()])) if aoa_dict else 1.0

    _cache = {"sent": None, "feats": None}

    def _get_sentence_feats(sentence_words):
        sentence = " ".join(sentence_words)
        if _cache["sent"] != sentence:
            if syn_extractor is not None:
                _cache["feats"] = syn_extractor.extract(sentence)
            else:
                _cache["feats"] = None
            _cache["sent"] = sentence
        return _cache["feats"]

    def get_features(word, sentence_words, pos):
        sent_feats = _get_sentence_feats(sentence_words)
        if sent_feats is not None and pos < len(sent_feats):
            return sent_feats[pos]

        # Fallback: lexical features only
        w = str(word).lower().strip(".,!?;:'\"()")
        wlen = len(re.sub(r"[^a-zA-Z]", "", word))
        aoa_mean, aoa_std = aoa_dict.get(w, (_default_aoa_mean, _default_aoa_std))
        return {
            "wlen":            wlen,
            "sent_len":        len(sentence_words),
            "wiki_freq":       wiki_freq_fn(w),
            "aoa_mean":        aoa_mean,
            "aoa_std":         aoa_std,
            "fwd_prob":        0.0,
            "bwd_prob":        0.0,
            "total_surprisal": 0.0,
            "lex_surprisal":   0.0,
            "syn_surprisal":   0.0,
            "entropy_red":     0.0,
            "emb_depth":       0,
            "emb_diff":        0,
            "h1": 0, "h2": 0, "h3": 0, "h4": 0,
            "h5": 0, "h6": 0, "h7": 0, "h8": 0,
        }

    return get_features


# ── Load Wikipedia / Simple Wikipedia pairs ───────────────────────────────────

def load_wiki_pairs(max_pairs=None):
    """
    Load sentence pairs from the aligned dataset.
    Format: simple_sentence<TAB>wiki_sentence<TAB>score
    Returns: list of (simple_sent, wiki_sent) tuples.
    """
    wiki_path = os.path.join(DATASET, "Wiki", "aligned-good(0.67).gz")
    pairs = []
    with open(wiki_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            # Col0 = Wikipedia (complex), Col1 = Simple Wikipedia (simpler)
            wiki   = parts[0].strip()
            simple = parts[1].strip()
            # Skip identical pairs (paper removes these)
            if simple == wiki or not simple or not wiki:
                continue
            pairs.append((simple, wiki))
            if max_pairs and len(pairs) >= max_pairs:
                break

    print(f"  Loaded {len(pairs):,} valid sentence pairs.")
    return pairs


# ── Build feature vectors for System 2 ───────────────────────────────────────

def build_pair_features(pairs, ffd_predictor, max_words=60):
    """
    Build feature matrix for sentence pairs.
    Feature: [sentence1_word1_ffd, ..., sentence1_wordN_ffd,
              sentence2_word1_ffd, ..., sentence2_wordN_ffd]
    Padded to max_words per sentence (paper: 2 × max_sentence_length).

    Returns: X (n_pairs, 2*max_words), y (n_pairs,)
    Label: 1 if sentence1=WIKI (harder), 0 if sentence1=SIMPLE
    """
    X_rows = []
    y_rows = []

    for i, (simple, wiki) in enumerate(pairs):
        if i % 10000 == 0:
            print(f"    Processed {i:,} / {len(pairs):,} pairs...")

        # Randomly assign which sentence goes first (to prevent ordering bias)
        # In the paper, the task is symmetric so we try both assignments
        # Convention: sentence1=WIKI (label=1), sentence2=SIMPLE (label=0)
        wiki_ffds   = ffd_predictor(wiki)[:max_words]
        simple_ffds = ffd_predictor(simple)[:max_words]

        # Pad to max_words
        wiki_ffds   = wiki_ffds   + [0.0] * (max_words - len(wiki_ffds))
        simple_ffds = simple_ffds + [0.0] * (max_words - len(simple_ffds))

        feat_vec = wiki_ffds + simple_ffds  # sentence1 (WIKI) then sentence2 (SIMPLE)
        X_rows.append(feat_vec)
        y_rows.append(1)   # sentence1 is WIKI (harder) → label 1

        # Add swapped version (sentence1=SIMPLE, label=0)
        feat_vec_swapped = simple_ffds + wiki_ffds
        X_rows.append(feat_vec_swapped)
        y_rows.append(0)   # sentence1 is SIMPLE (easier) → label 0

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def build_extended_features(pairs, X_base, ffd_predictor, lex_feats_fn):
    """
    Add sentence-level features to the base RT features (Extended RT model).
    Extra features per pair:
      - sent1_len, sent2_len
      - sent1_mean_ffd, sent2_mean_ffd  (normalized)
      - sent1_sum_surprisal, sent2_sum_surprisal
    """
    extra_rows = []

    for simple, wiki in pairs:
        wiki_words   = wiki.strip().split()
        simple_words = simple.strip().split()

        # Sentence lengths
        s1_len = len(wiki_words)
        s2_len = len(simple_words)

        # Predicted FFDs
        wiki_ffds   = ffd_predictor(wiki)
        simple_ffds = ffd_predictor(simple)

        s1_mean_ffd = np.mean(wiki_ffds)   if wiki_ffds   else 0.0
        s2_mean_ffd = np.mean(simple_ffds) if simple_ffds else 0.0

        # Sum of surprisal (use lex_feats_fn)
        def sent_surprisal(words):
            s = 0.0
            for i, w in enumerate(words):
                f = lex_feats_fn(w, words, i)
                s += f.get("lex_surprisal", 0.0)
            return s

        s1_surp = sent_surprisal(wiki_words)
        s2_surp = sent_surprisal(simple_words)

        extra_rows.append([s1_len, s2_len, s1_mean_ffd, s2_mean_ffd, s1_surp, s2_surp])
        # Also add swapped version
        extra_rows.append([s2_len, s1_len, s2_mean_ffd, s1_mean_ffd, s2_surp, s1_surp])

    extra = np.array(extra_rows, dtype=np.float32)
    return np.hstack([X_base, extra])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading System 1 models...")
    models = load_system1()
    ffd_info = models["ffd"]

    print("Loading support data (AoA, wiki freq)...")

    # Load AoA
    aoa_dict = {}
    try:
        aoa_df = pd.read_csv(os.path.join(DATASET, "AoA.csv"), encoding="latin-1")
        for _, row in aoa_df.iterrows():
            w = str(row["Word"]).lower().strip()
            v = row.get("AoA_Kup", float("nan"))
            if not pd.isna(v):
                vals = [row.get(c, float("nan")) for c in ["AoA_Kup", "AoA_Bird_lem", "AoA_Bristol_lem"]]
                vals = [x for x in vals if not pd.isna(x)]
                aoa_dict[w] = (float(v), float(np.std(vals)) if len(vals) > 1 else 0.0)
    except Exception as e:
        print(f"  AoA load warning: {e}")

    # Word frequency via wordfreq (same library as step2)
    import math
    try:
        from wordfreq import word_frequency
        def _wf(w):
            return math.log(word_frequency(w, "en") + 1e-9)
    except ImportError:
        def _wf(w):
            return 0.0

    # Use lexical features only for Wiki pair processing (spaCy on 235K sentences causes OOM)
    syn_ext = None

    lex_feats_fn = build_sentence_feature_extractor(aoa_dict, _wf, syn_extractor=syn_ext)

    def ffd_predictor(sentence):
        return predict_ffd_for_sentence(sentence, ffd_info, lex_feats_fn)

    print("\nLoading Wikipedia / Simple Wikipedia pairs...")
    pairs = load_wiki_pairs()   # all ~117K pairs

    print("\nBuilding Base RT feature matrix...")
    X_base, y = build_pair_features(pairs, ffd_predictor)
    print(f"  Feature matrix shape: {X_base.shape}")

    print("\nBuilding Extended RT feature matrix...")
    X_ext = build_extended_features(pairs, X_base, ffd_predictor, lex_feats_fn)
    print(f"  Extended feature matrix shape: {X_ext.shape}")

    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("\n--- System 2 Results (10-fold CV) ---")
    print(f"{'Model':50s}  {'Accuracy':>9}")

    for label, X in [("Base RT model", X_base), ("Extended RT model", X_ext)]:
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Logistic Regression (pairwise classification)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
        lr_acc = lr_scores.mean() * 100
        print(f"System 2 ({label}) - Pairwise Classification:  {lr_acc:.2f}%")

        # SVMrank approximation
        svm = LinearSVC(max_iter=2000, C=1.0, random_state=42)
        svm_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
        svm_acc = svm_scores.mean() * 100
        print(f"System 2 ({label}) - SVMrank:                   {svm_acc:.2f}%")

    print("\nPaper target:")
    print("  Base RT (Pairwise Classification):     73.82%")
    print("  Extended RT (Pairwise Classification): 75.21%")


if __name__ == "__main__":
    main()

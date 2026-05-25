"""
Step 4: Train System 1 - Linear Regression to predict per-word reading times.

Replicates paper Section 3.3 and 3.4:
  - Split at SENTENCE level: 60% train, 20% dev, 20% test
  - Standardize all features
  - Train LinearRegression (sklearn) for each of 4 RT measures
  - Ablation study: add features one by one, report R² (Table 2)
  - Threshold tuning on dev set: any prediction below T ms -> 0.0 (Table 3)
  - Final evaluation on test set

Expected results (paper Table 3):
  FFD: R² = 0.649 (threshold 84ms)
  FPD: R² = 0.600 (threshold 88ms)
  RPD: R² = 0.570 (threshold 91ms)
  TD:  R² = 0.516 (threshold 98ms)
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle

import platform
OUT = ("/mnt/e/Project/Final One/output" if platform.system() == "Linux"
       else "e:/Project/Final One/output")
os.makedirs(OUT, exist_ok=True)


# ── Feature columns (same order as paper's ablation study) ───────────────────

# Paper Table 2 adds features in this order:
FEATURE_GROUPS = [
    ("Word Length",             ["wlen"]),
    ("Sentence Length",         ["sent_len"]),
    ("Wikipedia Frequency",     ["wiki_freq"]),
    ("Mean AoA (in years)",     ["aoa_mean"]),
    ("Standard Dev. in AoA",    ["aoa_std"]),
    ("Fwd/Bwd Transition Prob", ["fwd_prob", "bwd_prob"]),
    ("Lexical Surprisal",       ["lex_surprisal"]),
    ("Syntactical Surprisal",   ["syn_surprisal"]),
    ("Entropy Reduction",       ["entropy_red"]),
    ("Embedding Depth",         ["emb_depth"]),
    ("Embedding Difference",    ["emb_diff"]),
    ("Hierarchical structure",  ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]),
]

ALL_FEATURES = [col for _, cols in FEATURE_GROUPS for col in cols]
RT_MEASURES  = ["ffd", "fpd", "rpd", "td"]


def load_data(normalize_per_subject=False):
    """Load and merge all feature files. Returns a single DataFrame.

    One row per word position; RT = mean across all subjects who fixated that word
    (subjects who skipped contribute 0, matching the paper's threshold mechanism).
    """
    lex = pd.read_csv(os.path.join(OUT, "lexical_features.csv"))
    syn = pd.read_csv(os.path.join(OUT, "syntactic_features.csv"))

    word_feat_cols = ["text_id", "wnum", "sent_id", "word", "wlen",
                      "wiki_freq", "aoa_mean", "aoa_std", "fwd_prob", "bwd_prob"]
    # Drop sent_len from the list — the CSV value is corrupted (counted subject rows,
    # not unique words). We recompute it below from deduplicated word positions.
    avail_cols = [c for c in word_feat_cols if c in lex.columns]
    word_df = lex[avail_cols].drop_duplicates(subset=["text_id", "wnum"])

    # Correct sent_len: number of unique word positions in each sentence
    correct_sent_len = (
        word_df.groupby("sent_id")["wnum"].count().rename("sent_len")
    )
    word_df = word_df.merge(correct_sent_len.reset_index(), on="sent_id", how="left")

    if "subj_id" in lex.columns:
        n_subj = lex["subj_id"].nunique()
        rt_sum = lex.groupby(["text_id", "wnum"])[RT_MEASURES].sum()
        rt_avg = (rt_sum / n_subj).reset_index()
        df = word_df.merge(rt_avg, on=["text_id", "wnum"], how="left")
        df[RT_MEASURES] = df[RT_MEASURES].fillna(0.0)
        n_fixated = (lex["ffd"] > 0).sum()
        n_total = len(df)
        print(f"  Word-level: {n_total:,} word positions, {n_fixated:,} fixated observations "
              f"averaged over {n_subj} subjects")
    else:
        df = lex.copy()

    df = df.merge(syn, on=["text_id", "wnum"], how="left")

    # Fill any NaN features with column mean
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = 0.0

    if normalize_per_subject and "subj_id" in df.columns:
        for rt in RT_MEASURES:
            grp = df.groupby("subj_id")[rt]
            df[rt] = (df[rt] - grp.transform("mean")) / grp.transform("std").clip(lower=1.0)
        print("  Applied per-subject Z-score normalization to RT values.")

    return df


def split_by_sentence(df, train_frac=0.6, dev_frac=0.2, seed=42):
    """
    Split dataset at SENTENCE level (not word level) as per paper.
    Returns (train_df, dev_df, test_df).
    """
    rng = np.random.default_rng(seed)
    sent_ids = df["sent_id"].unique()
    rng.shuffle(sent_ids)

    n = len(sent_ids)
    n_train = int(n * train_frac)
    n_dev   = int(n * dev_frac)

    train_sents = set(sent_ids[:n_train])
    dev_sents   = set(sent_ids[n_train:n_train + n_dev])
    test_sents  = set(sent_ids[n_train + n_dev:])

    train_df = df[df["sent_id"].isin(train_sents)].copy()
    dev_df   = df[df["sent_id"].isin(dev_sents)].copy()
    test_df  = df[df["sent_id"].isin(test_sents)].copy()

    print(f"  Sentences: train={len(train_sents)}, dev={len(dev_sents)}, test={len(test_sents)}")
    print(f"  Words:     train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")
    return train_df, dev_df, test_df


def apply_threshold(predictions, threshold):
    """Set predictions below threshold to 0.0 (as per paper)."""
    result = predictions.copy()
    result[result < threshold] = 0.0
    return result


def tune_threshold(model, scaler, X_dev, y_dev, feature_cols, rt_name):
    """
    Find best threshold on dev set.
    Paper uses thresholds: FFD=84ms, FPD=88ms, RPD=91ms, TD=98ms.
    We search from 50 to 200ms in 1ms steps.
    """
    X_dev_scaled = scaler.transform(X_dev[feature_cols])
    preds_raw = model.predict(X_dev_scaled)
    y_true = y_dev[rt_name].values

    best_r2  = r2_score(y_true, preds_raw)
    best_thr = 0

    for thr in range(50, 201, 1):
        preds = apply_threshold(preds_raw.copy(), thr)
        r2 = r2_score(y_true, preds)
        if r2 > best_r2:
            best_r2  = r2
            best_thr = thr

    return best_thr, best_r2


# ── Ablation study (Table 2) ─────────────────────────────────────────────────

def ablation_study(train_df, dev_df, rt_name="ffd"):
    """
    Add features one group at a time and report R² on dev set.
    Paper Table 2 uses FFD as the target.
    """
    print(f"\n--- Ablation Study ({rt_name.upper()}) ---")
    print(f"{'S.No':>5}  {'Features':40s}  {'R² score':>8}")

    # Use all words (non-fixated = RT 0) to match paper's threshold mechanism
    train = train_df.copy()
    dev   = dev_df.copy()

    cumulative_cols = []
    for num, (feat_name, cols) in enumerate(FEATURE_GROUPS, start=1):
        cumulative_cols.extend([c for c in cols if c in train.columns])
        if not cumulative_cols:
            continue

        X_train = train[cumulative_cols].fillna(0)
        y_train = train[rt_name].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)

        X_dev_s = scaler.transform(dev[cumulative_cols].fillna(0))
        y_dev = dev[rt_name].values
        preds = model.predict(X_dev_s)
        r2 = r2_score(y_dev, preds)

        print(f"{num:>5}  {feat_name:40s}  {r2:>8.3f}")

    return r2  # last R² (all features)


# ── Train final model for one RT measure ─────────────────────────────────────

def train_and_evaluate(train_df, dev_df, test_df, rt_name, feature_cols):
    """
    Train LinearRegression for one RT measure.
    1. Fit scaler + model on train (fixated words only).
    2. Tune threshold on dev.
    3. Evaluate on test with threshold.
    Returns: (model, scaler, threshold, train_r2, dev_r2, test_r2)
    """
    # Use all words (non-fixated = RT 0); threshold zeroes out predicted skips
    train = train_df.copy()
    dev   = dev_df.copy()
    test  = test_df.copy()

    X_train = train[feature_cols].fillna(0)
    y_train = train[rt_name].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Tune Ridge alpha on dev to handle collinear h1-h8 binary features
    best_alpha, best_dev = 1.0, -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha).fit(X_train_s, y_train)
        r2 = r2_score(dev_df[rt_name].values, m.predict(scaler.transform(dev_df[feature_cols].fillna(0))))
        if r2 > best_dev:
            best_dev, best_alpha = r2, alpha

    model = Ridge(alpha=best_alpha)
    model.fit(X_train_s, y_train)

    # Train R²
    train_r2 = r2_score(y_train, model.predict(X_train_s))

    # Dev: find best threshold
    best_thr, dev_r2 = tune_threshold(model, scaler, dev, dev, feature_cols, rt_name)

    # Test: apply threshold
    X_test_s = scaler.transform(test[feature_cols].fillna(0))
    preds_raw = model.predict(X_test_s)
    preds = apply_threshold(preds_raw, best_thr)
    test_r2 = r2_score(test[rt_name].values, preds)

    return model, scaler, best_thr, train_r2, dev_r2, test_r2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading all features...")
    df = load_data(normalize_per_subject=False)

    # Print available features
    avail = [c for c in ALL_FEATURES if c in df.columns and df[c].notna().any() and df[c].std() > 0]
    print(f"  Available features with variance: {avail}")

    print("\nSplitting at sentence level (60/20/20)...")
    train_df, dev_df, test_df = split_by_sentence(df)

    # Ablation study on FFD (Table 2)
    ablation_study(train_df, dev_df, rt_name="ffd")

    # Train final models for all 4 RT measures
    print("\n--- Final Model Results (Table 3) ---")
    print(f"{'Measure':>10}  {'Threshold':>10}  {'Train R²':>9}  {'Dev R²':>9}  {'Test R²':>9}")

    feature_cols = avail   # use all available non-zero features
    models = {}

    for rt_name in RT_MEASURES:
        model, scaler, thr, tr_r2, dev_r2, te_r2 = train_and_evaluate(
            train_df, dev_df, test_df, rt_name, feature_cols
        )
        models[rt_name] = {"model": model, "scaler": scaler, "threshold": thr,
                           "features": feature_cols}
        thr_str = f"{thr} ms" if thr > 0 else "none"
        print(f"{rt_name.upper():>10}  {thr_str:>10}  {tr_r2:>9.3f}  {dev_r2:>9.3f}  {te_r2:>9.3f}")

    # Save models for use in step 5
    model_path = os.path.join(OUT, "system1_models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved to {model_path}")

    # Save predictions for the full dataset (for System 2)
    print("\nGenerating predictions for full dataset (for System 2)...")
    ffd_model  = models["ffd"]["model"]
    ffd_scaler = models["ffd"]["scaler"]
    ffd_thr    = models["ffd"]["threshold"]

    X_all = df[feature_cols].fillna(0)
    X_all_s = ffd_scaler.transform(X_all)
    preds_raw = ffd_model.predict(X_all_s)
    df["predicted_ffd"] = apply_threshold(preds_raw, ffd_thr)

    pred_path = os.path.join(OUT, "system1_predictions.csv")
    # Collapse to word level (average predicted_ffd per word position; features are word-level)
    word_preds = df.groupby(["text_id", "wnum"])[["predicted_ffd", "ffd", "fpd", "rpd", "td"]].mean().reset_index()
    word_meta = df[["text_id", "wnum", "sent_id", "word"]].drop_duplicates(subset=["text_id", "wnum"])
    word_preds = word_meta.merge(word_preds, on=["text_id", "wnum"])
    word_preds.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path} ({len(word_preds)} word positions)")


if __name__ == "__main__":
    main()

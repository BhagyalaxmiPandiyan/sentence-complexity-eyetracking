"""
One-time script: fit the final System 2 (Extended RT) classifier on the
full 117,712 Wikipedia/Simple-Wikipedia pair dataset and save it, along
with the small supporting files the demo app needs, into this folder.

Run once locally (not part of the deployed app):
    python train_system2.py

Reuses the exact feature-extraction pipeline from
phase1_replication/src/step5_system2.py — same pipeline that reproduced
the paper's numbers (74.15% / 75.26% accuracy).
"""

import os
import pickle
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "phase1_replication", "src")
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import step5_system2 as s2


def main():
    print("Loading System 1 models...")
    models = s2.load_system1()
    ffd_info = models["ffd"]

    print("Loading AoA + word frequency...")
    aoa_dict = {}
    aoa_df = pd.read_csv(os.path.join(s2.DATASET, "AoA.csv"), encoding="latin-1")
    for _, row in aoa_df.iterrows():
        w = str(row["Word"]).lower().strip()
        v = row.get("AoA_Kup", float("nan"))
        if not pd.isna(v):
            vals = [row.get(c, float("nan")) for c in ["AoA_Kup", "AoA_Bird_lem", "AoA_Bristol_lem"]]
            vals = [x for x in vals if not pd.isna(x)]
            aoa_dict[w] = (float(v), float(np.std(vals)) if len(vals) > 1 else 0.0)

    import math
    from wordfreq import word_frequency

    def wiki_freq_fn(w):
        return math.log(word_frequency(w, "en") + 1e-9)

    lex_feats_fn = s2.build_sentence_feature_extractor(aoa_dict, wiki_freq_fn, syn_extractor=None)

    def ffd_predictor(sentence):
        return s2.predict_ffd_for_sentence(sentence, ffd_info, lex_feats_fn)

    print("Loading Wikipedia / Simple Wikipedia pairs...")
    pairs = s2.load_wiki_pairs()

    print("Building Base RT feature matrix...")
    X_base, y = s2.build_pair_features(pairs, ffd_predictor)

    print("Building Extended RT feature matrix...")
    X_ext = s2.build_extended_features(pairs, X_base, ffd_predictor, lex_feats_fn)

    print("Fitting final classifier on full data (Extended RT + Logistic Regression)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ext)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_scaled, y)
    train_acc = clf.score(X_scaled, y) * 100
    print(f"  Training accuracy (full fit, not CV): {train_acc:.2f}%")
    print("  (10-fold CV accuracy for this model is 75.26%, see phase1_replication/results/summary.md)")

    out_path = os.path.join(HERE, "models", "system2_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"scaler": scaler, "model": clf, "max_words": 60}, f)
    print(f"Saved {out_path}")

    print("Copying supporting files into toolkit/...")
    shutil.copy(
        os.path.join(s2.OUT, "system1_models.pkl"),
        os.path.join(HERE, "models", "system1_models.pkl"),
    )
    shutil.copy(
        os.path.join(s2.DATASET, "AoA.csv"),
        os.path.join(HERE, "data", "AoA.csv"),
    )
    print("Done. toolkit/ is now self-contained — run `python app.py` to launch the demo.")


if __name__ == "__main__":
    main()

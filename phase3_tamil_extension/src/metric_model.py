"""
The two difficulty metrics compared in Phase B/C, as one number per sentence.

Each metric is the pair model from step 6 (logistic regression on
features(b) - features(a), no intercept), trained on the StoryWeaver train
pairs (the cleaner of the two graded datasets). With no intercept the model is
linear, so it gives every sentence a score:

    score(sentence) = sum_i  w_i * feature_i / scale_i

and score(b) - score(a) is the model's log-odds that b is harder. Higher
score = harder. Scores only mean something relative to each other.

    word   : group A  (n_words, word_len, word_zipf, word_surprisal) -- the naive,
             English-style metric
    morph  : groups A + B  (adds morphs_per_word, stem/suffix surprisal,
             feats_per_word, lemma_zipf, nonfinite_verbs) -- the morphology-aware metric
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = os.path.join(BASE, "output", "storyweaver_features.csv")
PAIRS = os.path.join(BASE, "output", "storyweaver_pairs.csv")

A = ["n_words", "word_len", "word_zipf", "word_surprisal"]
B = ["morphs_per_word", "stem_surprisal", "suffix_surprisal",
     "feats_per_word", "lemma_zipf", "nonfinite_verbs"]
METRICS = {"word": A, "morph": A + B}


def fit_metrics():
    """Returns {metric name: (feature list, weights already divided by scale)}."""
    f = pd.read_csv(FEATS).drop_duplicates("sentence").set_index("sentence")
    p = pd.read_csv(PAIRS)
    p = p[p.split == "train"]
    y = (p.harder == "b").astype(int).to_numpy()
    out = {}
    for name, cols in METRICS.items():
        D = f.loc[p.sent_b, cols].to_numpy() - f.loc[p.sent_a, cols].to_numpy()
        scale = D.std(0) + 1e-9
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000).fit(D / scale, y)
        out[name] = (cols, clf.coef_[0] / scale)
    return out


def score(features, metrics):
    """features: DataFrame with the feature columns. Adds one 'score_<metric>' column each."""
    out = pd.DataFrame(index=features.index)
    for name, (cols, w) in metrics.items():
        out[f"score_{name}"] = features[cols].to_numpy() @ w
    return out


if __name__ == "__main__":
    for name, (cols, w) in fit_metrics().items():
        print(name, dict(zip(cols, np.round(w, 3))))

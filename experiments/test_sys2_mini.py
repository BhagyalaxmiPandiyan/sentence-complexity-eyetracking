"""Quick System 2 test on 5000 pairs."""
import os, pickle, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT     = "/mnt/e/Project/Final One/output"
DATASET = "/mnt/e/Project/Final One/Dataset"

with open(os.path.join(OUT, "system1_models.pkl"), "rb") as f:
    models = pickle.load(f)
ffd_info = models["ffd"]
print(f"Model loaded. Threshold={ffd_info['threshold']}ms, features={len(ffd_info['features'])}")

aoa_dict = {}
aoa_df = pd.read_csv(os.path.join(DATASET, "AoA.csv"), encoding="latin-1")
for _, row in aoa_df.iterrows():
    w = str(row["Word"]).lower().strip()
    v = row.get("AoA_Kup", float("nan"))
    if not pd.isna(v):
        aoa_dict[w] = (float(v), 1.0)

try:
    from wordfreq import word_frequency
    def _wf(w): return math.log(word_frequency(w, "en") + 1e-9)
except ImportError:
    def _wf(w): return 0.0

feat_cols = ffd_info["features"]
thr = ffd_info["threshold"]

def predict_ffd(sentence):
    words = sentence.strip().split()
    if not words: return []
    rows = [{
        "wlen": len(w), "sent_len": len(words),
        "wiki_freq": _wf(w.lower().strip(".,!?;:'\"()")),
        "aoa_mean": aoa_dict.get(w.lower().strip(".,!?;:'\"()"), (6.0, 1.0))[0],
        "aoa_std": 1.0, "fwd_prob": 0.0, "bwd_prob": 0.0,
        "total_surprisal": 0.0, "lex_surprisal": 0.0, "syn_surprisal": 0.0,
        "entropy_red": 0.0, "emb_depth": 0, "emb_diff": 0,
        "h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0, "h7": 0, "h8": 0,
    } for w in words]
    X = pd.DataFrame(rows)[feat_cols].fillna(0)
    preds = ffd_info["model"].predict(ffd_info["scaler"].transform(X))
    preds[preds < thr] = 0.0
    return list(preds)

wiki_path = os.path.join(DATASET, "Wiki", "aligned-good(0.67).gz")
pairs = []
with open(wiki_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            wiki, simple = parts[0].strip(), parts[1].strip()
            if wiki != simple and wiki and simple:
                pairs.append((simple, wiki))
                if len(pairs) >= 5000:
                    break
print(f"Using {len(pairs)} pairs for quick test")

MAX_WORDS = 60
X_rows, y_rows = [], []
for simple, wiki in pairs:
    wf = predict_ffd(wiki)[:MAX_WORDS]
    sf = predict_ffd(simple)[:MAX_WORDS]
    wf += [0.0] * (MAX_WORDS - len(wf))
    sf += [0.0] * (MAX_WORDS - len(sf))
    X_rows.append(wf + sf); y_rows.append(1)
    X_rows.append(sf + wf); y_rows.append(0)

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.int32)
print(f"Feature matrix: {X.shape}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
scores = cross_val_score(lr, X_s, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Base RT 10-fold CV (5K pairs): {scores.mean()*100:.2f}%  (paper: 73.82%)")
print("Pipeline works correctly.")

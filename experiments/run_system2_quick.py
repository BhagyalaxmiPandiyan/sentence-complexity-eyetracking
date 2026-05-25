"""Run System 2 pipeline inline (bypasses step5 import issues)."""
import os, pickle, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT     = "/mnt/e/Project/Final One/output"
DATASET = "/mnt/e/Project/Final One/Dataset"

with open(os.path.join(OUT, "system1_models.pkl"), "rb") as f:
    models = pickle.load(f)
ffd_info = models["ffd"]
print("System 1 loaded. Features:", len(ffd_info["features"]), "Threshold:", ffd_info["threshold"])

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
    print(f"AoA words: {len(aoa_dict)}")
except Exception as e:
    print(f"AoA warning: {e}")

try:
    from wordfreq import word_frequency
    def _wf(w):
        return math.log(word_frequency(w, "en") + 1e-9)
except ImportError:
    def _wf(w):
        return 0.0

DEFAULT_AOA = (6.0, 1.0)

def get_features(word, sentence_words, pos):
    w = word.lower().strip(".,!?;:'\"()")
    wlen = len(w)
    aoa_m, aoa_s = aoa_dict.get(w, DEFAULT_AOA)
    return {
        "wlen": wlen, "sent_len": len(sentence_words),
        "wiki_freq": _wf(w), "aoa_mean": aoa_m, "aoa_std": aoa_s,
        "fwd_prob": 0.0, "bwd_prob": 0.0, "total_surprisal": 0.0,
        "lex_surprisal": 0.0, "syn_surprisal": 0.0, "entropy_red": 0.0,
        "emb_depth": 0, "emb_diff": 0,
        "h1": 0, "h2": 0, "h3": 0, "h4": 0,
        "h5": 0, "h6": 0, "h7": 0, "h8": 0,
    }

feat_cols = ffd_info["features"]
thr = ffd_info["threshold"]

def predict_ffd(sentence):
    words = sentence.strip().split()
    if not words:
        return []
    rows = [get_features(w, words, i) for i, w in enumerate(words)]
    X = pd.DataFrame(rows)[feat_cols].fillna(0)
    preds = ffd_info["model"].predict(ffd_info["scaler"].transform(X))
    preds[preds < thr] = 0.0
    return list(preds)

# Load pairs
wiki_path = os.path.join(DATASET, "Wiki", "aligned-good(0.67).gz")
pairs = []
with open(wiki_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            wiki, simple = parts[0].strip(), parts[1].strip()
            if wiki != simple and wiki and simple:
                pairs.append((simple, wiki))
print(f"Pairs: {len(pairs):,}")

MAX_WORDS = 60
print("Building base feature matrix...")
X_rows, y_rows = [], []
for i, (simple, wiki) in enumerate(pairs):
    if i % 25000 == 0:
        print(f"  {i:,} / {len(pairs):,}")
    wf = predict_ffd(wiki)[:MAX_WORDS]
    sf = predict_ffd(simple)[:MAX_WORDS]
    wf += [0.0] * (MAX_WORDS - len(wf))
    sf += [0.0] * (MAX_WORDS - len(sf))
    X_rows.append(wf + sf)
    y_rows.append(1)
    X_rows.append(sf + wf)
    y_rows.append(0)

X_base = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.int32)
print(f"Base matrix: {X_base.shape}")

# Extended features
print("Building extended feature matrix...")
extra_rows = []
for simple, wiki in pairs:
    ww = wiki.strip().split()
    sw = simple.strip().split()
    wf = predict_ffd(wiki)
    sf = predict_ffd(simple)
    s1_mean = float(np.mean(wf)) if wf else 0.0
    s2_mean = float(np.mean(sf)) if sf else 0.0
    s1_surp = sum(get_features(w, ww, i).get("lex_surprisal", 0.0) for i, w in enumerate(ww))
    s2_surp = sum(get_features(w, sw, i).get("lex_surprisal", 0.0) for i, w in enumerate(sw))
    extra_rows.append([len(ww), len(sw), s1_mean, s2_mean, s1_surp, s2_surp])
    extra_rows.append([len(sw), len(ww), s2_mean, s1_mean, s2_surp, s1_surp])

X_ext = np.hstack([X_base, np.array(extra_rows, dtype=np.float32)])
print(f"Extended matrix: {X_ext.shape}")

# 10-fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("\n--- System 2 Results (10-fold CV) ---")
print(f"{'Model':55s}  Accuracy")

for label, X in [("Base RT", X_base), ("Extended RT", X_ext)]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_scores = cross_val_score(lr, X_s, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {label} + Pairwise Classification (LR):   {lr_scores.mean()*100:.2f}%")

    svm = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    svm_scores = cross_val_score(svm, X_s, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {label} + SVMrank:                        {svm_scores.mean()*100:.2f}%")

print()
print("Paper targets:")
print("  Base RT (Pairwise):     73.82%")
print("  Extended RT (Pairwise): 75.21%")

"""
One-time: trains the Phase 2 models used by the toolkit's "English · Phase 2" tab.

System 1 (word reading time) for all 4 Dundee measures, with the features the
app can compute live for any typed sentence:
    baseline : wlen, sent_len, wiki_freq, aoa_mean, aoa_std
    + GPT-2  : baseline + gpt2_surprisal (GPT-2 small, sentence only, as in
               phase1_replication/experiments/compute_gpt2_surprisal.py)

Same data, split (60/20/20 by sentence) and training code as the thesis
(phase1_replication/src/step4_system1.py: Ridge + skip threshold), imported
unchanged. Saves models/phase2_models.pkl with the test R² of each model.

Run: C:\\Python313\\python.exe train_phase2.py   (~1 min)
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.join(os.path.dirname(HERE), "phase1_replication")
sys.path.insert(0, os.path.join(PHASE1, "src"))
os.chdir(os.path.join(PHASE1, "src"))
import step4_system1 as s1  # noqa: E402
import pandas as pd  # noqa: E402

SURPRISAL = os.path.join(PHASE1, "output", "gpt2_surprisal_gpt2.csv")
BASE = ["wlen", "sent_len", "wiki_freq", "aoa_mean", "aoa_std"]
VARIANTS = {"baseline": BASE, "gpt2": BASE + ["gpt2_surprisal"]}


def main():
    df = s1.load_data(normalize_per_subject=False)
    g = pd.read_csv(SURPRISAL)
    df = df.merge(g, on=["text_id", "wnum"], how="left")
    df["gpt2_surprisal"] = df["gpt2_surprisal"].fillna(df["gpt2_surprisal"].mean())
    train, dev, test = s1.split_by_sentence(df)

    out = {}
    print(f"{'measure':>8} {'baseline R2':>12} {'+GPT-2 R2':>10}")
    for rt in s1.RT_MEASURES:
        out[rt] = {}
        for name, cols in VARIANTS.items():
            model, scaler, thr, _, _, te = s1.train_and_evaluate(train, dev, test, rt, cols)
            out[rt][name] = {"model": model, "scaler": scaler, "threshold": thr,
                             "features": cols, "test_r2": te}
        print(f"{rt.upper():>8} {out[rt]['baseline']['test_r2']:>12.3f} {out[rt]['gpt2']['test_r2']:>10.3f}")

    path = os.path.join(HERE, "models", "phase2_models.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print("saved", path)


if __name__ == "__main__":
    main()

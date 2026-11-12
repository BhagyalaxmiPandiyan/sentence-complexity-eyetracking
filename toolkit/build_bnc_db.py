"""
One-time: turn the BNC bigram counts from Phase 1 step 2 into a small indexed SQLite file,
so the toolkit can look up any word pair instantly without loading the 220 MB CSV.

Input : ../phase1_replication/output/bnc_bigrams_full.csv  (w1, w2, count)
        ../phase1_replication/output/bnc_bigrams.csv       (word, count)  unigram counts
Output: data/bnc.sqlite   (tables bigram(w1, w2, c) and unigram(w, c))

Run: C:\\Python313\\python.exe build_bnc_db.py     (~2 minutes; the .sqlite file is not in git)
"""
import os
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "phase1_replication", "output")
DB = os.path.join(HERE, "data", "bnc.sqlite")


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE unigram (w TEXT PRIMARY KEY, c INTEGER)")
    con.execute("CREATE TABLE bigram (w1 TEXT, w2 TEXT, c INTEGER)")
    # Same parsing as step 2 / step 3 (unigrams default dtypes, bigrams as strings)
    uni = pd.read_csv(os.path.join(SRC, "bnc_bigrams.csv"))
    con.executemany("INSERT OR REPLACE INTO unigram VALUES (?, ?)",
                    ((str(w), int(c)) for w, c in zip(uni["word"], uni["count"])))
    n = 0
    for chunk in pd.read_csv(os.path.join(SRC, "bnc_bigrams_full.csv"),
                             dtype={"w1": str, "w2": str, "count": int}, chunksize=500_000):
        con.executemany("INSERT INTO bigram VALUES (?, ?, ?)",
                        zip(chunk["w1"].astype(str), chunk["w2"].astype(str), chunk["count"].astype(int)))
        n += len(chunk)
        print(f"  {n:,} bigrams", flush=True)
    print("indexing ...", flush=True)
    con.execute("CREATE INDEX ix_bigram ON bigram (w1, w2)")
    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"wrote {DB}  ({os.path.getsize(DB) / 1e6:.0f} MB, {len(uni):,} unigrams, {n:,} bigrams)")


if __name__ == "__main__":
    main()

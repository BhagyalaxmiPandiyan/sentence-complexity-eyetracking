"""
Live left-corner syntactic surprisal (the paper's method), for the toolkit's Phase 1 tab.

Runs INSIDE WSL (Ubuntu), started once by toolkit/phase1_full.py:
    wsl -d Ubuntu -e python3 "/mnt/e/Project/Final One/toolkit/lc_server.py"

Same pipeline as phase1_replication/experiments/build_lcparse_surprisal.py, one sentence
at a time: Berkeley parser (kept running, so the 22 MB grammar loads only once) ->
strip function tags + right-binarize (same functions) -> linetrees2synprocdecpars ->
per-word surprisal from the CPT model trained on PTB WSJ 02-21 (output/lcparse_model.cpt).

Protocol: one sentence per line on stdin (words separated by spaces); one JSON line on
stdout: {"tree": "...", "surprisal": [s1, s2, ...]} or {"error": "..."}.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/e/Project/Final One/phase1_replication/experiments")
os.environ["_IN_WSL"] = "1"                      # stop the module from re-launching itself
import build_lcparse_surprisal as lc            # noqa: E402
from nltk import Tree                             # noqa: E402

model = lc.load_model(lc.MODEL_FILE)
berk = subprocess.Popen(["java", "-jar", lc.BERK_JAR, "-gr", lc.BERK_GR, "-maxLength", "200"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        text=True, bufsize=1)
print(json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    try:
        berk.stdin.write(" ".join(words) + "\n")
        berk.stdin.flush()
        raw = berk.stdout.readline().strip()
        if not raw or raw == "()" or raw == "(())":
            raise ValueError("no parse")
        t = Tree.fromstring(raw)
        if t.label() == "" and len(t) == 1:      # Berkeley wraps the tree in "( ... )"
            t = t[0]
        lc.normalize_labels(t)
        lc.binarize(t)
        proc = subprocess.run(lc.BINARY, input=lc.tree_to_oneline(t) + "\n",
                              capture_output=True, text=True, timeout=5)
        surp = lc._parse_decisions(proc.stdout, list(range(len(words))), model)
        print(json.dumps({"tree": raw, "surprisal": [surp.get(i, 0.0) for i in range(len(words))]}),
              flush=True)
    except Exception as e:                        # one bad sentence must not stop the server
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)

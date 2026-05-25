"""
Full pipeline to build the modelblocks left-corner parser model and run it on Dundee sentences.

Pipeline:
  1. PTB trees → build_pcfg_model.py → output/wsj02to21.model (Cr/CC/Cu/X format)
  2. wsj02to21.model → pcfg2pxmodel.py → intermediate (CC^L,d format)
  3. intermediate → calc-fabp-model → F/A/Wa/Wb/P/X format
  4. F/A/Wa/Wb/P/X model + dundee_sentences.txt → parser-x-efabp → surprisal
"""

import os
import sys
import subprocess
import json

MBDIR = "/mnt/e/Mtech Project/Updated Folder/modelblocks-release"
LCPARSE = f"{MBDIR}/resource-lcparse"
GCG_SCRIPTS = f"{MBDIR}/resource-gcg/scripts"
OUT = "/mnt/e/Project/Final One/output"
PARSER_BIN = "/mnt/e/Project/Final One/parser-x-efabp"

PCFG_MODEL = f"{OUT}/wsj02to21.model"              # Cr/CC/Cu/X format
PX_MODEL   = f"{OUT}/wsj02to21.px.model"           # after pcfg2pxmodel.py
FABP_MODEL = f"{OUT}/wsj02to21.fabp.model"         # after calc-fabp-model
SENT_FILE  = f"{OUT}/dundee_sentences.txt"
MAP_FILE   = f"{OUT}/dundee_sent_map.json"
OUT_CSV    = f"{OUT}/dundee_surprisal.csv"


def compile_calc_fabp():
    bin_path = "/mnt/e/Project/Final One/calc-fabp-model"
    if os.path.exists(bin_path):
        print(f"calc-fabp-model already compiled at {bin_path}")
        return bin_path

    RVTL = f"/tmp/rvtl-patched"
    INCLUDE = f"{LCPARSE}/include"
    SRC = f"{LCPARSE}/src/calc-fabp-model.cpp"

    # Need to patch rvtl first
    import shutil
    os.makedirs(RVTL, exist_ok=True)
    for fname in os.listdir(f"{MBDIR}/resource-rvtl"):
        src = os.path.join(f"{MBDIR}/resource-rvtl", fname)
        dst = os.path.join(RVTL, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    # Patch nl-randvar.h: uncomment JointArrayRV block
    with open(f"{RVTL}/nl-randvar.h", "r") as f:
        lines = f.readlines()
    in_comment = False
    start_line = end_line = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "/*" and not in_comment and i > 500:
            in_comment = True
            start_line = i
        elif stripped == "*/" and in_comment:
            end_line = i
            in_comment = False
            break
    if start_line >= 0 and end_line >= 0:
        lines[start_line] = "// (JointArrayRV uncommented)\n"
        lines[end_line]   = "// (end JointArrayRV)\n"
    with open(f"{RVTL}/nl-randvar.h", "w") as f:
        f.writelines(lines)

    cmd = ["g++", f"-I{INCLUDE}", f"-I{RVTL}", "-Wall", "-fpermissive",
           "-std=c++11", "-lm", "-Wno-deprecated", SRC, "-o", bin_path]
    print("Compiling calc-fabp-model...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Compilation failed:", r.stderr[:500])
        sys.exit(1)
    print(f"Compiled: {bin_path}")
    return bin_path


def run_pcfg2px():
    """Run pcfg2pxmodel.py to convert PCFG to parser-compatible format."""
    print("Running pcfg2pxmodel.py...")
    pcfg2px = f"{LCPARSE}/scripts/pcfg2pxmodel.py"

    with open(PCFG_MODEL, "r") as f:
        pcfg_data = f.read()

    # pcfg2pxmodel.py needs sys.path with resource-gcg
    env = dict(os.environ)
    env["PYTHONPATH"] = GCG_SCRIPTS

    r = subprocess.run(
        ["python3", pcfg2px],
        input=pcfg_data,
        capture_output=True,
        text=True,
        env=env,
        timeout=300
    )

    if r.returncode != 0:
        print(f"pcfg2pxmodel.py failed (rc={r.returncode})")
        print("stderr:", r.stderr[:500])
        sys.exit(1)

    with open(PX_MODEL, "w") as f:
        f.write(r.stdout)

    print(f"  Written {len(r.stdout.splitlines())} lines to {PX_MODEL}")
    if r.stderr:
        print("  stderr:", r.stderr[:300])


def run_calc_fabp(calc_fabp_bin):
    """Run calc-fabp-model to compute F/A/Wa/Wb/P/X models."""
    print("Running calc-fabp-model...")

    with open(PX_MODEL, "r") as f:
        px_data = f.read()

    r = subprocess.run(
        [calc_fabp_bin],
        input=px_data,
        capture_output=True,
        text=True,
        timeout=300
    )

    if r.returncode != 0:
        print(f"calc-fabp-model failed (rc={r.returncode})")
        print("stderr:", r.stderr[:500])
        sys.exit(1)

    with open(FABP_MODEL, "w") as f:
        f.write(r.stdout)

    print(f"  Written {len(r.stdout.splitlines())} lines to {FABP_MODEL}")
    if r.stderr:
        print("  stderr:", r.stderr[:300])


def run_parser_on_dundee():
    """Run parser-x-efabp on all Dundee sentences."""
    print(f"Running parser on Dundee sentences...")

    with open(MAP_FILE) as f:
        sent_map = json.load(f)
    with open(SENT_FILE) as f:
        sentences = f.readlines()

    print(f"  {len(sentences)} sentences, beam=5000")

    BEAM = 5000
    batch_size = 50
    all_rows = []

    for batch_start in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_start:batch_start+batch_size]
        batch_map   = sent_map[batch_start:batch_start+batch_size]
        batch_input = "".join(batch_sents)

        r = subprocess.run(
            [PARSER_BIN, f"-b{BEAM}", FABP_MODEL],
            input=batch_input,
            capture_output=True,
            text=True,
            timeout=600
        )

        # Parse output
        word_rows = []
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if not line or "word" in line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    word_rows.append({
                        "word": parts[0],
                        "totsurp": float(parts[1]),
                        "lexsurp": float(parts[2]),
                        "synsurp": float(parts[3]),
                    })
                except ValueError:
                    pass

        # Map word rows to (text_id, sent_id, wnum)
        wi = 0
        for sm in batch_map:
            for wnum in sm["wnums"]:
                if wi < len(word_rows):
                    wr = word_rows[wi]
                    all_rows.append({
                        "text_id": sm["text_id"],
                        "sent_id": sm["sent_id"],
                        "wnum": wnum,
                        "totsurp": wr["totsurp"],
                        "lexsurp": wr["lexsurp"],
                        "synsurp": wr["synsurp"],
                    })
                    wi += 1
                else:
                    all_rows.append({
                        "text_id": sm["text_id"], "sent_id": sm["sent_id"],
                        "wnum": wnum,
                        "totsurp": float("nan"), "lexsurp": float("nan"), "synsurp": float("nan"),
                    })

        if batch_start % 500 == 0:
            print(f"  {min(batch_start+batch_size, len(sentences))}/{len(sentences)} sentences...")

    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id","sent_id","wnum","totsurp","lexsurp","synsurp"])
        writer.writeheader()
        writer.writerows(all_rows)

    valid = sum(1 for r in all_rows if r["totsurp"] == r["totsurp"])
    print(f"\nDone. {len(all_rows)} rows, {valid} with valid surprisal → {OUT_CSV}")


if __name__ == "__main__":
    # Step 1: Compile calc-fabp-model
    calc_fabp_bin = compile_calc_fabp()

    # Step 2: Convert PCFG to px format
    run_pcfg2px()

    # Step 3: Compute F/A/Wa/Wb/P/X models
    run_calc_fabp(calc_fabp_bin)

    # Step 4: Run parser on Dundee
    run_parser_on_dundee()

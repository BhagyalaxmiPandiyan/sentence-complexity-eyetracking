"""
Run the modelblocks left-corner parser on Dundee sentences.
Reads: output/dundee_sentences.txt + output/wsj02to21.model
Writes: output/dundee_surprisal.csv  (text_id, sent_id, wnum, word, totsurp, lexsurp, synsurp)
"""
import os
import sys
import subprocess
import json
import csv

OUT = "/mnt/e/Project/Final One/output"
PARSER = "/mnt/e/Project/Final One/parser-x-efabp"
MODEL = os.path.join(OUT, "wsj02to21.model")
SENT_FILE = os.path.join(OUT, "dundee_sentences.txt")
MAP_FILE  = os.path.join(OUT, "dundee_sent_map.json")
OUT_CSV   = os.path.join(OUT, "dundee_surprisal.csv")

BEAM = 5000  # beam width (paper uses b=5000)

def test_parser():
    """Quick test with first 3 sentences."""
    print("Testing parser with 3 sentences...")
    with open(SENT_FILE) as f:
        test_input = "\n".join([next(f).strip() for _ in range(3)])

    result = subprocess.run(
        [PARSER, f"-b{BEAM}", MODEL],
        input=test_input,
        capture_output=True,
        text=True,
        timeout=60
    )
    print("STDOUT:", result.stdout[:500])
    print("STDERR:", result.stderr[:300])
    print("RC:", result.returncode)
    return result.returncode == 0

def run_full_parser():
    print("Loading sentence map...")
    with open(MAP_FILE) as f:
        sent_map = json.load(f)

    print(f"Running parser on {len(sent_map)} sentences with beam={BEAM}...")
    print("(This may take 30-60 minutes for all 2368 sentences)")

    with open(SENT_FILE) as f:
        sentences = f.readlines()

    rows = []
    batch_size = 100

    for batch_start in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_start:batch_start+batch_size]
        batch_map   = sent_map[batch_start:batch_start+batch_size]

        batch_input = "".join(batch_sents)

        result = subprocess.run(
            [PARSER, f"-b{BEAM}", MODEL],
            input=batch_input,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"  WARNING: parser failed for batch {batch_start}-{batch_start+batch_size}")
            print(f"  stderr: {result.stderr[:200]}")
            # Add NaN rows for this batch
            for sm in batch_map:
                for wnum in sm["wnums"]:
                    rows.append({
                        "text_id": sm["text_id"],
                        "sent_id": sm["sent_id"],
                        "wnum": wnum,
                        "totsurp": float("nan"),
                        "lexsurp": float("nan"),
                        "synsurp": float("nan"),
                    })
            continue

        # Parse output: header line then word-by-word lines
        output_lines = result.stdout.strip().split("\n")

        # Find header
        header_idx = 0
        for i, line in enumerate(output_lines):
            if "word" in line and "totsurp" in line:
                header_idx = i
                break

        # Collect word lines (skip blanks and header)
        word_lines = []
        for line in output_lines[header_idx+1:]:
            line = line.strip()
            if line and not line.startswith("word"):
                parts = line.split()
                if len(parts) >= 4:
                    word_lines.append(parts)

        # Map to sentences
        word_idx = 0
        for sm in batch_map:
            wnums = sm["wnums"]
            for wnum in wnums:
                if word_idx < len(word_lines):
                    parts = word_lines[word_idx]
                    try:
                        rows.append({
                            "text_id": sm["text_id"],
                            "sent_id": sm["sent_id"],
                            "wnum": wnum,
                            "totsurp": float(parts[1]) if len(parts) > 1 else float("nan"),
                            "lexsurp": float(parts[2]) if len(parts) > 2 else float("nan"),
                            "synsurp": float(parts[3]) if len(parts) > 3 else float("nan"),
                        })
                    except (ValueError, IndexError):
                        rows.append({"text_id": sm["text_id"], "sent_id": sm["sent_id"],
                                     "wnum": wnum, "totsurp": float("nan"),
                                     "lexsurp": float("nan"), "synsurp": float("nan")})
                    word_idx += 1
                else:
                    rows.append({"text_id": sm["text_id"], "sent_id": sm["sent_id"],
                                 "wnum": wnum, "totsurp": float("nan"),
                                 "lexsurp": float("nan"), "synsurp": float("nan")})

        if (batch_start // batch_size) % 5 == 0:
            print(f"  Processed {min(batch_start+batch_size, len(sentences))}/{len(sentences)} sentences, "
                  f"{len(rows)} word rows so far")

    # Write output CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id","sent_id","wnum","totsurp","lexsurp","synsurp"])
        writer.writeheader()
        writer.writerows(rows)

    non_nan = sum(1 for r in rows if r["totsurp"] == r["totsurp"])
    print(f"\nDone. Wrote {OUT_CSV}")
    print(f"  Total rows: {len(rows)}, valid surprisal: {non_nan}")

if __name__ == "__main__":
    if not test_parser():
        print("ERROR: Parser test failed. Check model file and binary.")
        sys.exit(1)
    print()
    run_full_parser()

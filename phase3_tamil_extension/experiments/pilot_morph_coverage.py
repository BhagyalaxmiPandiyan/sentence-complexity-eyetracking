"""
Pilot check (proposal Section 7 risk #1): how well does ThamizhiMorph cover
real Tamil text?

Runs every word of Dataset/raw_corpus/wiki_sample_20.txt through the
ThamizhiMorph FST models (lexicon models first, guessers only as fallback)
and reports:
  - % of words analysed by a lexicon model (reliable)
  - % only analysed by a guesser (unreliable)
  - % not analysed at all
  - % where the lemma appears unchanged at the start of the word, i.e.
    where we can cut the surface word into stem + suffix string (needed
    for morpheme-level surprisal later)

Must run inside WSL (flookup is a Linux binary):
  wsl -d Ubuntu -- python3 "/mnt/e/Project/Final One/phase3_tamil_extension/experiments/pilot_morph_coverage.py"
"""
import os, re, subprocess, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FST_DIR = os.path.join(BASE, "tools", "thamizhi-morph", "FST-Models")
CORPUS = os.path.join(BASE, "Dataset", "raw_corpus", "wiki_sample_20.txt")
OUT = os.path.join(BASE, "output", "pilot_morph_analyses.tsv")

FOMA = os.path.expanduser("~/foma_local/root/usr")
FLOOKUP = os.path.join(FOMA, "bin", "flookup")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.join(FOMA, "lib", "x86_64-linux-gnu"))

LEXICON_FSTS = ["noun.fst", "pronoun.fst", "adj.fst", "adv.fst", "part.fst",
                "verb-c3.fst", "verb-c4.fst", "verb-c62.fst", "verb-c11.fst",
                "verb-c12.fst", "verb-c-rest.fst"]
GUESS_FSTS = ["noun-guess.fst", "verb-guess.fst", "adj-guess.fst", "adv-guess.fst"]


def lookup(fst, words):
    """Run all words through one FST in a single flookup call -> {word: [analyses]}"""
    p = subprocess.run([FLOOKUP, os.path.join(FST_DIR, fst)], input="\n".join(words) + "\n",
                       capture_output=True, text=True, env=ENV)
    out = collections.defaultdict(list)
    for line in p.stdout.splitlines():
        if "\t" not in line:
            continue
        w, a = line.split("\t", 1)
        if a != "+?":
            out[w].append(a)
    return out


def main():
    sents = []
    for line in open(CORPUS, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[2].strip():
            sents.append(parts[2])
    tokens = [w for s in sents for w in re.findall(r"[஀-௿]+", s)]
    vocab = sorted(set(tokens))

    lex = collections.defaultdict(list)
    for f in LEXICON_FSTS:
        for w, a in lookup(f, vocab).items():
            lex[w] += a
    missing = [w for w in vocab if w not in lex]
    guess = collections.defaultdict(list)
    for f in GUESS_FSTS:
        for w, a in lookup(f, missing).items():
            guess[w] += a

    n = len(tokens)
    c_lex = sum(t in lex for t in tokens)
    c_guess = sum(t not in lex and t in guess for t in tokens)
    prefix_ok = sum(1 for t in tokens if t in lex and t.startswith(lex[t][0].split("+")[0]))
    ambig = sum(1 for t in tokens if t in lex and len(set(lex[t])) > 1)

    print(f"sentences={len(sents)}  word tokens={n}  unique={len(vocab)}")
    print(f"lexicon-analysed : {c_lex/n:6.1%}")
    print(f"guesser only     : {c_guess/n:6.1%}")
    print(f"unanalysed       : {(n-c_lex-c_guess)/n:6.1%}")
    print(f"ambiguous (>1 analysis, of lexicon hits): {ambig/max(c_lex,1):6.1%}")
    print(f"lemma is surface prefix (of lexicon hits): {prefix_ok/max(c_lex,1):6.1%}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("word\tsource\tanalyses\n")
        for w in vocab:
            src = "lexicon" if w in lex else "guesser" if w in guess else "none"
            f.write(f"{w}\t{src}\t{' | '.join(dict.fromkeys(lex.get(w) or guess.get(w) or []))}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

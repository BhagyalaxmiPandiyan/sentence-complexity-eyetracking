"""
Build a PCFG .model file from Penn Treebank WSJ sections 2-21
for use with the modelblocks left-corner parser (parser-x-efabp).

Output format (modelblocks CKY model):
  Cr : RHS_0 = prob         # root unary rule (LHS = ROOT)
  Cu LHS_0 : RHS_0 = prob   # non-root unary rule
  CC LHS_0 : RHS1_0 RHS2_0 = prob  # binary rule
  CC POS_0 : - - = 1        # lexical category declaration
  X POS_0 : word = prob      # lexical entry
"""

import os
import sys
import re
import glob
from collections import defaultdict

# NLTK for tree parsing
import nltk
from nltk import Tree

PTB_WSJ = "/mnt/e/Project/Final One/Dataset/penn_treebank_3/parsed/mrg/wsj"
OUT_MODEL = "/mnt/e/Project/Final One/output/wsj02to21.model"
TRAIN_SECTS = [f"{i:02d}" for i in range(2, 22)]  # sections 02..21
UNK_THRESHOLD = 1  # words seen <= this many times are mapped to <unk>


def strip_func_tags(label):
    """NP-SBJ -> NP, VP-TPC -> VP, keep -LRB-, -RRB-"""
    if label in ("-LRB-", "-RRB-", "-NONE-", "--"):
        return label
    label = re.sub(r"-[A-Z]+(?:-\d+)*$", "", label)
    label = re.sub(r"=\d+$", "", label)
    label = re.sub(r"-\d+$", "", label)
    return label if label else None


def clean_tree(tree):
    """Return a new Tree with functional tags stripped from all node labels."""
    if isinstance(tree, str):
        return tree
    label = strip_func_tags(tree.label())
    if label is None:
        label = "X"
    return Tree(label, [clean_tree(c) for c in tree])


def binarize_tree(tree):
    """Right-binarize an NLTK Tree in place using intermediate @LHS nodes."""
    if isinstance(tree, str):
        return
    for i, child in enumerate(tree):
        if not isinstance(child, str):
            binarize_tree(child)
    # Binarize if more than 2 children
    while len(tree) > 2:
        lbl = f"@{tree.label()}"
        last = tree.pop()
        second_last = tree.pop()
        tree.append(Tree(lbl, [second_last, last]))


def extract_from_tree(tree, unary_counts, binary_counts, lex_counts, word_counts, root_lhs):
    """Recursively extract grammar productions from an NLTK Tree."""
    if isinstance(tree, str):
        return

    lhs = tree.label()
    children = list(tree)

    # Pre-terminal: POS -> word
    if len(children) == 1 and isinstance(children[0], str):
        word = children[0].lower()
        lex_counts[lhs][word] += 1
        word_counts[word] += 1
        return

    child_labels = []
    for child in children:
        if isinstance(child, str):
            child_labels.append(f"<word:{child}>")
        else:
            child_labels.append(child.label())
            extract_from_tree(child, unary_counts, binary_counts, lex_counts, word_counts, root_lhs)

    if len(child_labels) == 1:
        unary_counts[lhs][child_labels[0]] += 1
    elif len(child_labels) == 2:
        binary_counts[lhs][(child_labels[0], child_labels[1])] += 1
    # len > 2 means binarize_tree didn't finish - should not happen


def build_grammar():
    print("Reading PTB WSJ sections 2-21...")
    unary_counts  = defaultdict(lambda: defaultdict(int))
    binary_counts = defaultdict(lambda: defaultdict(int))
    lex_counts    = defaultdict(lambda: defaultdict(int))
    word_counts   = defaultdict(int)
    root_counts   = defaultdict(int)

    tree_count = 0
    for sect in TRAIN_SECTS:
        sect_dir = os.path.join(PTB_WSJ, sect)
        if not os.path.isdir(sect_dir):
            print(f"  WARNING: section {sect} not found at {sect_dir}")
            continue
        for mrg_file in sorted(glob.glob(os.path.join(sect_dir, "*.mrg"))):
            with open(mrg_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Split into individual tree strings by tracking bracket depth
            tree_strs = []
            depth = 0
            start = -1
            for ci, ch in enumerate(content):
                if ch == "(":
                    if depth == 0:
                        start = ci
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and start >= 0:
                        tree_strs.append(content[start:ci+1])
                        start = -1
            for tree_str in tree_strs:
                try:
                    tree = Tree.fromstring(tree_str)
                except Exception:
                    continue

                # Get root label (TOP, S1, etc.)
                top_label = tree.label().upper()
                actual_root = "ROOT"

                # Record what comes directly under the top node
                for child in tree:
                    if isinstance(child, Tree):
                        clabel = strip_func_tags(child.label()) or child.label()
                        root_counts[clabel] += 1

                # Replace top label with ROOT
                tree = Tree(actual_root, list(tree))

                # Strip functional tags from all nodes
                tree = clean_tree(tree)

                # Right-binarize
                binarize_tree(tree)

                # Extract productions (skip ROOT level - handled separately)
                for child in tree:
                    if isinstance(child, Tree):
                        extract_from_tree(child, unary_counts, binary_counts,
                                           lex_counts, word_counts, actual_root)

                tree_count += 1

        if tree_count % 5000 == 0 and tree_count > 0:
            print(f"  {tree_count} trees processed...")

    print(f"  Total trees processed: {tree_count}")
    print(f"  Unary rules: {sum(len(v) for v in unary_counts.values())}")
    print(f"  Binary rules: {sum(len(v) for v in binary_counts.values())}")
    print(f"  Lexical categories: {len(lex_counts)}")
    print(f"  Vocabulary: {len(word_counts)} words")
    return unary_counts, binary_counts, lex_counts, word_counts, root_counts


def write_model(unary_counts, binary_counts, lex_counts, word_counts, root_counts, out_path):
    print(f"Writing model to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rare_words = {w for w, c in word_counts.items() if c <= UNK_THRESHOLD}
    print(f"  Rare words (<= {UNK_THRESHOLD}): {len(rare_words)}")

    # Build <unk> counts per POS from rare words
    unk_lex = defaultdict(int)
    normal_lex = defaultdict(lambda: defaultdict(int))
    for pos, words in lex_counts.items():
        for word, cnt in words.items():
            if word in rare_words:
                unk_lex[pos] += cnt
            else:
                normal_lex[pos][word] += cnt

    lines = []

    # Root unary rules (Cr): what goes directly under ROOT
    root_total = max(sum(root_counts.values()), 1)
    for rhs, cnt in sorted(root_counts.items()):
        prob = cnt / root_total
        lines.append(f"Cr : {rhs}_0 = {prob:.8f}\n")

    # Non-root unary rules (Cu)
    for lhs in sorted(unary_counts):
        if lhs == "ROOT":
            continue
        total = max(
            sum(unary_counts[lhs].values()) + sum(binary_counts.get(lhs, {}).values()),
            1
        )
        for rhs, cnt in sorted(unary_counts[lhs].items()):
            prob = cnt / total
            lines.append(f"Cu {lhs}_0 : {rhs}_0 = {prob:.8f}\n")

    # Binary rules (CC for phrase structure)
    for lhs in sorted(binary_counts):
        if lhs == "ROOT":
            continue
        total = max(
            sum(unary_counts.get(lhs, {}).values()) + sum(binary_counts[lhs].values()),
            1
        )
        for (rhs1, rhs2), cnt in sorted(binary_counts[lhs].items()):
            prob = cnt / total
            lines.append(f"CC {lhs}_0 : {rhs1}_0 {rhs2}_0 = {prob:.8f}\n")

    # Lexical entries
    for pos in sorted(lex_counts):
        words = normal_lex[pos]
        unk_cnt = unk_lex.get(pos, 0)
        total = max(sum(words.values()) + unk_cnt, 1)

        lines.append(f"CC {pos}_0 : - - = 1\n")
        if unk_cnt > 0:
            prob = unk_cnt / total
            lines.append(f"X {pos}_0 : <unk> = {prob:.8f}\n")
        for word, cnt in sorted(words.items()):
            prob = cnt / total
            lines.append(f"X {pos}_0 : {word} = {prob:.8f}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  Written {len(lines)} model lines")
    print(f"  Model size: {os.path.getsize(out_path) / 1024:.0f} KB")


if __name__ == "__main__":
    unary_counts, binary_counts, lex_counts, word_counts, root_counts = build_grammar()
    write_model(unary_counts, binary_counts, lex_counts, word_counts, root_counts, OUT_MODEL)
    print("Done.")

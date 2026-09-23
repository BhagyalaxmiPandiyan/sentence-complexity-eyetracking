"""
Compute GPT-2 contextual surprisal per word for the Dundee corpus, aligned
to the same (text_id, wnum) keys used throughout this project.

Motivation: syn_surprisal (van Schijndel left-corner parser) was confirmed
to carry ~0 signal for FFD (r=-0.017, adds 0.000 R² in the step4
ablation) — and the paper's own Table 1 shows the same near-zero
correlation (r=-0.067) in their data too. GPT-2 gives a contextual,
next-word surprisal that does not depend on a hand-built grammar/parser
and correlates far better with reading time (r=0.454).

Method: sentence-by-sentence teacher forcing. For each Dundee sentence,
run GPT-2 once, take -log P(token | previous tokens) for every subword
token (natural log, i.e. nats — same unit as lex_surprisal/syn_surprisal),
then sum subtoken surprisal within each whitespace-delimited word to get
one surprisal value per word. The first word of a sentence gets only the
BOS token as context (matches how the rest of the pipeline treats
sentence-initial words: no cross-sentence context).

Output: output/gpt2_surprisal.csv with columns text_id, wnum,
gpt2_surprisal — same schema as output/lcparse_syn_surprisal.csv, so it
can be merged into step3_syntactic_features.py / step4_system1.py the
same way.
"""
import os
import json
import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output")

SENT_TXT = os.path.join(OUT_DIR, "dundee_sentences.txt")
SENT_MAP = os.path.join(OUT_DIR, "dundee_sent_map.json")


def load_sentences():
    with open(SENT_TXT, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(SENT_MAP, encoding="utf-8") as f:
        sent_map = json.load(f)
    assert len(lines) == len(sent_map), (
        f"sentence count mismatch: {len(lines)} lines vs {len(sent_map)} map entries"
    )
    return lines, sent_map


def build_context(prev_sentences, tokenizer, max_sentences, max_tokens):
    """Take the most recent previous sentences (same text_id) that fit within
    max_tokens, most recent first, then re-order to reading order."""
    if max_sentences <= 0 or not prev_sentences:
        return ""
    picked = []
    total = 0
    for s in reversed(prev_sentences[-max_sentences:]):
        n = len(tokenizer.encode(s))
        if total + n > max_tokens:
            break
        picked.insert(0, s)
        total += n
    return (" ".join(picked) + " ") if picked else ""


def word_token_surprisals(sentence, tokenizer, model, device, context=""):
    """Return one surprisal value (nats) per whitespace-delimited word in `sentence`.

    If `context` is given, it's prepended (e.g. preceding sentences) so the
    model conditions on it, but surprisal is only summed for `sentence`'s
    own words — context tokens are scored (needed for causal LM) but
    discarded from the output.
    """
    words = sentence.split()
    full_text = context + sentence

    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
    full_ids = torch.cat([bos, input_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (seq_len, vocab)

    log_probs = torch.log_softmax(logits, dim=-1)
    # token i (in input_ids) is predicted by position i in full_ids (which includes BOS)
    target_ids = input_ids[0]
    token_logp = log_probs[:-1].gather(1, target_ids.unsqueeze(1)).squeeze(1)
    token_surprisal = (-token_logp).tolist()  # nats, one per subword token

    # map char offsets (within full_text) -> word index in `sentence`,
    # sum subtoken surprisal per word; offsets before len(context) belong
    # to the context and are skipped
    ctx_len = len(context)
    word_starts = []
    pos = ctx_len
    for w in words:
        start = sentence.index(w, pos - ctx_len) + ctx_len
        word_starts.append((start, start + len(w)))
        pos = start + len(w)

    word_surp = [0.0] * len(words)
    wi = 0
    for (tok_start, tok_end), surp in zip(offsets, token_surprisal):
        if tok_start == tok_end:
            continue  # special token, no offset
        if tok_end <= word_starts[0][0]:
            continue  # token entirely inside context (BPE can merge a leading
                       # space into the next word's token, so a token that
                       # *ends* past the boundary belongs to the first word,
                       # even if it *starts* inside the context)
        while wi < len(words) - 1 and tok_start >= word_starts[wi][1]:
            wi += 1
        word_surp[wi] += surp

    return word_surp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="HF model name (default: gpt2, 124M)")
    parser.add_argument("--limit", type=int, default=None, help="only process first N sentences (debug)")
    parser.add_argument("--out", default=None, help="output CSV path (default: output/gpt2_surprisal_<model>[_ctx].csv)")
    parser.add_argument("--context-sentences", type=int, default=0,
                         help="number of preceding sentences (same text_id) to prepend as context (default: 0, matches original sentence-isolated behavior)")
    parser.add_argument("--context-tokens", type=int, default=300,
                         help="max token budget for the prepended context (default: 300)")
    args = parser.parse_args()

    suffix = "_ctx" if args.context_sentences > 0 else ""
    out_csv = args.out or os.path.join(OUT_DIR, f"gpt2_surprisal_{args.model.replace('/', '_')}{suffix}.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.eval()

    lines, sent_map = load_sentences()
    if args.limit:
        lines, sent_map = lines[: args.limit], sent_map[: args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    prev_sentences_by_text = {}
    for i, (sentence, meta) in enumerate(zip(lines, sent_map)):
        wnums = meta["wnums"]
        text_id = meta["text_id"]

        prev_sentences = prev_sentences_by_text.setdefault(text_id, [])
        context = build_context(prev_sentences, tokenizer, args.context_sentences, args.context_tokens)

        surps = word_token_surprisals(sentence, tokenizer, model, device, context=context)
        assert len(surps) == len(wnums), (
            f"sentence {i} (text_id={text_id}): {len(surps)} words vs {len(wnums)} wnums"
        )
        for wnum, s in zip(wnums, surps):
            rows.append((text_id, wnum, round(s, 4)))

        prev_sentences.append(sentence)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(lines)} sentences...")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("text_id,wnum,gpt2_surprisal\n")
        for text_id, wnum, s in rows:
            f.write(f"{text_id},{wnum},{s}\n")

    print(f"Saved {len(rows):,} word surprisal values to {out_csv}")


if __name__ == "__main__":
    main()

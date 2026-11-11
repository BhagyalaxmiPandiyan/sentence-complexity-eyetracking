# Phase 3 Datasets

None of these are labeled "readability" datasets — no such thing exists for
Tamil yet (see novelty check in `../Tamil_Readability_Thesis_Proposal.md`,
section 2). Each folder holds a different *kind* of input the pipeline
needs, not a single pre-made benchmark.

## `raw_corpus/`
**Purpose:** generic Tamil text to sanity-check `ThamizhiMorph` segmentation
and to make sure the Tamil GPT-2 model produces sensible surprisal values,
before touching any real experiment data.

**Source (pick one, either works for this purpose):**
- Tamil Wikipedia dump via Hugging Face `datasets`: `load_dataset("wikimedia/wikipedia", "20231101.ta")`
- AI4Bharat `IndicCorpV2` (Tamil subset) — larger, more news-register text: https://huggingface.co/datasets/ai4bharat/IndicCorpV2

A few thousand sentences sampled from either is plenty; this is a pilot
sanity-check corpus, not training data (Phase 3 does not train an LM from
scratch, only uses a pretrained Tamil GPT-2).

## `complex_sentences/`
**Purpose:** the actual input sentences that Phase B's LLM simplifier will
rewrite, and that both metrics will score in Phase C. This is the dataset
that most directly represents "the thesis's test set" — quality here matters
more than quantity.

**Source:** Tamil news editorials (e.g. Dinamani, The Hindu Tamil) or
long/complex Tamil Wikipedia sentences, hand-picked — not scraped in bulk.
Target: 50-100 sentences, manually filtered for genuine syntactic/
morphological complexity (long agglutinated words, embedded clauses), not
just long sentences. Store as `complex_sentences/original.csv` with one
sentence per row plus a source citation column.

## `human_study/`
**Purpose:** the ground truth the two competing metrics (naive word-level
vs. morphology-aware) are validated against. This dataset does not exist
yet anywhere — it's the one Phase 3 creates itself in Phase C.

**How it gets built:** after Phase B generates original/simplified pairs,
10-20 native Tamil speakers rate each pair (e.g. via a Google Form) on
perceived simplification quality/ease. Store raw responses as
`human_study/ratings_raw.csv` and the aggregated per-pair score as
`human_study/ratings_aggregated.csv`.

## Not used (and why)
- **AoA norms** — none published for Tamil.
- **Eye-tracking reading-time corpus** — no Tamil coverage in MultiplEYE or
  elsewhere, so there is no System-1-style ground truth to replicate here.
- **Constituency treebank** — no mature Tamil parser at scale, so
  hierarchical/embedding-depth features from Phase 1/2 are dropped, not
  ported.

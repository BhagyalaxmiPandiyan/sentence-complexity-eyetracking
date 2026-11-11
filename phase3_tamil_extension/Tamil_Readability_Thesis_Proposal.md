# Morphology-Aware Complexity and LLM-Based Simplification for Tamil

*Proposed extension of the Singh et al. (2016) replication — Phase 3*

## Abstract

Computational readability and text-simplification research has largely
been built around English and other morphologically simple languages,
where a word's orthographic length is a reasonable proxy for how much
information it carries. Tamil, a morphologically rich, agglutinative
Dravidian language, breaks this assumption: a single Tamil word can carry
case, tense, negation, and agreement information that English spreads
across an entire clause, so word-length-based complexity metrics —
including the one used by Singh et al. (2016), whose method this thesis
already replicated and extended with GPT-2 surprisal for English in
Phase 2 — are structurally mismatched to Tamil. This thesis proposes and
tests a **morphology-aware complexity metric** for Tamil, combining
morpheme-level segmentation (via the open-source `ThamizhiMorph`
analyser) with language-model surprisal, and compares it against a naive
word-level metric ported directly from the English approach. The metric
is then used to drive and evaluate an **LLM-based Tamil sentence
simplification** system: a capable language model generates simplified
Tamil rewrites of complex sentences, and simplification quality is
scored using both metrics, with results validated against a small
native-speaker judgment study. The central claim is that the
morphology-aware metric tracks perceived simplification quality more
closely than the naive word-level metric — demonstrating that
complexity metrics designed for English do not transfer safely to
agglutinative languages, and providing Tamil with both a validated
complexity metric and a working simplification pipeline, neither of
which currently exists.

## 1. Motivation

- Singh et al. (2016)'s word-length feature is already an imperfect
  proxy in English — this thesis's own investigation found the paper's
  Table 1 (r=0.765) and Table 2 (R²=0.267) for word length alone are
  internally inconsistent (see `phase1_replication/README.md`,
  Improvement Roadmap #7). If word length is already unstable as a
  complexity signal in English, it is reasonable to expect it fails more
  seriously in a morphologically rich language like Tamil, where word
  length conflates orthographic length with stacked grammatical
  morphemes.
- This reframes the open question from this thesis's Phase 1/2 work
  (why does word length behave strangely?) into a testable hypothesis
  for Phase 3: word length is not just noisy, it is the *wrong unit of
  analysis* for agglutinative languages — morphemes are.
- A working, open-source Tamil morphological analyser exists
  (`ThamizhiMorph`, FST-based, GitHub: `sarves/thamizhi-morph`), making
  morpheme-level features genuinely buildable, not just theoretical.
- LLM-based text simplification is an active, modern NLP direction
  (cf. the TSAR workshop series) — grounding it in a Tamil-specific,
  linguistically motivated metric (rather than porting an English-style
  metric unchanged) is what makes this a contribution rather than a port.

## 2. Related Work / Novelty Check

Searched specifically for prior Tamil work before committing to this
direction:

- **No Tamil eye-tracking-while-reading corpus** — the multilingual
  MultiplEYE initiative does not yet cover Tamil, so ground-truth
  reading-time data (System 1 in the original paper) is unavailable.
- **No Tamil readability/complexity classifier or simplification
  system found** in DravidianLangTech (the main Tamil/Dravidian NLP
  workshop, ACL Anthology) — active Tamil work there covers sentiment,
  hate speech, ASR, and code-mixing, but not readability or complexity.
- **Closest Indic-language precedent**: a 2023 ACM TALLIP paper building
  a machine-learning readability model for **Gujarati** — confirms this
  general style of work is recognized and publishable for an Indic
  language, while leaving Tamil specifically unaddressed.
- One paper appeared promising but is **not** real prior art on closer
  inspection: Gooding et al. (2021, Google Research), "Predicting Text
  Readability from Scrolling Interactions," includes Tamil-first-language
  readers — but only as a demographic variable while those readers read
  **English** text; no Tamil text or Tamil-specific model was built.
- No published Tamil AoA (age-of-acquisition) norms and no mature Tamil
  constituency parser/treebank at scale were found — this is precisely
  why the original paper's AoA-based and parse-tree-based features
  (embedding depth/difference, hierarchical structure, entropy
  reduction) are not used in this proposal; morphological features are
  the substitute this thesis leans into instead.
- Caveat: this is a search-tool-based check, not exhaustive (paywalled
  papers and non-indexed regional proceedings may exist) — worth a
  direct Google Scholar/ICON-proceedings check before finalizing the
  novelty claim in the actual thesis writeup.

## 3. Research Questions

1. Does a morpheme-level complexity metric (morpheme count + morpheme-
   level surprisal) predict perceived Tamil sentence difficulty better
   than a word-level metric (word length + word-level GPT-2 surprisal,
   ported directly from the English approach)?
2. Can an LLM-generated Tamil simplification be reliably evaluated using
   the morphology-aware metric, and does it agree better with human
   judgments than the naive word-level metric?

## 4. What's Reused vs. What's New

| Component | Status |
|---|---|
| Word-level GPT-2 surprisal pipeline | Reused directly from Phase 2 (`compute_gpt2_surprisal.py`), swapped to a Tamil GPT-2 model |
| Word length / sentence length features | Reused as the "naive" baseline metric to beat |
| AoA, embedding depth/difference, hierarchical structure (h1–h8), entropy reduction | **Dropped** — no Tamil AoA norms or constituency parser exist; explicitly out of scope, not silently ignored |
| Morpheme segmentation + morpheme count | **New** — via `ThamizhiMorph` |
| Morpheme-level surprisal | **New** — feed morpheme sequences (not word sequences) through a Tamil LM |
| LLM-based simplification generator | **New** — prompted, not trained from scratch |
| Human judgment validation study | **New** — small native-speaker panel |

## 5. Plan (phased)

**Phase A — Morphology-aware metric (2 weeks)**
1. Integrate `ThamizhiMorph` to segment Tamil sentences into morphemes.
2. Build morpheme-level surprisal (morpheme sequence through a Tamil LM)
   alongside the existing word-level GPT-2 surprisal pipeline.
3. Define the composite morphology-aware complexity score.

**Phase B — Simplification generator (1–2 weeks)**
4. Prompt-engineer an LLM to simplify complex Tamil sentences (start with
   a small hand-picked set of complex Tamil sentences — news editorials
   are a convenient source).
5. Generate simplified candidates; filter for basic fluency/validity.

**Phase C — Evaluation (1–2 weeks)**
6. Score original vs. simplified pairs with both metrics (naive
   word-level vs. morphology-aware).
7. Run a small human judgment study (~10–20 native Tamil speakers, a few
   dozen sentence pairs) rating perceived simplification quality/ease.
8. Compare which metric's scores correlate better with human judgments —
   this comparison is the thesis's central empirical result.

**Phase D — Writeup (1 week)**
9. Report correlation results, discuss where/why the morphology-aware
   metric wins or loses, and package the pipeline for reuse.

**Total estimated time: ~6–8 weeks** — comparable in scope to the
earlier System-2-port plan, but the deliverable is a genuinely new
metric and system rather than a straight adaptation.

## 6. Tools & Resources

| Need | Source |
|---|---|
| Tamil morphological segmentation | `ThamizhiMorph` (github.com/sarves/thamizhi-morph) |
| Tamil surprisal (word- and morpheme-level) | `abinayam/gpt-2-tamil` or `Lagstill/GPT-2-Tamil` (Hugging Face) |
| Simplification generation | A capable general-purpose LLM, prompted (no fine-tuning needed for MVP) |
| Complex source sentences | Tamil newspaper editorials / Tamil Wikipedia |
| Human validation | Small panel of native Tamil speakers (~10–20) |

## 7. Risks / Open Questions

- `ThamizhiMorph`'s coverage/accuracy on newspaper-register text is
  unverified — needs a quick pilot check before committing Phase A's
  timeline.
- LLM-generated Tamil simplifications may have fluency or meaning-
  preservation issues that need manual filtering — budget time for this.
- Human validation panel size (~10–20) is small; frame results as an
  initial/pilot-scale finding, not a definitive claim — honest framing
  is stronger than overclaiming on a small n.
- The morphology-aware metric could turn out *not* to beat the naive
  one — that would still be a valid, reportable (if less exciting)
  thesis result, since the research question is genuinely open.

## 8. Expected Contribution

1. First morphology-aware text-complexity metric for Tamil, directly
   motivated by a weakness (word length as a proxy) this thesis already
   identified empirically in the English replication.
2. First Tamil sentence-simplification pipeline evaluated against a
   linguistically-motivated (not just ported) complexity metric.
3. A concrete empirical answer to whether English-shaped complexity
   metrics transfer to agglutinative languages — a comparative finding
   of interest beyond Tamil specifically (any agglutinative language:
   Turkish, Finnish, Korean, etc. faces the same mismatch).

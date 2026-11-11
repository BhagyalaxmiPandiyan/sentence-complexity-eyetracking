# Phase 3 Feasibility Pilot (2026-09-29)

Tested each component of the proposal on this machine (Windows + WSL Ubuntu,
CPU-only PyTorch, GTX 1650 Ti 4 GB, 24 GB RAM) before writing pipeline code.

## Verdict per component

| Step | Component | Verdict | Evidence |
|---|---|---|---|
| A1 | ThamizhiMorph segmentation | **Works, but low coverage** | 42.2% of word tokens analysed by lexicon; 53.8% guesser-only (often wrong); 4.0% none |
| A1 | Morfessor (unsupervised) as backup | **Works, noisy** | 74% of words left unsplit, rare words over-split |
| A2 | Tamil GPT-2 surprisal (`abinayam/gpt-2-tamil`) | **Works** | Loads on CPU, fluent generation, 13.1 bits/word on sample |
| A2 | `Lagstill/GPT-2-Tamil` | **Not usable** | Only TensorFlow weights (`tf_model.h5`), no PyTorch |
| A3 | Word-level baseline | **Trivial** | Direct port of Phase 2 script |
| B | LLM simplification | **Feasible, needs an API** | No local option (4 GB GPU); ~100 sentences via API costs a few dollars |
| C | Human study | **Feasible only if participants are recruited** | Not a technical issue; may need ethics approval |

## Key findings

### 1. ThamizhiMorph: accurate when it knows the word, unreliable otherwise

Run: `experiments/pilot_morph_coverage.py` (inside WSL) → `output/pilot_morph_analyses.tsv`

On 20 Tamil Wikipedia sentences (223 word tokens):

| | % of tokens |
|---|---|
| Analysed by lexicon model | 42.2% |
| Guesser only | 53.8% |
| Not analysed | 4.0% |
| Ambiguous (>1 lexicon analysis) | 25.5% of lexicon hits |
| Lemma appears unchanged at start of word | 76.6% of lexicon hits |

Lexicon analyses are good:
`அழகுக்குப் → அழகு+noun+dat+sandhip`, `இடைக்காலத்தில் → இடைக்காலம்+noun+infInc+loc`.

Guesser analyses are often wrong, precisely on the long, heavily
suffixed words the thesis is about:
`அலங்காரங்களும் → +noun+sg+nom` (misses plural கள் and உம்),
`அளவுவிகிதங்களைக் → +verb` (it is a noun).

Also: ThamizhiMorph outputs **lemma + grammatical tags**, not a surface
split. For morpheme-level surprisal we need to know which *characters*
belong to which morpheme; that works directly only when the lemma is a
prefix of the word (76.6%), not when sandhi changes the stem
(e.g. இடைக்காலம் → இடைக்காலத்-).

### 2. Morfessor alone is not a fix

Run: `experiments/pilot_morfessor.py` → `output/pilot_morfessor_segments.tsv`
(trained on 5,000 Tamil Wikipedia articles, ~2M tokens)

- Good: `அழகு + க்குப்`
- Under-segments frequent words: `கட்டிடங்கள்`, `இடைக்காலத்தில்` left whole
- Over-segments rare words: `ஒரு + ங் + கி + னை + ப் + பை + யும்`

### 3. Tamil GPT-2 tokenizes at roughly character level

`abinayam/gpt-2-tamil` uses the English GPT-2 byte-level BPE vocabulary:
7.3 tokens per Tamil word, vowel signs split off
(`கட | ் | ட | ி | டக | ...`). Consequence: GPT-2 tokens never line up with
morphemes, so morpheme surprisal must be computed as the **sum of token
surprisals over each morpheme's character span**. Works, but needs a
surface (character-level) segmentation as input.

## Recommended design change

Use a **hybrid segmenter** instead of ThamizhiMorph alone:

1. ThamizhiMorph lexicon hit → use its analysis (morpheme count = lemma + tags)
2. Otherwise → rule-based stripper for Tamil's small closed set of
   inflectional suffixes (plural கள், cases ஐ/க்கு/இல்/இன்/ஆல்/ஓடு/உடன்,
   clitics உம்/ஏ/ஓ, ஆன/ஆக ...), applied right to left
3. Morfessor only as a last resort

Validate it on a **hand-checked gold set of ~100 words**, segmented by a
native speaker (the student). This gives a reportable segmentation accuracy
number, and the gold set is itself a small contribution.

## Remaining inputs still needed from the student

- `Dataset/complex_sentences/`: 50–100 complex Tamil sentences (not collected yet)
- LLM access for Phase B (Claude or Gemini API key)
- 10–20 native Tamil speakers for Phase C; check whether the university requires ethics approval

## Setup reproduced here

- foma installed without sudo: `experiments/setup_foma_wsl.sh` (WSL, into `~/foma_local`)
- ThamizhiMorph FST models: `git clone https://github.com/sarves/thamizhi-morph tools/thamizhi-morph`
- `pip install morfessor`

---

## Update (same day): existing open resources beat the home-made design

Run: `experiments/pilot_open_resources.py`, `experiments/eval_stanza_ud.py`

| Resource | What it gives | Result |
|---|---|---|
| **Stanza Tamil** (Stanford, `pip install stanza`) | Lemma + case/number/tense/person for every word, using sentence context | On UD Tamil-TTB **test** (hand-annotated news text, 1,989 words): lemma **91.4%**, POS 86.7%, all features exact **88.9%**, case **94.6%** |
| **IndicNLP Tamil segmenter** (pretrained Morfessor, `indic_nlp_resources/morph/morfessor/ta.model`) | Surface split into stem + suffixes | Better than our own Morfessor: `கட்டமைப்பு + களை`, `தத்துவ + த்தின்`, `கட்டிடங் + கள்`; still leaves 54% of words unsplit |
| **UD Tamil-TTB treebank** | 600 hand-annotated sentences (9,581 words) with gold lemma + morphology | Free gold evaluation set, so no need to hand-label 100 words ourselves |
| **wordfreq** (`pip install wordfreq`) | Tamil word frequency (Zipf scale) | Works; replaces Phase 1's `wiki_freq` feature |

Words ThamizhiMorph got wrong, which Stanza gets right:
`அளவுவிகிதங்களைக் → அளவுவிகிதம், Plur, Acc`; `கட்டமைப்புகளை → கட்டமைப்பு, Plur, Acc`;
`அலங்காரங்களும் → அலங்காரம், Plur` (still misses the clitic உம்).
Stanza still errs on some verbs (`அமைந்திருக்கிறது → lemma அமைந்திர்`, `கலைஞனதும் → VERB`).

### Revised design (replaces the hybrid segmenter above)

| Need | Use |
|---|---|
| Morpheme count / morphological features | **Stanza** (non-default features per word: Case≠Nom, Plur, Tense, Person, …) |
| Surface morpheme spans for morpheme surprisal | **IndicNLP segmenter** |
| Morpheme and word surprisal | `abinayam/gpt-2-tamil` |
| Word frequency | **wordfreq** |
| Segmenter/feature accuracy to report | **UD Tamil-TTB** gold |
| ThamizhiMorph | Optional cross-check only; drops the WSL/foma dependency |

Everything above runs on Windows with pip installs, with no WSL needed.

---

## Update: graded texts as readability ground truth (replacing part of the human study)

No open Tamil readability dataset exists (web search, Hugging Face search,
Tamil NLP catalogues). Two graded sources were tested:

### Tamil Nadu state textbooks (Class 1–12): **usable**
- Free PDFs for every class, Tamil medium. The official site
  (textbooksonline.tn.nic.in) was unreachable from here; the same PDFs are
  mirrored on Google Drive via tntextbooks.net.
- Tested: Class 6 Term 1 Tamil (2024–25), 160 pages, 12,795 Tamil words.
- Raw extraction is garbled by the PDF font (`அடித்்தளம்`, `தொொடக்்கமும்`):
  only 55.3% of words are real words.
- Three regex repairs (`experiments/pilot_textbook_extraction.py`) bring it
  to **76.5%**, close to the Wikipedia baseline (78.7%). Remaining unknowns are
  mostly genuine textbook words (`எழுதுக`, `குறுவினா`), plus a few broken
  fragments (`ொருள்`) that can be filtered out.
- Caveats: (1) Tamil *subject* books contain classical poetry (செய்யுள்),
  so use only prose sections (உரைநடை), or Tamil-medium science/social-science
  books; (2) the grade is a document-level label, so it is noisy per sentence;
  (3) government copyright: use for research, don't redistribute the text.

### StoryWeaver (Pratham Books): **usable, but needs a free account**
- 3,211 Tamil stories with reading levels 1–4 (824 / 1,259 / 896 / 232),
  **CC BY 4.0** (the cleanest license of all sources).
- Search API works without login; story text returns 401 without login, so
  a free account is needed to download ePub/PDF.
- Covers early readers only (levels 1–4), not adult-level text.

SIL Bloom Library: no Tamil, gated. Ruled out.

### How this changes Phase C
Build sentence pairs (low-grade vs high-grade sentence) and test which
metric ranks them correctly: the same pairwise setup as Phase 1 System 2.
This gives **thousands of automatically labelled pairs** instead of a few
dozen human-rated ones. The small human study is still needed, but only for
judging the LLM *simplifications* (meaning preserved? fluent?), which grade
labels can't answer.

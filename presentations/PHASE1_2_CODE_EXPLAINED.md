# Phase 1 + Phase 2: Code Explained (file by file, line by line)

Use this when the professor asks "show me where you did that in the code".
Every line number below is from the current code (checked 2026-10-20).
Click a link to jump straight to that line in VS Code.

All paths are inside `E:\Project\Final One\phase1_replication\`.

**For every feature calculated by hand from the raw data, with real numbers, see
[PHASE1_2_FEATURES_BY_HAND.md](PHASE1_2_FEATURES_BY_HAND.md).**

---

## 0. The big picture in one picture

```
 Dataset/ (INPUTS, never changed)            output/ (everything the code makes)
 ─────────────────────────────────           ─────────────────────────────────────
 dundee_corpus/RT_dataset2P/*.dat ──step1──▶ dundee_rt.csv            (307,383 rows)
 dundee_corpus/per-word-info/                    │
 dundee_corpus/per-word-lexical-annotation/      ▼
 AoA.csv, BNC/, wordfreq library ─────step2──▶ lexical_features.csv    (307,383 rows)
                                                 │   + bnc_bigrams.csv, bnc_bigrams_full.csv (cache)
 penn_treebank_3/, dependency parse ──step3──▶ syntactic_features.csv  (50,647 rows)
 lcparse_syn_surprisal.csv (parser) ──┘          │   + ptb_pos_lm.pkl (cache)
                                                 ▼
                                      step4──▶ system1_models.pkl  +  system1_predictions.csv
                                                 │
 Wiki/aligned-good(0.67).gz ──────────step5──▶ accuracy printed on screen (74.15% / 75.26%)

 PHASE 2
 dundee_sentences.txt + dundee_sent_map.json ─compute_gpt2_surprisal─▶ gpt2_surprisal_gpt2_ctx.csv
 lexical + syntactic + gpt2 CSVs ─────────────train_system1_gpt2────▶ before/after R² table on screen
```

**Rule of the pipeline:** each step **reads CSVs from `output/`** and **writes a new CSV to `output/`**.
The paths are set at the top of every file, for example
[step1_parse_dundee.py:22-23](../phase1_replication/src/step1_parse_dundee.py#L22-L23):
`DUNDEE = ".../Dataset/dundee_corpus"` and `OUT = ".../output"`.

**The join key everywhere is `(text_id, wnum)`** = (which of the 20 newspaper texts, which word number).
That is how a word's reading time, its features and its GPT-2 score are matched together.

---

## 1. One example sentence used all the way through

Dundee text 1, sentence 1 (sent_id 10000):

> **Are tourists enticed by these attractions threatening their very existence?**

We follow the word **"tourists"** (text_id 1, wnum 2) through every step.
All values below are real, copied from the output files.

---

## 2. STEP 1: `src/step1_parse_dundee.py`: raw eye movements → reading times

**Run:** `python src/step1_parse_dundee.py` (about 3 min)

| | |
|---|---|
| **Input** | `Dataset/dundee_corpus/RT_dataset2P/*.dat`: one file per reader per article (10 readers `a`–`j`, 20 articles). Each line = one fixation: which word (WNUM), how long (FDUR, ms), and in what order (FXNO). |
| | `Dataset/dundee_corpus/per-word-info/*.dat`: the word list (word, length, global word number) |
| | `Dataset/dundee_corpus/per-word-lexical-annotation/text*.txt`: which sentence each word belongs to |
| **Output** | `output/dundee_rt.csv`: **307,383 rows**, one row per (word × reader) that was looked at |
| **Columns** | `text_id, wnum, subj_id, sent_id, word, wlen, ffd, fpd, rpd, td` |

### The 4 reading times (what the eye did)

| Measure | Simple meaning | Code line |
|---|---|---|
| **FFD** First Fixation Duration | how long the **first** look lasted | [line 83](../phase1_replication/src/step1_parse_dundee.py#L83) `ffd = fdurs[0]` |
| **TD** Total Duration | **all** looks added up, even coming back later | [line 84](../phase1_replication/src/step1_parse_dundee.py#L84) `td = sum(fdurs)` |
| **FPD** First Pass Duration | looks added up **until the eye leaves the word** the first time | [lines 89-100](../phase1_replication/src/step1_parse_dundee.py#L89-L100): keep adding while no other word was looked at in between |
| **RPD** Regression Path Duration | looks on this word **before the eye moves to a later word** | [lines 104-111](../phase1_replication/src/step1_parse_dundee.py#L104-L111): find the first fixation on a word with a bigger wnum |

### Key lines to point at

| Lines | What it does (say this) |
|---|---|
| [27-51](../phase1_replication/src/step1_parse_dundee.py#L27-L51) `parse_rt2p_file()` | Reads one reader's file. Takes column 6 = word number, 7 = duration, 10 = fixation order ([lines 42-44](../phase1_replication/src/step1_parse_dundee.py#L42-L44)). Keeps only real fixations (`fdur > 0`). |
| [54-115](../phase1_replication/src/step1_parse_dundee.py#L54-L115) `compute_measures_one_article()` | Sorts fixations in time order and computes FFD/FPD/RPD/TD for every word. |
| [124-146](../phase1_replication/src/step1_parse_dundee.py#L124-L146) `load_word_index()` | Reads the word list: word text, length (`wlen`, [line 143](../phase1_replication/src/step1_parse_dundee.py#L143)) and global word number. |
| [149-179](../phase1_replication/src/step1_parse_dundee.py#L149-L179) `build_sentence_map()` | Gives each word a sentence id: `text_num * 10000 + sent_id` ([line 176](../phase1_replication/src/step1_parse_dundee.py#L176)), so text 1 sentence 0 → **10000**. |
| [205](../phase1_replication/src/step1_parse_dundee.py#L205) | Reader id comes from the file name: `sa01ma2p.dat` → letter `a`. |
| [237-238](../phase1_replication/src/step1_parse_dundee.py#L237-L238) | Words a reader skipped are not added. |
| [255-257](../phase1_replication/src/step1_parse_dundee.py#L255-L257) | **Saves** `output/dundee_rt.csv`. |

### Real example: "tourists" in `dundee_rt.csv` (10 readers)

| subj_id | ffd | fpd | rpd | td |
|---|---|---|---|---|
| 0 | 156 | 156 | 156 | 156 |
| 2 | 126 | 360 | 360 | 360 |
| 3 | 221 | 221 | 221 | **527** ← came back later, so TD > FPD |
| 6 | 177 | 526 | 526 | 738 |
| … | | | | |

Reader 3 looked at "tourists" once (221 ms), moved on, and **came back** later.
That is why their TD (527) is bigger than their FPD (221).

---

## 3. STEP 2: `src/step2_lexical_features.py`: word features

**Run:** `python src/step2_lexical_features.py` (about 10 min, mostly reading the BNC cache)

| | |
|---|---|
| **Input** | `output/dundee_rt.csv` (from step 1) |
| | `Dataset/AoA.csv`: age-of-acquisition norms (Kuperman et al.) |
| | `wordfreq` Python library: word frequency |
| | `Dataset/BNC/`: British National Corpus, 100M words (only on the first run; then the cache is used) |
| **Output** | `output/lexical_features.csv`: **307,383 rows** (same rows as step 1 + 6 new columns) |
| **Cache files** | `output/bnc_bigrams.csv` (word counts), `output/bnc_bigrams_full.csv` (word-pair counts) |
| **Columns** | `text_id, wnum, subj_id, sent_id, word, wlen, sent_len, wiki_freq, aoa_mean, aoa_std, fwd_prob, bwd_prob, ffd, fpd, rpd, td` |

### Feature by feature

| Feature (CSV column) | Simple meaning | Where it's computed | "tourists" value |
|---|---|---|---|
| `wlen` | number of letters | already from step 1 ([step1 line 143](../phase1_replication/src/step1_parse_dundee.py#L143)) | **8** |
| `sent_len` | words in the sentence | [lines 213-216](../phase1_replication/src/step2_lexical_features.py#L213-L216), called at [235](../phase1_replication/src/step2_lexical_features.py#L235) | **10** |
| `wiki_freq` | how common the word is: log(frequency). Closer to 0 = common, very negative = rare | [lines 31-41](../phase1_replication/src/step2_lexical_features.py#L31-L41), formula at [39](../phase1_replication/src/step2_lexical_features.py#L39): `log(word_frequency(w,'en') + 1e-9)` | **−11.37** ("by" = −5.39, common) |
| `aoa_mean` | age (years) when children usually learn the word | [lines 46-76](../phase1_replication/src/step2_lexical_features.py#L46-L76), uses column `AoA_Kup` ([line 60](../phase1_replication/src/step2_lexical_features.py#L60)) | **8.76** years |
| `aoa_std` | how much the different AoA lists disagree | [lines 65-70](../phase1_replication/src/step2_lexical_features.py#L65-L70): std of 4 AoA columns | **0.0** |
| `fwd_prob` | how likely this word is **after the previous word**, log P(w \| prev) | [lines 188-193](../phase1_replication/src/step2_lexical_features.py#L188-L193), formula at [193](../phase1_replication/src/step2_lexical_features.py#L193) (add-1 smoothing) | **−11.65** |
| `bwd_prob` | how likely this word is **before the next word**, log P(w \| next) | [lines 198-203](../phase1_replication/src/step2_lexical_features.py#L198-L203) | **−12.72** |

### Key lines to point at

| Lines | What it does |
|---|---|
| [230](../phase1_replication/src/step2_lexical_features.py#L230) | Features belong to a **word**, not a reader, so we work on unique `(text_id, wnum)`: 50,647 words. |
| [81-94](../phase1_replication/src/step2_lexical_features.py#L81-L94) | Reads `<w>word</w>` tags from BNC XML files. |
| [154-159](../phase1_replication/src/step2_lexical_features.py#L154-L159) | Counts every word pair (w1 followed by w2) in BNC. |
| [104-129](../phase1_replication/src/step2_lexical_features.py#L104-L129) | If the cache exists, load it instead of re-reading BNC (saves 20-40 min). |
| [247-254](../phase1_replication/src/step2_lexical_features.py#L247-L254) | A word not in the AoA list gets the **average** AoA. |
| [287](../phase1_replication/src/step2_lexical_features.py#L287) | Copies the word features back onto every reader's row. |
| [296-298](../phase1_replication/src/step2_lexical_features.py#L296-L298) | **Saves** `output/lexical_features.csv`. |

---

## 4. STEP 3: `src/step3_syntactic_features.py`: grammar features

**Run:** `python src/step3_syntactic_features.py` (about 35 sec)

| | |
|---|---|
| **Input** | `output/lexical_features.csv` (from step 2) |
| | `output/bnc_bigrams.csv`, `bnc_bigrams_full.csv` (BNC caches from step 2) |
| | `Dataset/dundee_corpus/per-word-lexical-annotation/`: POS tags + dependency tree for every sentence |
| | `Dataset/penn_treebank_3/` → POS model cached in `output/ptb_pos_lm.pkl` |
| | `output/lcparse_syn_surprisal.csv`: parser surprisal, made once by `experiments/build_lcparse_surprisal.py` (runs the real left-corner parser in WSL Linux) |
| **Output** | `output/syntactic_features.csv`: **50,647 rows** (one per word, not per reader) |
| **Columns** | `text_id, wnum, total_surprisal, lex_surprisal, syn_surprisal, entropy_red, emb_depth, emb_diff, h1 … h8` |

### Feature by feature

| Feature | Simple meaning | Where it's computed | "tourists" value |
|---|---|---|---|
| `lex_surprisal` | how **surprising** the word is after the previous word: −log P(w \| prev), from BNC | [lines 94-109](../phase1_replication/src/step3_syntactic_features.py#L94-L109), formula at [108-109](../phase1_replication/src/step3_syntactic_features.py#L108-L109) | **11.65** |
| `syn_surprisal` | how surprising the **grammar structure** is (from the parser) | read from `lcparse_syn_surprisal.csv` at [lines 341-349](../phase1_replication/src/step3_syntactic_features.py#L341-L349) | **6.74** |
| `total_surprisal` | lex + syn | [line 347](../phase1_replication/src/step3_syntactic_features.py#L347) (saved, but **not used** by the model) | 18.38 |
| `entropy_red` | how much **uncertainty drops** about what comes next (POS-based) | [lines 295-301](../phase1_replication/src/step3_syntactic_features.py#L295-L301): `max(0, H_prev − H_now)` | **0.010** |
| `emb_depth` | how deep the word is in the sentence tree (steps to the root) | [lines 135-196](../phase1_replication/src/step3_syntactic_features.py#L135-L196) | **1** |
| `emb_diff` | depth now − depth of previous word | [line 307](../phase1_replication/src/step3_syntactic_features.py#L307) | **0** |
| `h1`–`h8` | 8 yes/no flags about how the tree depth changes from word to word | [lines 201-241](../phase1_replication/src/step3_syntactic_features.py#L201-L241) | h8 = 1 (same depth as "Are") |

`h1` depth changed · `h2` went deeper · `h3` went up · `h4` word is the root · `h5` always 0 (not computable, [line 233](../phase1_replication/src/step3_syntactic_features.py#L233), so step 4 drops it) · `h6` jumped deeper by >1 · `h7` jumped up by >1 · `h8` same depth.

### Key lines to point at

| Lines | What it does |
|---|---|
| [253](../phase1_replication/src/step3_syntactic_features.py#L253) | Drop duplicates → one row per word (grammar doesn't depend on the reader). |
| [276-318](../phase1_replication/src/step3_syntactic_features.py#L276-L318) | Main loop: goes through each text **in word order**, because surprisal needs the previous word. |
| [291](../phase1_replication/src/step3_syntactic_features.py#L291) | `syn_surp = 0.0` is only a placeholder, replaced by the real parser value at [346](../phase1_replication/src/step3_syntactic_features.py#L346). |
| [354-355](../phase1_replication/src/step3_syntactic_features.py#L354-L355) | **Saves** `output/syntactic_features.csv`. |

**Why "by" matters (Phase 2 link):** "by" in *"enticed by"* has `lex_surprisal` = **10.64** (the bigram method finds it surprising),
but GPT-2 gives it **0.90** (it knows "enticed by" is a normal phrase).

---

## 5. STEP 4: `src/step4_system1.py`: System 1 (predict reading time)

**Run:** `python src/step4_system1.py` (about 8 sec), **the best one to run live**

| | |
|---|---|
| **Input** | `output/lexical_features.csv`, `output/syntactic_features.csv`, `output/lcparse_syn_surprisal.csv` |
| **Output 1** | `output/system1_models.pkl`: the 4 trained models (FFD, FPD, RPD, TD), each with its scaler, threshold and feature list |
| **Output 2** | `output/system1_predictions.csv`: **50,647 rows**, predicted FFD for every word |
| **On screen** | Ablation table (paper Table 2) and the final R² table (paper Table 3) |

### What happens, in order

| # | Lines | What it does (say this) |
|---|---|---|
| 1 | [36-49](../phase1_replication/src/step4_system1.py#L36-L49) `FEATURE_GROUPS` | The feature list, **in the same order as the paper's Table 2**. |
| 2 | [55-117](../phase1_replication/src/step4_system1.py#L55-L117) `load_data()` | Joins lexical + syntactic features on `(text_id, wnum)` ([line 90](../phase1_replication/src/step4_system1.py#L90)). |
| 3 | [77-82](../phase1_replication/src/step4_system1.py#L77-L82) | **Averages reading time over the 10 readers**: sum ÷ 10. 307,383 rows → 50,647 words. "tourists" FFD = (156+156+126+221+206+137+177+270+108+179) ÷ 10 = **173.6 ms**. |
| 4 | [72-75](../phase1_replication/src/step4_system1.py#L72-L75) | Recomputes `sent_len` from unique words (safety check). |
| 5 | [120-143](../phase1_replication/src/step4_system1.py#L120-L143) `split_by_sentence()` | Splits by **sentence**, not by word: 60% train / 20% dev / 20% test = 1,421 / 473 / 475 sentences. `seed=42` makes it the same every run. **Why:** words from one sentence must not appear in both train and test, or the test would be too easy. |
| 6 | [178-211](../phase1_replication/src/step4_system1.py#L178-L211) `ablation_study()` | Adds feature groups **one at a time** and prints R² after each → paper Table 2. |
| 7 | [232-233](../phase1_replication/src/step4_system1.py#L232-L233) | `StandardScaler`: puts all features on the same scale (mean 0, std 1). |
| 8 | [236-244](../phase1_replication/src/step4_system1.py#L236-L244) | **Ridge regression**, trying 5 strengths (alpha) and keeping the best on dev. Ridge = linear regression that handles features which overlap (the h-flags). |
| 9 | [153-173](../phase1_replication/src/step4_system1.py#L153-L173) `tune_threshold()` | Tries thresholds 50-200 ms: any prediction below T becomes 0 (= "the reader skips this word"). Best for FFD = **56 ms**. For FPD/RPD/TD no threshold helped, so "none". |
| 10 | [253-256](../phase1_replication/src/step4_system1.py#L253-L256) | Final **test R²**, on sentences the model has never seen. |
| 11 | [294-296](../phase1_replication/src/step4_system1.py#L294-L296) | **Saves** `system1_models.pkl`. |
| 12 | [305-315](../phase1_replication/src/step4_system1.py#L305-L315) | Predicts FFD for every word and **saves** `system1_predictions.csv`. |

### How many features does the model use?
[Line 268](../phase1_replication/src/step4_system1.py#L268) keeps only columns that actually vary, giving **19 columns**:
`wlen, sent_len, wiki_freq, aoa_mean, aoa_std, fwd_prob, bwd_prob, lex_surprisal, syn_surprisal, entropy_red, emb_depth, emb_diff` (12) + `h1 h2 h3 h4 h6 h7 h8` (7; `h5` dropped because it is always 0).

### Real example from `system1_predictions.csv`

| word | predicted_ffd | real ffd (10-reader avg) |
|---|---|---|
| Are | 101.8 | 108.0 |
| **tourists** | **174.7** | **173.6** |
| enticed | 184.8 | 227.2 |
| by | 75.2 | 145.1 |
| attractions | 205.4 | 213.0 |

### Result (what appears on screen)
```
Measure  Threshold  Train R²  Dev R²  Test R²
FFD      56 ms      0.498     0.506   0.444
FPD      none       0.582     0.582   0.558
RPD      none       0.579     0.583   0.555
TD       none       0.562     0.564   0.536
```
**R² = 0.444** means the model explains about 44% of why reading times differ between words.

---

## 6. STEP 5: `src/step5_system2.py`: System 2 (which sentence is harder?)

**Run:** `python src/step5_system2.py` (about 22 min, **don't run live**)

| | |
|---|---|
| **Input** | `output/system1_models.pkl` (from step 4), `Dataset/AoA.csv`, `wordfreq` |
| | `Dataset/Wiki/aligned-good(0.67).gz`: 117,712 pairs: a normal Wikipedia sentence + its Simple Wikipedia version |
| **Output** | **Nothing saved**; accuracy is printed on screen |

### What happens, in order

| # | Lines | What it does |
|---|---|---|
| 1 | [136-163](../phase1_replication/src/step5_system2.py#L136-L163) `load_wiki_pairs()` | Reads pairs. Column 0 = Wikipedia (hard), column 1 = Simple (easy) ([lines 153-154](../phase1_replication/src/step5_system2.py#L153-L154)). Skips identical pairs ([156](../phase1_replication/src/step5_system2.py#L156)). |
| 2 | [104-129](../phase1_replication/src/step5_system2.py#L104-L129) `get_features()` | Makes features for a **new** sentence. Wiki sentences have no eye-tracking data or parser output, so it uses the word features (length, frequency, AoA, sentence length) and sets the rest to 0 ([lines 284-285](../phase1_replication/src/step5_system2.py#L284-L285): `syn_ext = None`, because running a parser on 235K sentences ran out of memory). |
| 3 | [48-72](../phase1_replication/src/step5_system2.py#L48-L72) `predict_ffd_for_sentence()` | Uses the **System 1 FFD model** to predict reading time for each word ([lines 67-71](../phase1_replication/src/step5_system2.py#L67-L71): scale → predict → apply 56 ms threshold). |
| 4 | [168-204](../phase1_replication/src/step5_system2.py#L168-L204) `build_pair_features()` | **Base RT model.** Each sentence → 60 numbers (predicted FFD per word, padded with 0). Pair = 60 + 60 = **120 numbers**. |
| 5 | [195-202](../phase1_replication/src/step5_system2.py#L195-L202) | Each pair is added **twice**: (hard, easy) → label 1 and (easy, hard) → label 0. So 117,712 pairs → **235,424 rows**, and the model can't cheat by always saying "first one". |
| 6 | [207-248](../phase1_replication/src/step5_system2.py#L207-L248) `build_extended_features()` | **Extended RT model.** Adds 6 sentence numbers: length, mean predicted FFD and total surprisal for both sentences → **126 numbers**. |
| 7 | [304](../phase1_replication/src/step5_system2.py#L304) | **10-fold cross-validation**: split into 10 parts, train on 9, test on 1, repeat 10 times, then average. |
| 8 | [315-318](../phase1_replication/src/step5_system2.py#L315-L318) | **Logistic Regression** = "pairwise classification" in the paper. |
| 9 | [321-324](../phase1_replication/src/step5_system2.py#L321-L324) | **LinearSVC** = our approximation of the paper's SVMrank. |

### Result (on screen)
```
Base RT     - Pairwise Classification: 74.15%   (paper 73.82%)
Base RT     - SVMrank:                 74.13%   (paper 73.79%)
Extended RT - Pairwise Classification: 75.26%   (paper 75.21%)
Extended RT - SVMrank:                 75.23%   (paper 75.09%)
```

---

## 7. PHASE 2, part 1: `experiments/compute_gpt2_surprisal.py`

**Run (full):** `python experiments/compute_gpt2_surprisal.py --context-sentences 3 --context-tokens 300` (about 9 min)
**Run (live demo):** `python experiments/compute_gpt2_surprisal.py --context-sentences 3 --limit 5 --out output/gpt2_demo.csv` (13 sec)

| | |
|---|---|
| **Input** | `output/dundee_sentences.txt`: the 2,368 Dundee sentences, one per line, lowercase |
| | `output/dundee_sent_map.json`: for each line, its `text_id` and the `wnum` of each word (e.g. line 1 → text 1, wnums 1-10) |
| | GPT-2 (124M) model, downloaded by the `transformers` library |
| **Output** | `output/gpt2_surprisal_gpt2_ctx.csv`: **51,238 rows**, columns `text_id, wnum, gpt2_surprisal` |

Why 51,238 and not 50,647? The sentence file has **every** word of the text, including 591 words that no reader fixated.
They are simply ignored when step 4 joins on `(text_id, wnum)`.

### Key lines to point at

| Lines | What it does (say this) |
|---|---|
| [122-130](../phase1_replication/experiments/compute_gpt2_surprisal.py#L122-L130) | Command-line options: `--model`, `--limit`, `--out`, `--context-sentences`, `--context-tokens`. |
| [137-139](../phase1_replication/experiments/compute_gpt2_surprisal.py#L137-L139) | Loads the GPT-2 tokenizer and model. `model.eval()` = prediction mode only, no training. |
| [50-63](../phase1_replication/experiments/compute_gpt2_surprisal.py#L50-L63) `build_context()` | Takes up to **3 previous sentences of the same text** (max 300 tokens) as context. This is our improvement: GPT-2 reads like a human, with what came before. |
| [81-85](../phase1_replication/experiments/compute_gpt2_surprisal.py#L81-L85) | Puts a start token (BOS) in front and runs GPT-2 once on the whole text. |
| [87-91](../phase1_replication/experiments/compute_gpt2_surprisal.py#L87-L91) | **The surprisal formula:** `log_softmax` gives log-probabilities, we take the one for the real next token, and surprisal = **−log P(token \| all previous tokens)**. |
| [96-116](../phase1_replication/experiments/compute_gpt2_surprisal.py#L96-L116) | GPT-2 splits some words into pieces (e.g. "enticed" → "ent" + "iced"). We **add the pieces' surprisal** to get one number per word. Context tokens are skipped. |
| [109-113](../phase1_replication/experiments/compute_gpt2_surprisal.py#L109-L113) | **Bug I found and fixed:** GPT-2 glues the space onto the next word (" the"), so the first word of a sentence was wrongly getting 0. Now we check where a token **ends**, not where it starts. |
| [156-158](../phase1_replication/experiments/compute_gpt2_surprisal.py#L156-L158) | Safety check: the number of scores must equal the number of words. |
| [167-170](../phase1_replication/experiments/compute_gpt2_surprisal.py#L167-L170) | **Saves** the CSV. |

### Real values (sentence 1)

| wnum | word | gpt2_surprisal | lex_surprisal (old, BNC bigram) |
|---|---|---|---|
| 1 | Are | 8.72 | 0.00 (first word) |
| 2 | tourists | 11.68 | 11.65 |
| 3 | enticed | 10.12 | 12.73 |
| 4 | **by** | **0.90** | **10.64** |
| 5 | these | 6.59 | 6.84 |
| 6 | attractions | 4.82 | 11.65 |

"attractions" after *"enticed by these"* is expected (GPT-2: 4.82), but the bigram method only sees "these" and calls it surprising (11.65).

---

## 8. PHASE 2, part 2: `experiments/train_system1_gpt2.py`

**Run:** `python experiments/train_system1_gpt2.py --surprisal-file output/gpt2_surprisal_gpt2_ctx.csv` (seconds)

| | |
|---|---|
| **Input** | everything step 4 uses + `output/gpt2_surprisal_gpt2_ctx.csv` |
| **Output** | **Nothing saved**; the comparison table is printed on screen |

### Key lines to point at

| Lines | What it does (say this) |
|---|---|
| [23-24](../phase1_replication/experiments/train_system1_gpt2.py#L23-L24) | `import step4_system1 as s1`: **reuses the Phase 1 code exactly**, so the comparison is fair. Same data, same split, same Ridge, same threshold. |
| [27-31](../phase1_replication/experiments/train_system1_gpt2.py#L27-L31) `add_gpt2_feature()` | Joins the GPT-2 column on `(text_id, wnum)`. |
| [50-53](../phase1_replication/experiments/train_system1_gpt2.py#L50-L53) | Loads data and splits by sentence with Phase 1's own functions. |
| [55-57](../phase1_replication/experiments/train_system1_gpt2.py#L55-L57) | **The 3 versions:** `baseline` = Phase 1 features · `gpt2_added` = baseline + GPT-2 · `gpt2_swap` = parser surprisal **replaced** by GPT-2 ([line 57](../phase1_replication/experiments/train_system1_gpt2.py#L57)). |
| [67-73](../phase1_replication/experiments/train_system1_gpt2.py#L67-L73) | Trains all 3 versions for all 4 measures and prints test R². |
| [76-80](../phase1_replication/experiments/train_system1_gpt2.py#L76-L80) | Prints the correlation **r** of each surprisal with reading time. |

### Result (on screen)
```
Measure  baseline  gpt2_added  gpt2_swap      Paper
FFD      0.444     0.458       0.458          0.649
FPD      0.558     0.583       0.583          0.600
RPD      0.555     0.582       0.582          0.570  ← beats paper
TD       0.536     0.568       0.568          0.516  ← beats paper

r(syn_surprisal, ffd)  = -0.017   ← parser: no link
r(gpt2_surprisal, ffd) =  0.465   ← GPT-2: strong link
```

---

## 9. Every feature in one table (the "where is X?" cheat sheet)

| Feature | Made in | Line | Saved in (CSV column) | "tourists" |
|---|---|---|---|---|
| FFD, FPD, RPD, TD | step1 | [83-113](../phase1_replication/src/step1_parse_dundee.py#L83-L113) | `dundee_rt.csv` → `ffd fpd rpd td` | 173.6 / 261.0 / 261.0 / 312.8 ms (avg) |
| word length | step1 | [143](../phase1_replication/src/step1_parse_dundee.py#L143) | `dundee_rt.csv` → `wlen` | 8 |
| sentence length | step2 | [213-216](../phase1_replication/src/step2_lexical_features.py#L213-L216) | `lexical_features.csv` → `sent_len` | 10 |
| word frequency | step2 | [39](../phase1_replication/src/step2_lexical_features.py#L39) | `lexical_features.csv` → `wiki_freq` | −11.37 |
| age of acquisition | step2 | [60](../phase1_replication/src/step2_lexical_features.py#L60), [70](../phase1_replication/src/step2_lexical_features.py#L70) | `lexical_features.csv` → `aoa_mean`, `aoa_std` | 8.76, 0.0 |
| forward / backward prob | step2 | [193](../phase1_replication/src/step2_lexical_features.py#L193), [203](../phase1_replication/src/step2_lexical_features.py#L203) | `lexical_features.csv` → `fwd_prob`, `bwd_prob` | −11.65, −12.72 |
| lexical surprisal | step3 | [108-109](../phase1_replication/src/step3_syntactic_features.py#L108-L109) | `syntactic_features.csv` → `lex_surprisal` | 11.65 |
| syntactic surprisal | parser → step3 | [346](../phase1_replication/src/step3_syntactic_features.py#L346) | `syntactic_features.csv` → `syn_surprisal` | 6.74 |
| entropy reduction | step3 | [300](../phase1_replication/src/step3_syntactic_features.py#L300) | `syntactic_features.csv` → `entropy_red` | 0.010 |
| embedding depth / diff | step3 | [304-307](../phase1_replication/src/step3_syntactic_features.py#L304-L307) | `syntactic_features.csv` → `emb_depth`, `emb_diff` | 1, 0 |
| tree flags | step3 | [229-236](../phase1_replication/src/step3_syntactic_features.py#L229-L236) | `syntactic_features.csv` → `h1`–`h8` | h8 = 1, others 0 |
| **GPT-2 surprisal** (Phase 2) | compute_gpt2 | [90-91](../phase1_replication/experiments/compute_gpt2_surprisal.py#L90-L91) | `gpt2_surprisal_gpt2_ctx.csv` → `gpt2_surprisal` | 11.68 |
| predicted FFD | step4 | [308](../phase1_replication/src/step4_system1.py#L308) | `system1_predictions.csv` → `predicted_ffd` | 174.7 |

**To show any value live:** open the CSV in VS Code, press Ctrl+F, and search `1,2,` (text 1, word 2 = "tourists").

---

## 10. Questions the professor may ask

**"Why split by sentence, not by word?"**
Words in one sentence affect each other. If half a sentence is in training and half in testing, the test is too easy.
→ [step4 lines 120-143](../phase1_replication/src/step4_system1.py#L120-L143).

**"Why average over the 10 readers?"**
The paper pools readers. One reader is noisy (for "tourists", FFD ranges from 108 to 270 ms). The average is stable.
→ [step4 lines 77-82](../phase1_replication/src/step4_system1.py#L77-L82).

**"Why Ridge and not plain linear regression?"**
The h-flags overlap (h1 = h2 or h3), and Ridge handles that. Alpha is tuned on the dev set.
→ [step4 lines 236-244](../phase1_replication/src/step4_system1.py#L236-L244).

**"What is the threshold?"**
The paper's way of modelling skipped words: a prediction under T ms is set to 0. FFD's best T is 56 ms.
→ [step4 lines 146-173](../phase1_replication/src/step4_system1.py#L146-L173).

**"Why is each Wikipedia pair used twice?"**
So the classifier learns "which is harder", not "the first one is always harder". Half the rows have label 1, half label 0.
→ [step5 lines 195-202](../phase1_replication/src/step5_system2.py#L195-L202).

**"Does System 2 use all the features?"**
No. For Wikipedia sentences it uses the word features (length, frequency, AoA, sentence length) and sets the parser features to 0, because running the parser on 235K sentences ran out of memory. It still beats the paper.
→ [step5 lines 109-129](../phase1_replication/src/step5_system2.py#L109-L129), [284-285](../phase1_replication/src/step5_system2.py#L284-L285).

**"How did you make sure Phase 2 is a fair comparison?"**
It imports the Phase 1 code instead of copying it. Same split, same model, same threshold; only one column changes.
→ [train_system1_gpt2 lines 23-24, 57](../phase1_replication/experiments/train_system1_gpt2.py#L23-L24).

**"What is surprisal?"**
−log of the probability of the word, given what came before. A likely word gives a small number ("by" = 0.90), and an unexpected word gives a big number. People read surprising words more slowly.

**"Is the result reproducible?"**
Yes. I reran all steps on 2026-10-20 and every output file was byte-for-byte identical. The split uses `seed=42` ([step4 line 120](../phase1_replication/src/step4_system1.py#L120)) and CV uses `random_state=42` ([step5 line 304](../phase1_replication/src/step5_system2.py#L304)).

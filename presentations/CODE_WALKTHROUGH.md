# Code Walkthrough: what I did and how the code works

This is my guide for explaining the project folder. For every script it gives:

- **What it does**, in one line
- **Input → Output** files
- **Real example** values from my output files
- **Say this:** what to say out loud

The rule for every script is the same: **read a file → do one job → write a file.** The next step reads that file. This makes each step easy to check on its own.

---

## The big picture

```
PHASE 1 (English)                         PHASE 2                PHASE 3 (Tamil)
─────────────────                         ───────                ───────────────
Dundee eye-tracking                                              TN textbooks + StoryWeaver
   │ step1  reading times per word                                  │ step0-2, step4  sentences + pairs
   │ step2  word features                                           │ step3, step5    12 features per sentence
   │ step3  syntax features   ◄── GPT-2 surprisal replaces ──┐      │ step6           which features work?
   │ step4  System 1 (reading time)    parser surprisal      │      │ metric_model    word score vs morph score
   │ step5  System 2 (which is harder?)                      │      │ step7-8         LLM simplification
   ▼                                                         │      │ step9           human study
 toolkit/app.py  (live demo)            experiments/compute_gpt2_…  │ step10          Gemini judge
```

---

# PHASE 1: Replication (`phase1_replication/src/`)

### `step1_parse_dundee.py`: get reading times from raw eye-tracking

**What it does:** reads the raw Dundee eye-tracker files and computes 4 reading times for every word, for every reader.

**Input → Output:** `Dataset/dundee_corpus/` → `output/dundee_rt.csv` (307,383 rows = words × readers)

The 4 measures:

| Measure | Meaning in simple words |
|---|---|
| FFD | first look at the word |
| FPD | all looks before the eyes move on |
| RPD | time from first look until the eyes move past the word (includes going back) |
| TD | total time on the word, including coming back later |

**Real example** (the first word of text 1, read by different readers):

| word | reader | wlen | ffd (ms) |
|---|---|---|---|
| Are | 0 | 3 | 216 |
| Are | 1 | 3 | 102 |
| Are | 2 | 3 | 150 |

**Say this:** "The raw data is one row per eye fixation. This script groups fixations by word and computes the four reading times the paper uses. Readers differ a lot: the same word took 102 ms for one person and 216 ms for another. So later I average over the 10 readers."

---

### `step2_lexical_features.py`: word-level features

**What it does:** gives every word 7 simple features.

**Input → Output:** `dundee_rt.csv` + AoA list + Wikipedia frequency + BNC corpus → `output/lexical_features.csv`

**Real example** (sentence: *"Are tourists enticed by these attractions threatening…"*):

| word | length | frequency (log) | age learned (years) |
|---|---|---|---|
| Are | 3 | −5.2 (common) | 5.1 |
| enticed | 7 | −14.4 (rare) | 12.5 |
| by | 2 | −5.4 (common) | 5.4 |
| attractions | 11 | −11.7 | 9.6 |

**Say this:** "Each word gets its length, how common it is, the age at which people usually learn it, and how likely it is after or before its neighbour word, using the British National Corpus. Look at 'enticed': it's rare and learned late, at 12.5 years. We expect people to read it slowly."

---

### `step3_syntactic_features.py`: sentence-structure features

**What it does:** adds surprisal and sentence-structure features for every word.

**Input → Output:** BNC bigrams + LC-parser output + dependency parse → `output/syntactic_features.csv`

- `lex_surprisal`: how unexpected the word is after the previous word
- `syn_surprisal`: how unexpected the grammar structure is, from a real parser (the paper's method)
- `entropy_red`, `emb_depth`, `emb_diff`, `h1`–`h8`: how deep in the sentence structure the word sits

**One worked example for each feature:** see [Appendix A](#appendix-a-every-english-feature-with-one-example) at the end.

**Say this:** "These are the paper's syntax features. The hardest one was syntactic surprisal. I had to run the original left-corner parser inside Linux (`experiments/build_lcparse_surprisal.py`) on all 50,647 words."

---

### `step4_system1.py`: System 1, predicting reading time

**What it does:** trains a linear regression model (Ridge) that predicts reading time from the features above.

**Input → Output:** `lexical_features.csv` + `syntactic_features.csv` → `output/system1_models.pkl`, `output/system1_predictions.csv`

How it works, step by step:

1. Average the reading times of the 10 readers, giving one row per word (50,647 words).
2. Split by **sentence**: 60% train, 20% dev, 20% test. A sentence never appears in both train and test.
3. Scale all features to the same range, then train one model per measure (FFD, FPD, RPD, TD).
4. **Ablation:** add the features one at a time and see how much each adds (the paper's Table 2).
5. **Threshold:** predictions below a cut-off (55 ms for FFD) are set to 0, like the paper does. The cut-off is tuned on dev.

**Result:** test R² FFD 0.444, FPD 0.558, RPD 0.555, TD 0.536 (paper: 0.649, 0.600, 0.570, 0.516).

**Say this:** "R² tells how much of the reading-time differences the model explains. 1 would be perfect. We are close to the paper on three measures and beat it on TD. Only FFD is lower."

---

### `step5_system2.py`: System 2, which sentence is harder?

**What it does:** for two sentences, predicts which one is harder, using System 1's predicted reading times.

**Input → Output:** 117,712 Wikipedia / Simple Wikipedia sentence pairs + `system1_models.pkl` → accuracy numbers

How it works:

1. For each sentence, System 1 predicts the FFD of every word (up to 60 words).
2. The pair becomes one row: [times of sentence A] + [times of sentence B].
3. **Extended model:** also adds sentence length, average time and total surprisal.
4. Logistic regression and LinearSVC, tested with **10-fold cross-validation**.

**Result:** 74.15% (base) and **75.26%** (extended). The paper has 73.82% and 75.21%, so we beat it on all 4.

**Say this:** "Normal Wikipedia is harder than Simple Wikipedia, so the label comes for free. The model guesses which one is the normal-Wikipedia sentence. 50% would be guessing; we get 75%, the same as the paper."

---

### Helper files

- `syntactic_extractor.py`: computes the same features for **any new sentence** (not only Dundee). The toolkit and System 2 use it.
- `feature_demo.py`: prints, step by step, how all 17 features are computed from the raw data. **Good to run live if the professor asks "show me how one feature is computed".**

---

### FFD-gap experiments (`phase1_replication/experiments/`)

| Script | What I tested | Result |
|---|---|---|
| `build_lcparse_surprisal.py` | Is our syntax feature computed wrongly? Ran the real parser. | No. r = −0.017, the same as the paper's own table. |
| `fixated_only_system1.py` | Are skipped words (time = 0) inflating word length? | No. After averaging, no word has time 0. |
| `check_wlen_r2_consistency.py` | Are the paper's Table 1 and Table 2 consistent? | No. r = 0.765 means R² ≈ 0.585, but Table 2 says 0.267. |
| `compute_pcfg_surprisal_python.py` | An earlier attempt to compute syntax surprisal in Python | Replaced by the real parser |

**Say this:** "I did not just accept the gap. I tested three explanations with separate scripts. The last one shows the paper's own two tables contradict each other."

---

# PHASE 2: GPT-2 (`phase1_replication/experiments/`)

### `compute_gpt2_surprisal.py`

**What it does:** asks GPT-2 how unexpected each Dundee word is, given the words before it.

**Input → Output:** `output/dundee_sentences.txt` → `output/gpt2_surprisal_gpt2_ctx.csv` (51,238 rows)

**Real example:**

| word | GPT-2 surprisal |
|---|---|
| Are | 8.7 |
| tourists | 11.7 |
| enticed | 10.1 |
| **by** | **0.9**: after "enticed", "by" is almost certain |
| threatening | 12.0 |

How it works: GPT-2 splits words into small pieces (tokens). The script adds up the surprisal of all pieces of a word. It also gives GPT-2 **the 3 previous sentences** as context, which worked best.

**Bug I found and fixed:** the first word of each sentence was getting surprisal 0. GPT-2 attaches the space to the next word, so the code was checking the wrong position. I fixed the check from the token's start to its end.

### `train_system1_gpt2.py`

**What it does:** re-trains System 1 three ways: without GPT-2 (baseline), with GPT-2 added, and with GPT-2 **replacing** the parser surprisal. It reuses `step4_system1.py` as it is; nothing is copied.

**Result:** FFD 0.444 → 0.458, FPD 0.558 → 0.583, RPD → **0.582**, TD → **0.568** (both beat the paper).

### `gpt2_sentence_correlation.py`

**What it does:** checks that GPT-2 is new information, not a copy of an existing feature. Its correlation with syntax surprisal is only 0.014, so it is new.
**Warning I found:** summing GPT-2 over a sentence just measures sentence length (r = 0.91), so I use the average instead.

**Say this:** "GPT-2 is a language model, so it knows which words are expected. The old parser only knows grammar rules. That's why GPT-2 tracks reading time much better: r = 0.465 against almost zero."

---

# PHASE 3: Tamil (`phase3_tamil_extension/src/`)

### Building the data: `step0`, `step1`, `step2`, `step4`

| Script | What it does | Output |
|---|---|---|
| `step0_fetch_textbooks.py` | Downloads TN Tamil textbooks for Class 3–12. Checks page 1 says the right class (e.g. "ஆறாம் வகுப்பு"). | `Dataset/graded_textbooks/` |
| `tamil_text.py` | Repairs broken PDF text, e.g. `அடித்்தளம்` → `அடித்தளம்`. Real words went from 55% to 76.5%. | helper |
| `step1_extract_sentences.py` | Keeps only **prose** sections (உரைநடை உலகம்). Skips poetry, grammar and exercises. Removes broken lines. | `output/graded_sentences.csv` (2,351) |
| `step2_build_pairs.py` | Makes pairs at least 3 classes apart. Splits train/test **by page**. | `output/graded_pairs.csv` (8,004) |
| `step4_storyweaver_data.py` | Downloads StoryWeaver book descriptions with reading level 1–4 | `output/storyweaver_sentences.csv` |

**Say this:** "No Tamil readability dataset existed, so I built two. The textbook PDFs had broken fonts, so I wrote repair rules. I kept only prose, because poetry uses old Tamil and would confuse the test. I split by page, so the model is never tested on a passage it has already seen."

**If asked "why I didn't remove rare words":** "Rare words are exactly what makes text hard. Removing them would bias the test. So I remove broken text by its symptoms, not by frequency."

---

### `step3_textbook_features.py` and `step5_storyweaver_features.py`: 12 features per sentence

| Group | Features | Tool |
|---|---|---|
| **A · Word-level** (English-style baseline) | number of words, word length, word frequency, GPT-2 surprisal | wordfreq, `abinayam/gpt-2-tamil` |
| **B · Morphology** (my new idea) | morphemes per word, stem surprisal, suffix surprisal, grammar features per word, lemma frequency, non-finite verbs | IndicNLP Morfessor, Stanza |
| **C · Syntax** | tree depth, dependency distance | Stanza parser |

**Real data:** the average for each StoryWeaver level (7,080 sentences):

| Level | words | word length | morphemes/word | non-finite verbs |
|---|---|---|---|---|
| 1 (easiest) | 6.8 | 7.4 | 1.38 | 1.2 |
| 4 (hardest) | 8.6 | 8.0 | 1.46 | 1.4 |

**One worked example for each feature:** see [Appendix B](#appendix-b-every-tamil-feature-with-one-example) at the end.

**Say this:** "As the level goes up, sentences get longer, words get longer, and words have more parts. But look: word length and morphemes per word rise **together**. That's the first hint of my final result."

---

### `step6_evaluate_metric.py`: which features can pick the harder sentence?

**What it does:** the same pair idea as System 2. The model sees features(B) − features(A) and guesses which is harder. It tests groups A, B, C and their combinations.

**Output:** `results/summary.md`, `results/storyweaver/summary.md`, `fig_gap.png`

**Result (StoryWeaver):** A = **63.8%**, B = 60.5%, A + B = 63.4%. On same-length pairs, B is best at 59.7%.

**Detail worth mentioning:** confidence intervals come from resampling **sentences**, not pairs. One sentence appears in many pairs, so resampling pairs would make the result look more certain than it really is.

---

### `metric_model.py`: the two metrics as one score

**What it does:** turns each feature group into **one difficulty number per sentence** (higher = harder). It is trained on StoryWeaver.

- **word metric** = group A
- **morph metric** = group A + group B

**Real weights** (morph metric): morphemes per word **+0.89** (the biggest: more parts = harder), word frequency −0.19 (common words = easier).

---

### `step7_select_complex.py` → `step8_score_simplified.py`: simplification

- **step 7** picks 60 hard sentences **at random** (30 textbook Class 11–12, 30 StoryWeaver level 4). They are not chosen by my metric, so the test doesn't favour either metric. It writes ready-to-paste prompts for a free LLM chat.
- **step 8** reads the LLM answers, scores the original and simplified versions with both metrics, and **creates the Google Forms automatically** (`make_google_form.gs`). It randomises which side (A or B) is the original.

**Real example (S13, StoryWeaver):**

> **Original:** ஒரு பெரிய காட்டில், தொலைவிலுள்ள பகுதியில், ஏராளமான ஏரிகள், குளங்கள் மற்றும் ஓடைகள் இருந்தன.
> **Simplified:** ஒரு பெரிய காடு இருந்தது. அதன் ஒரு தூரமான பகுதியில் நிறைய ஏரிகளும் குளங்களும் ஓடைகளும் இருந்தன.

| | Original | Simplified |
|---|---|---|
| words per sentence | 11 | 6.5 |
| morphemes per word | 1.45 | 1.28 |
| word score | +0.11 | **−0.64** (easier) |
| morph score | +0.60 | **−0.24** (easier) |

**Say this:** "The LLM split one long sentence into two short ones. Both metrics' scores drop, so both say it got easier. When an LLM splits a sentence, I average over the sentences; otherwise splitting would look harder just because there are more words."

---

### `step9_human_eval.py`: the human study

**What it does:** reads the Google Form answers (`Dataset/human_study/responses/form1.csv`, `form2.csv`), keeps only native or fluent speakers, and compares the humans with both metrics.

**Output:** `results/human_eval_summary.md`

| Question | Result |
|---|---|
| Do readers find the simplified version easier? | Yes: picked 49% vs 29%; difficulty 2.31 → 1.97 |
| Which metric agrees with readers? | word ρ = +0.32, morph ρ = +0.28 |
| Is morph better? | No: the difference is −0.04, with a 95% CI from −0.11 to +0.02 |

**Say this:** "Each pair was rated by about 6 people. I compare each metric's score with the average human difficulty. Both agree, but morphology is not better. That's the same answer as step 6, from different evidence."

---

### `step10_llm_judge.py`: Gemini as an extra judge

**What it does:** sends Gemini the **exact same questions** the humans saw, through the free API. It caches answers, so re-running doesn't waste the daily limit. It reports Gemini **separately**; Gemini is never counted as a human.

**Result:** Gemini vs the average human is +0.30; one human vs the others is +0.11. But Gemini says 90% of simplifications are easier (humans: 49%).

---

### `experiments/`: early pilot tests

These are small scripts from the start of Phase 3 that I used to **choose the tools**. For example, `eval_stanza_ud.py` showed Stanza gets 94.6% accuracy on Tamil case; `pilot_morph_coverage.py` showed ThamizhiMorph covered only 42% of words, so I didn't use it.

**Say this:** "Before building the pipeline, I tested the available Tamil tools and picked the ones that worked best."

---

# TOOLKIT (`toolkit/app.py`)

**What it does:** a web app (Gradio). You type two English sentences and it says which is harder, with a confidence score.

How it works:

1. Loads the saved System 1 and System 2 models (`models/*.pkl`).
2. For each word: length, frequency (wordfreq) and age learned (AoA). System 1 predicts its reading time.
3. System 2 takes both sentences' reading times and gives the answer.

**Be honest if asked:** the demo uses only the simple features (length, frequency, age learned, sentence length). The corpus-based ones (BNC probabilities, parser surprisal) are set to 0, because they need the large datasets, which can't run in a live app.

**Run:** `cd toolkit` then `python app.py`, and open `http://127.0.0.1:7860`.

**Planned for the final evaluation:** a Phase 2 tab (reading time and GPT-2 surprisal per word) and a Phase 3 tab (Tamil sentence → difficulty score, word parts, simplified version).

---

# Questions about the code

| Question | Answer |
|---|---|
| Did you write this code yourself? | "I designed every step and wrote it with AI coding help. I can explain every step and its output." *(Say this honestly.)* |
| Why so many separate scripts? | "Each step does one job and saves a file. If something looks wrong, I can check that one file without rerunning everything." |
| How long does it take to run? | Phase 1 BNC step: 20–40 min the first time. Phase 3 features: ~40 min (textbooks), ~1.5 h (StoryWeaver) on CPU. Feature files are saved, so step 6 onwards runs in minutes. |
| Why split by sentence / page / book? | "So the model is never tested on text it has already seen. Otherwise the accuracy would look better than it really is." |
| What is a bootstrap confidence interval? | "I resample the data 1,000–2,000 times and see how much the result changes. That range is the 95% interval." |
| Where are the final numbers? | Phase 1: `phase1_replication/results/summary.md`. Phase 3: `phase3_tamil_extension/results/*.md` |

---

# Appendix A: every English feature with one example

**Example sentence (Dundee, text 1, first sentence):**
*"Are tourists enticed by these attractions threatening their very …"*

All values below come from `phase1_replication/output/lexical_features.csv`, `syntactic_features.csv` and `gpt2_surprisal_gpt2_ctx.csv`. To print the same kind of example live, run `python src/feature_demo.py`.

### Group 1: simple word features (`step2_lexical_features.py`)

| # | Feature | What it means | How the code computes it | Example | Why it matters |
|---|---|---|---|---|---|
| 1 | `wlen` | word length | count the letters | "by" = **2**, "attractions" = **11** | Long words take longer to read |
| 2 | `sent_len` | sentence length | count the words in the sentence | every word in this sentence = **10** | Long sentences are harder to keep in memory |
| 3 | `wiki_freq` | how common the word is | `log(word_frequency)` from the wordfreq list (Wikipedia and more) | "tourists": frequency 0.0000115 → log = **−11.37**. "Are" = **−5.20** | Higher (closer to 0) = more common = read faster |
| 4 | `aoa_mean` | age when people learn the word | look up the word in the AoA list (Kuperman) | "by" = **5.4 years**, "enticed" = **12.5 years** | Words learned late are read more slowly |
| 5 | `aoa_std` | how much the AoA lists disagree | standard deviation of the different AoA lists | "by": lists give 5.41, 4.51, 4.27 → **0.49** | High = the word's learning age is uncertain |
| 6 | `fwd_prob` | how likely this word is after the previous word | `log((count(prev, word) + 1) / (count(prev) + vocabulary))` from the BNC (100 million words) | "these" after "by" = **−6.84** (likely). "enticed" after "tourists" = **−12.73** (unlikely) | Expected words are read faster |
| 7 | `bwd_prob` | how likely this word is before the next word | the same formula, looking at the next word | "by" before "these" = **−8.82** | Shows how well the word fits with what follows |

**The +1 in the formula (smoothing):** if a word pair never appears in the BNC, its probability would be 0 and the log would break. Adding 1 to every count fixes that.

### Group 2: surprisal and structure (`step3_syntactic_features.py`)

| # | Feature | What it means | How the code computes it | Example | Why it matters |
|---|---|---|---|---|---|
| 8 | `lex_surprisal` | how surprising the word is after the previous word | `−log P(word ∣ previous word)` from BNC word pairs | "by" = **10.64**, "these" = **6.84**. First word "Are" = **0** (no previous word) | Surprising words slow the reader down |
| 9 | `syn_surprisal` | how surprising the grammar structure is | from the real left-corner parser (the paper's method), trained on Penn Treebank | "Are" = **35.83** (a question start is rare in news), "by" = **0.20** (a preposition here is expected) | The paper's main syntax idea. In our data it had almost no effect (r = −0.017) |
| 10 | `total_surprisal` | both surprisals together | `lex_surprisal + syn_surprisal` | "tourists": 11.65 + 6.74 = **18.38** | One combined "unexpectedness" number |
| 11 | `entropy_red` | how much the uncertainty about what comes next drops at this word | `max(0, uncertainty before − uncertainty after)`, using part-of-speech pairs from Penn Treebank | "their" = **1.11** (after "their" a noun must follow, so uncertainty drops). "attractions" = **0** | When uncertainty drops, the brain does work, which costs time |
| 12 | `emb_depth` | how deep the word sits in the sentence tree | steps from the word up to the main verb in the dependency parse | "enticed" = **0** (the main verb), "by" = **2**, "their" = **4** | Deeper words are harder to keep in memory |
| 13 | `emb_diff` | change in depth from the previous word | `depth(this word) − depth(previous word)` | "by": 2 − 0 = **+2**. "enticed": 0 − 1 = **−1** | Big jumps mean the reader must open or close a phrase |

### Group 3: tree movement flags `h1`–`h8` (`step3_syntactic_features.py`)

These are yes/no (1/0) flags. Each one describes how the depth changed from the previous word to this word.

| Flag | Is 1 when… | Example word | Value |
|---|---|---|---|
| `h1` | depth changed | "enticed" (1 → 0) | **1** |
| `h2` | depth went up | "by" (0 → 2) | **1** |
| `h3` | depth went down | "enticed" (1 → 0) | **1** |
| `h4` | the word is the main verb (depth 0) | "enticed" | **1** |
| `h5` | the word has no dependents | always 0 in our code (needs child lists, not computed) | **0** |
| `h6` | depth jumped up by more than 1 | "by" (0 → 2) | **1** |
| `h7` | depth jumped down by more than 1 | none in this sentence | **0** |
| `h8` | depth stayed the same | "tourists" (1 → 1), "these" (2 → 2) | **1** |

**Be honest if asked:** "h1–h8 are my approximation of the paper's hierarchical features, built from the dependency tree. The paper's exact definitions come from its parser. h5 is always 0 in my version."

### Phase 2 feature: GPT-2 surprisal (`experiments/compute_gpt2_surprisal.py`)

| Feature | What it means | How the code computes it | Example | Why it matters |
|---|---|---|---|---|
| `gpt2_surprisal` | how surprising the word is, given **all** previous words (and 3 earlier sentences) | GPT-2 gives a probability for each token; sum `−log P` over the word's tokens | "by" after "tourists enticed" = **0.90** (almost certain). "threatening" = **12.04** (very unexpected) | Uses the whole context, not just the one previous word. Correlates 0.465 with reading time |

**Compare lex vs GPT-2 for "by":** BNC word pairs say **10.64** (only looks at "enticed → by"). GPT-2 says **0.90**, because it knows "enticed by" is a common phrase in context. That's why GPT-2 tracks reading time better.

---

# Appendix B: every Tamil feature with one example

**Example sentence (StoryWeaver, level 4, item S13):**
> ஒரு பெரிய காட்டில், தொலைவிலுள்ள பகுதியில், ஏராளமான ஏரிகள், குளங்கள் மற்றும் ஓடைகள் இருந்தன.
> *"In a big forest, in a far-away part, there were many lakes, ponds and streams."*

I ran the real pipeline (`step3_textbook_features.py`) on this sentence. Its sentence-level values match `output/simplification_scores.csv` exactly.

### Word by word: what the tools see

| Word | Letters | Frequency (Zipf) | GPT-2 surprisal | Morfessor split | Stanza: lemma + grammar |
|---|---|---|---|---|---|
| ஒரு | 3 | 7.01 | 17.64 | ஒரு | ஒரு |
| பெரிய | 5 | 5.98 | 7.91 | பெரிய | பெரிய |
| காட்டில் | 8 | 4.51 | 11.55 | காட்டில் (not split) | காடு + **Case=Loc** ("in") |
| தொலைவிலுள்ள | 11 | 3.56 | 18.63 | **தொலை + விலுள்ள** | தொலை + **Loc**, உள் + **VerbForm=Part** |
| பகுதியில் | 9 | 5.77 | 10.95 | பகுதியில் | பகுதி + **Case=Loc** |
| ஏராளமான | 7 | 5.00 | 11.53 | ஏராளமான | ஏராளம் |
| ஏரிகள் | 6 | 4.03 | 9.54 | **ஏரி + கள்** | ஏரி + **Number=Plur** |
| குளங்கள் | 8 | 4.06 | 3.56 | **குளங் + கள்** | குளம் + **Number=Plur** |
| மற்றும் | 7 | 6.68 | 3.58 | மற்றும் | மற்றும் |
| ஓடைகள் | 6 | 3.20 | 10.95 | **ஓடை + கள்** | ஓடை + **Number=Plur** |
| இருந்தன | 7 | 5.20 | 5.32 | **இருந்த + ன** | இரு + **Number=Plur, Tense=Past** |

### Group A: word-level (the English-style baseline)

| Feature | What it means | How the code computes it | This sentence | Why it matters |
|---|---|---|---|---|
| `n_words` | number of words | count the words | **11** | Longer sentences are harder |
| `word_len` | average letters per word | average of the "Letters" column | 77 ÷ 11 = **7.0** | Long words are harder (the English idea) |
| `word_zipf` | average word frequency | wordfreq Zipf scale: 7 = very common, 3 = rare | **5.0**. Rarest word: ஓடைகள் = 3.20 | Rare words are harder |
| `word_surprisal` | average GPT-2 surprisal per word | Tamil GPT-2 (`abinayam/gpt-2-tamil`), in bits | **10.10**. Most surprising: தொலைவிலுள்ள = 18.63 | Unexpected words are harder |

### Group B: morphology (my new idea)

| Feature | What it means | How the code computes it | This sentence | Why it matters |
|---|---|---|---|---|
| `morphs_per_word` | average number of word parts | IndicNLP Morfessor splits each word | 16 parts ÷ 11 words = **1.45** (e.g. ஏரி + கள் = 2) | More parts = more to decode (the main hypothesis) |
| `stem_surprisal` | how surprising the **first part** (the stem) is | GPT-2 surprisal of the stem's letters only | **9.23**. தொலை = 11.91 | Separates "is the root word hard?"… |
| `suffix_surprisal` | how surprising the **endings** are | GPT-2 surprisal of the rest of the word | **0.88**. விலுள்ள = 6.71, கள் = 0.38 | …from "are the endings hard?". Common endings like கள் cost almost nothing |
| `feats_per_word` | grammar markings per word | Stanza grammar tags, counting only *marked* ones (skip defaults like Case=Nom, Singular) | 9 markings ÷ 12 Stanza words = **0.75** (3 × Loc, 4 × Plur, 1 × Part, 1 × Past) | Each marking is extra grammar to process |
| `lemma_zipf` | how common the **dictionary form** is | Zipf frequency of Stanza's lemma | **5.13** (higher than `word_zipf` 5.0: காடு is more common than காட்டில்) | A word can be rare only because of its ending; the lemma shows if the root itself is rare |
| `nonfinite_verbs` | verb forms that join clauses (participles, "-ing" forms, infinitives) | count words with VerbForm = Part / Conv / Inf / Ger / Vnoun | **1**: உள்ள ("which is") inside தொலைவிலுள்ள | This is how Tamil packs several clauses into one sentence |

### Group C: syntax

| Feature | What it means | How the code computes it | This sentence | Why it matters |
|---|---|---|---|---|
| `tree_depth` | the longest chain in the sentence tree | from each word, follow "head" links up to the main verb | **4**: தொலைவில் → உள்ள → பகுதியில் → இருந்தன → top | Deep trees are harder to hold in memory |
| `dep_distance` | average distance between a word and the word it attaches to | position difference between a word and its head, averaged | **3.18**. காட்டில் (word 3) attaches to இருந்தன (word 15): distance 12 | Long distances mean the reader must wait longer to connect words |

### What to say about this example

> "Look at தொலைவிலுள்ள. One Tamil word holds 'far' + 'in' + 'which is'. It has 11 letters, 2 parts, a participle, and the highest surprisal in the sentence. That's exactly the kind of word my morphology features were built to catch."

**Be honest if asked about tool errors:** "The tools are not perfect. Here Morfessor did not split காட்டில் (காடு + இல்), but Stanza did find its Locative case. That's one reason I use **both** tools: Morfessor for the split, Stanza for the grammar."

### Why morphology still didn't beat word length

In this sentence, the long words are the ones with many parts: தொலைவிலுள்ள has 11 letters and 2 parts, while ஒரு has 3 letters and 1 part. Across 7,080 StoryWeaver sentences, this pattern holds, so **word length already carries most of the morphology information**. That's the main finding.

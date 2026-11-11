# Phase 1 + 2: Every Feature Calculated by Hand

This file takes **one word, "tourists"**, and calculates **every feature from the raw data, by hand**,
then shows the **exact value stored in the output CSV**. I checked every number on 2026-10-20:
the hand calculation gives exactly the stored value.

**The sentence** (Dundee text 1, sentence 10000):

| wnum | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| word | Are | **tourists** | enticed | by | these | attractions | threatening | their | very | existence? |

**How to find any value yourself:** open the CSV in VS Code, press **Ctrl+F**, and type `1,2,` (text 1, word 2).

Code paths are in `phase1_replication/`. Line links open the exact line.

---

## Where everything is stored (summary)

| Feature | Calculated in | Stored in file | Column | "tourists" |
|---|---|---|---|---|
| FFD, FPD, RPD, TD (per reader) | step1 | `output/dundee_rt.csv` | `ffd fpd rpd td` | reader 3: 221 / 221 / 221 / 527 |
| FFD … TD (10-reader average) | step4 | `output/system1_predictions.csv` | `ffd fpd rpd td` | 173.6 / 261.0 / 261.0 / 312.8 |
| Word length | step1 | `output/dundee_rt.csv` (copied on) | `wlen` | 8 |
| Sentence length | step2 | `output/lexical_features.csv` | `sent_len` | 10 |
| Word frequency | step2 | `output/lexical_features.csv` | `wiki_freq` | −11.373 |
| Age of acquisition (mean) | step2 | `output/lexical_features.csv` | `aoa_mean` | 8.76 |
| Age of acquisition (spread) | step2 | `output/lexical_features.csv` | `aoa_std` | 0.0 |
| Forward probability | step2 | `output/lexical_features.csv` | `fwd_prob` | −11.646 |
| Backward probability | step2 | `output/lexical_features.csv` | `bwd_prob` | −12.722 |
| Lexical surprisal | step3 | `output/syntactic_features.csv` | `lex_surprisal` | 11.646 |
| Syntactic surprisal | parser → step3 | `output/syntactic_features.csv` | `syn_surprisal` | 6.738 |
| Total surprisal | step3 | `output/syntactic_features.csv` | `total_surprisal` | 18.384 |
| Entropy reduction | step3 | `output/syntactic_features.csv` | `entropy_red` | 0.0099 |
| Embedding depth | step3 | `output/syntactic_features.csv` | `emb_depth` | 1 |
| Embedding difference | step3 | `output/syntactic_features.csv` | `emb_diff` | 0 |
| Tree flags h1–h8 | step3 | `output/syntactic_features.csv` | `h1` … `h8` | h8 = 1, rest 0 |
| GPT-2 surprisal (Phase 2) | compute_gpt2 | `output/gpt2_surprisal_gpt2_ctx.csv` | `gpt2_surprisal` | 11.678 |
| Predicted FFD (System 1) | step4 | `output/system1_predictions.csv` | `predicted_ffd` | 174.7 |
| Trained models | step4 | `output/system1_models.pkl` | (Python pickle) | — |
| Pair vector (System 2) | step5 | not saved, built in memory | 120 / 126 numbers | — |

---

# PART 1: READING TIMES (the target the model learns)

## 1.1 The raw data

**File:** `Dataset/dundee_corpus/RT_dataset2P/sd01ma2p.dat`
(name = `s` + reader letter `d` + article `01` + `ma2p`). Reader `d` gets `subj_id` **3**,
because readers are numbered a=0, b=1, c=2, d=3 in file order ([step1 lines 205-208](../phase1_replication/src/step1_parse_dundee.py#L205-L208)).

Each line is **one fixation** (one stop of the eye). The code reads 3 columns ([step1 lines 42-44](../phase1_replication/src/step1_parse_dundee.py#L42-L44)):
`WNUM` (column 6) = which word, `FDUR` (column 7) = how long in ms, `FXNO` (column 10) = order in time.

What reader d's eyes did at the start of text 1, in time order:

| FXNO (time order) | word | WNUM | FDUR (ms) |
|---|---|---|---|
| 3 | **tourists** | 2 | **221** |
| 4 | enticed | 3 | 169 |
| 5 | these | 5 | 171 |
| 6 | these | 5 | 144 |
| 7 | **tourists** | 2 | **190** ← eye went **back** |
| 8 | **tourists** | 2 | **116** |
| 9 | by | 4 | 162 |
| 10 | attractions | 6 | 149 |

Fixations on "tourists": **(221 ms, #3), (190 ms, #7), (116 ms, #8)**.

## 1.2 FFD: First Fixation Duration

**Meaning:** how long the **very first** look at the word lasted.
**Code:** [step1 line 83](../phase1_replication/src/step1_parse_dundee.py#L83) `ffd = fdurs[0]` (fixations already sorted by time at [line 79](../phase1_replication/src/step1_parse_dundee.py#L79)).

**By hand:** first fixation = #3 → **FFD = 221 ms**

## 1.3 FPD: First Pass Duration

**Meaning:** add up looks **until the eye leaves the word for the first time**.
**Code:** [step1 lines 89-100](../phase1_replication/src/step1_parse_dundee.py#L89-L100). Start with the first look, then keep adding the next look on the same word **only if no other word was looked at in between** ([lines 94-96](../phase1_replication/src/step1_parse_dundee.py#L94-L96)). Stop at the first gap ([line 100](../phase1_replication/src/step1_parse_dundee.py#L100) `break`).

**By hand:** start = 221. Next look on "tourists" is #7, but #4, #5, #6 were other words → stop.
**FPD = 221 ms**

(Compare "these": #5 = 171 and #6 = 144 come straight after each other → FPD = 171 + 144 = **315**.)

## 1.4 RPD: Regression Path Duration

**Meaning:** add up looks on this word that happen **before the eye first moves to a word further right**.
**Code:** [step1 lines 104-111](../phase1_replication/src/step1_parse_dundee.py#L104-L111).
[Line 106](../phase1_replication/src/step1_parse_dundee.py#L106) finds fixations on a word with a **bigger wnum**, and
[line 110](../phase1_replication/src/step1_parse_dundee.py#L110) takes the earliest one. [Line 111](../phase1_replication/src/step1_parse_dundee.py#L111) sums the looks before it.

**By hand:** the first move to a word right of wnum 2 is #4 ("enticed", wnum 3). Looks on "tourists" before #4: only #3.
**RPD = 221 ms**

## 1.5 TD: Total Duration

**Meaning:** **all** looks added together, including coming back later.
**Code:** [step1 line 84](../phase1_replication/src/step1_parse_dundee.py#L84) `td = sum(fdurs)`.

**By hand:** 221 + 190 + 116 = **TD = 527 ms**

### ✅ Stored: `output/dundee_rt.csv`, row `1,2,3,10000,tourists,8,221,221,221,527`
(columns `text_id, wnum, subj_id, sent_id, word, wlen, ffd, fpd, rpd, td`; saved at [step1 lines 255-257](../phase1_replication/src/step1_parse_dundee.py#L255-L257))

## 1.6 Average over the 10 readers (what the model actually learns)

**Code:** [step4 lines 77-82](../phase1_replication/src/step4_system1.py#L77-L82): `sum over readers ÷ 10`.

FFD of "tourists" for all 10 readers (from `dundee_rt.csv`):

| reader | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| FFD | 156 | 156 | 126 | 221 | 206 | 137 | 177 | 270 | 108 | 179 |

**By hand:** (156+156+126+221+206+137+177+270+108+179) ÷ 10 = 1736 ÷ 10 = **173.6 ms**

### ✅ Stored: `output/system1_predictions.csv` → `ffd` = **173.6** (also `fpd` 261.0, `rpd` 261.0, `td` 312.8)

---

# PART 2: WORD FEATURES (step 2 → `output/lexical_features.csv`)

## 2.1 `wlen`: word length

**Meaning:** number of characters in the word.
**Where from:** Dundee's own word list, `Dataset/dundee_corpus/per-word-info/tx01wrdp.dat`, column 7 ([step1 line 143](../phase1_replication/src/step1_parse_dundee.py#L143)).
**By hand:** t-o-u-r-i-s-t-s = **8**
### ✅ Stored: `lexical_features.csv` → `wlen` = **8**

## 2.2 `sent_len`: sentence length

**Meaning:** how many words are in the sentence.
**Code:** [step2 lines 213-216](../phase1_replication/src/step2_lexical_features.py#L213-L216): group the words by `sent_id` and count them.
**By hand:** "Are tourists enticed by these attractions threatening their very existence?" = **10 words**
### ✅ Stored: `lexical_features.csv` → `sent_len` = **10**
(Step 4 counts it again from unique words as a safety check, [step4 lines 72-75](../phase1_replication/src/step4_system1.py#L72-L75), with the same result.)

## 2.3 `wiki_freq`: word frequency

**Meaning:** how common the word is in English. Common words are read faster.
**Formula:** `log( frequency + 0.000000001 )`. The tiny number stops log(0) for unknown words.
**Code:** [step2 line 39](../phase1_replication/src/step2_lexical_features.py#L39), using the `wordfreq` library (Wikipedia, books, subtitles…).

**By hand:**
- `word_frequency("tourists")` = 0.0000115 (= 11.5 times per million words)
- log(0.0000115 + 0.000000001) = **−11.373**

For comparison: "are" = 0.0055 → **−5.203** (very common), "enticed" = 0.000000562 → **−14.390** (rare).
**Closer to 0 = more common. More negative = rarer.**
### ✅ Stored: `lexical_features.csv` → `wiki_freq` = **−11.37307656985383**

## 2.4 `aoa_mean`: age of acquisition

**Meaning:** the age (in years) at which people usually learn the word. Late-learned words are read slower.
**File:** `Dataset/AoA.csv` (Kuperman et al. 2012 norms).
**Code:** [step2 lines 60-62](../phase1_replication/src/step2_lexical_features.py#L60-L62): use column `AoA_Kup`. If that's empty, use `AoA_Kup_lem` (the value for the word's base form).

**By hand:** row "tourists" in `AoA.csv`:

| AoA_Kup | AoA_Kup_lem | AoA_Bird_lem | AoA_Bristol_lem | AoA_Cort_lem |
|---|---|---|---|---|
| (empty) | **8.76** | (empty) | 8.47 | (empty) |

`AoA_Kup` is empty → fall back to `AoA_Kup_lem` → **8.76 years**
### ✅ Stored: `lexical_features.csv` → `aoa_mean` = **8.76**

## 2.5 `aoa_std`: age of acquisition spread

**Meaning:** how much the different AoA studies **disagree** about the word.
**Code:** [step2 lines 65-70](../phase1_replication/src/step2_lexical_features.py#L65-L70): take the standard deviation of the non-empty values among `AoA_Kup`, `AoA_Bird_lem`, `AoA_Bristol_lem`, `AoA_Cort_lem`. With only 1 value, the spread is 0.

**By hand ("tourists"):** non-empty values = only 8.47 (Bristol) → just 1 value → **0.0**
**By hand ("are"):** values = Bird 5.53, Cort 3.54 → mean 4.535 → std = |5.53 − 4.535| = **0.995** ✓ (stored 0.995)

Words not in `AoA.csv` get the **average** of all words ([step2 lines 247-254](../phase1_replication/src/step2_lexical_features.py#L247-L254)).
### ✅ Stored: `lexical_features.csv` → `aoa_std` = **0.0**

## 2.6 `fwd_prob`: forward transition probability

**Meaning:** how likely this word is, **given the word before it**: P(tourists | are).
**Data:** British National Corpus (100 million words). Counts are cached in
`output/bnc_bigrams.csv` (single word counts) and `output/bnc_bigrams_full.csv` (word-pair counts),
made at [step2 lines 148-170](../phase1_replication/src/step2_lexical_features.py#L148-L170).
**Formula (add-1 smoothing):**

```
fwd_prob = log(  (count(prev word, this word) + 1)  /  (count(prev word) + V)  )
```
**Code:** [step2 lines 191-193](../phase1_replication/src/step2_lexical_features.py#L191-L193), log at [line 207](../phase1_replication/src/step2_lexical_features.py#L207).

**By hand:**

| Number | Value (from the BNC caches) |
|---|---|
| count("are tourists") | 6 |
| count("are") | 464,556 |
| V = number of different words in BNC | 335,051 |

(6 + 1) / (464,556 + 335,051) = 7 / 799,607 = 0.000008754 → log = **−11.646**

**Why +1 and +V?** Without it, any word pair never seen in BNC would get probability 0 and log(0) = −∞.
### ✅ Stored: `lexical_features.csv` → `fwd_prob` = **−11.64596548689194**

## 2.7 `bwd_prob`: backward transition probability

**Meaning (paper):** how likely this word is, **given the word after it**: P(tourists | enticed).
**Code:** [step2 lines 198-203](../phase1_replication/src/step2_lexical_features.py#L198-L203).

**By hand:** count("enticed") = 77, and the pair never appears in BNC, so the count is 0.
(0 + 1) / (77 + 335,051) = 1 / 335,128 = 0.000002984 → log = **−12.722**
### ✅ Stored: `lexical_features.csv` → `bwd_prob` = **−12.72226782738182**

> ⚠️ **Known issue (found 2026-10-20).** The code looks up the pair in the **wrong order**:
> it counts "next word + this word" instead of "this word + next word".
> For "tourists" it makes no difference (both pairs have count 0), but for "Are" it does:
> the code uses count("tourists are") = 44 → **−8.92** (stored), where the paper's definition uses count("are tourists") = 6 → −10.78.
> **Effect on results:** very small. In the ablation, adding fwd + bwd moves R² only from 0.505 to 0.506.
> **If asked, say:** *"I found this while checking the code. It barely affects the result, and I will fix it and rerun."*

---

# PART 3: SURPRISAL AND GRAMMAR FEATURES (step 3 → `output/syntactic_features.csv`)

## 3.1 `lex_surprisal`: lexical surprisal

**Meaning:** how **surprised** a reader is by this word after the previous word. Big = unexpected.
**Formula:** `−log P(word | previous word)`, with the same BNC counts and add-1 smoothing as `fwd_prob`.
**Code:** [step3 lines 104-109](../phase1_replication/src/step3_syntactic_features.py#L104-L109). The first word of a text gets 0 ([line 98-99](../phase1_replication/src/step3_syntactic_features.py#L98-L99)).

**By hand:** P(tourists | are) = 0.000008754 (from 2.6) → −log = **11.646**
(It is exactly `fwd_prob` with the sign flipped.)

For "by" after "enticed": count("enticed by") is small → **10.643**. Remember this number for GPT-2.
### ✅ Stored: `syntactic_features.csv` → `lex_surprisal` = **11.6460**

## 3.2 `syn_surprisal`: syntactic surprisal

**Meaning:** how surprising the **grammar structure** is at this word.
**How it's made:** `experiments/build_lcparse_surprisal.py` (run once, inside WSL Linux):
1. Trains a left-corner parser model on Penn Treebank WSJ sections 2-21
2. Parses every Dundee sentence with the Berkeley parser
3. For each word, surprisal = −log P(parser decisions at this word)
4. Saves `output/lcparse_syn_surprisal.csv` (column `syn_surp_pcfg`)

**Then step 3 copies it in:** [step3 lines 341-349](../phase1_replication/src/step3_syntactic_features.py#L341-L349), joined on `(text_id, wnum)`.

**Values:** Are = 35.826 (first word, everything is new), **tourists = 6.738**, enticed = 5.546, by = 0.199
### ✅ Stored: `syntactic_features.csv` → `syn_surprisal` = **6.7381** (from `lcparse_syn_surprisal.csv` = 6.73812)

**Key point:** its correlation with FFD is **r = −0.017**, almost none. That's why Phase 2 replaces it.

## 3.3 `total_surprisal`

**Formula:** lex + syn. **Code:** [step3 line 347](../phase1_replication/src/step3_syntactic_features.py#L347).
**By hand:** 11.646 + 6.738 = **18.384**
### ✅ Stored: `syntactic_features.csv` → `total_surprisal` = **18.3841**
(Saved but **not used** by the model, because it is not in `FEATURE_GROUPS`, [step4 lines 36-49](../phase1_replication/src/step4_system1.py#L36-L49).)

## 3.4 `entropy_red`: entropy reduction

**Meaning:** **entropy** = how uncertain we are about what kind of word comes next.
Entropy **reduction** = how much that uncertainty **drops** at this word. A big drop = the reader has to do more work.
**Data:** part-of-speech (POS) pairs from Penn Treebank, cached in `output/ptb_pos_lm.pkl`.
The POS tag of each Dundee word comes from `per-word-lexical-annotation/text1.txt`, column 2.

**Formula:**
```
H(tag) = − Σ  P(next tag | tag) × log P(next tag | tag)
entropy_red = max(0,  H(previous word's tag) − H(this word's tag))
```
**Code:** H at [syntactic_extractor.py lines 123-133](../phase1_replication/src/syntactic_extractor.py#L123-L133), reduction at [step3 lines 295-301](../phase1_replication/src/step3_syntactic_features.py#L295-L301).

**By hand:**
- "Are" = **VBP** (verb). After VBP, Penn Treebank has: VBN 16.6%, RB 16.4%, DT 12.0%, IN 9.3%, … → H(VBP) = **2.6328**
- "tourists" = **NNS** (plural noun). After NNS: IN 23.5%, "." 13.5%, "," 12.4%, … → H(NNS) = **2.6228**
- entropy_red = max(0, 2.6328 − 2.6228) = **0.0100**
### ✅ Stored: `syntactic_features.csv` → `entropy_red` = **0.0099** (same number, rounded)

## 3.5 `emb_depth`: embedding depth

**Meaning:** how deep the word sits in the sentence's **dependency tree** = how many steps up to the main verb (root).
**Data:** `Dataset/dundee_corpus/per-word-lexical-annotation/text1.txt`. Column 6 = word id in the sentence, column 8 = id of its **head** (0 = root).
**Code:** [step3 lines 135-196](../phase1_replication/src/step3_syntactic_features.py#L135-L196): follow heads upward until reaching 0, counting the steps.

**The real annotation of the sentence:**

| id | word | head | path to root | depth |
|---|---|---|---|---|
| 1 | Are | 3 | Are → enticed | 1 |
| 2 | **tourists** | 3 | **tourists → enticed** | **1** |
| 3 | enticed | 0 | (root) | 0 |
| 4 | by | 6 | by → attractions → enticed | 2 |
| 5 | these | 6 | these → attractions → enticed | 2 |
| 6 | attractions | 3 | attractions → enticed | 1 |

### ✅ Stored: `syntactic_features.csv` → `emb_depth` = **1**

## 3.6 `emb_diff`: embedding difference

**Formula:** depth(this word) − depth(previous word). **Code:** [step3 lines 304-307](../phase1_replication/src/step3_syntactic_features.py#L304-L307).
**By hand:** depth(tourists) − depth(Are) = 1 − 1 = **0**
(For "by": 2 − 0 = **+2**, a jump deeper into the tree.)
### ✅ Stored: `syntactic_features.csv` → `emb_diff` = **0**

## 3.7 `h1`–`h8`: hierarchical structure flags

**Meaning:** 8 yes/no (1/0) flags describing **how the tree depth changes** from the previous word to this one.
**Code:** [step3 lines 229-236](../phase1_replication/src/step3_syntactic_features.py#L229-L236).

| Flag | Rule | "tourists" (depth 1, prev 1) | "enticed" (0, prev 1) | "by" (2, prev 0) |
|---|---|---|---|---|
| h1 | depth changed | 0 | 1 | 1 |
| h2 | went deeper | 0 | 0 | 1 |
| h3 | went up | 0 | 1 | 0 |
| h4 | this word is the root (depth 0) | 0 | 1 | 0 |
| h5 | always 0 (needs child lists, not done) | 0 | 0 | 0 |
| h6 | jumped deeper by more than 1 | 0 | 0 | 1 |
| h7 | jumped up by more than 1 | 0 | 0 | 0 |
| h8 | same depth | **1** | 0 | 0 |

### ✅ Stored: `syntactic_features.csv` → for wnum 2: `h1…h8` = **0 0 0 0 0 0 0 1**

> ⚠️ **Known issue (found 2026-10-20).** [Line 218](../phase1_replication/src/step3_syntactic_features.py#L218) numbers the words 1, 2, 3, … instead of
> using the real `wnum`. All 20 texts have gaps (words that no reader looked at), so after the first gap the flags
> belong to a shifted word. Only **57%** of `h2` values match `emb_diff > 0`.
> At the start of text 1 (above) they are still correct.
> **Effect on results:** very small. Adding the whole hierarchical group changes R² by 0.000 in the ablation (0.506 → 0.506).
> **If asked, say:** *"I found this alignment bug while checking. These flags add nothing to R², so the results stand, and I will fix it."*

---

# PART 4: SYSTEM 1 PREDICTION (step 4 → `output/system1_predictions.csv`)

How the model turns the 19 features of "tourists" into **174.7 ms**.

**Step A: standardise** each feature ([step4 lines 232-233](../phase1_replication/src/step4_system1.py#L232-L233)):
`z = (value − mean over training words) ÷ std over training words`

**Step B: multiply** each z by the Ridge coefficient learned in training ([lines 243-244](../phase1_replication/src/step4_system1.py#L243-L244)), then **add** them all to the intercept.

| Feature | value | train mean | train std | z | coefficient | z × coef (ms) |
|---|---|---|---|---|---|---|
| **wlen** | 8 | 4.939 | 2.729 | 1.122 | 25.771 | **+28.91** |
| sent_len | 10 | 27.064 | 13.016 | −1.311 | −0.221 | +0.29 |
| **wiki_freq** | −11.373 | −7.617 | 3.284 | −1.143 | −22.155 | **+25.33** |
| aoa_mean | 8.76 | 6.163 | 2.614 | 0.993 | −0.841 | −0.84 |
| aoa_std | 0.0 | 0.356 | 0.338 | −1.052 | 0.840 | −0.88 |
| fwd_prob | −11.646 | −8.412 | 3.298 | −0.981 | 0.079 | −0.08 |
| bwd_prob | −12.722 | −10.462 | 2.512 | −0.900 | −1.612 | +1.45 |
| lex_surprisal | 11.646 | 8.407 | 3.301 | 0.981 | −3.009 | −2.95 |
| syn_surprisal | 6.738 | 4.616 | 6.326 | 0.335 | 0.483 | +0.16 |
| entropy_red | 0.010 | 0.258 | 0.378 | −0.654 | 0.317 | −0.21 |
| emb_depth | 1 | 2.752 | 1.775 | −0.987 | −0.076 | +0.08 |
| emb_diff | 0 | 0.013 | 1.361 | −0.009 | −1.090 | +0.01 |
| h1 | 0 | 0.670 | 0.470 | −1.426 | −0.738 | +1.05 |
| h2 | 0 | 0.321 | 0.467 | −0.688 | −0.411 | +0.28 |
| h3 | 0 | 0.349 | 0.477 | −0.733 | −0.325 | +0.24 |
| h4 | 0 | 0.045 | 0.208 | −0.218 | 0.047 | −0.01 |
| h6 | 0 | 0.157 | 0.364 | −0.432 | −0.206 | +0.09 |
| h7 | 0 | 0.068 | 0.252 | −0.270 | 0.459 | −0.12 |
| h8 | 1 | 0.329 | 0.470 | 1.427 | −0.879 | −1.25 |
| | | | | | **sum** | **+51.55** |

**Step C:** prediction = intercept + sum = **123.18 + 51.55 = 174.73 ms**

**Step D: threshold** ([step4 line 308](../phase1_replication/src/step4_system1.py#L308)): FFD's threshold is 56 ms; 174.7 > 56, so it is kept.
(A prediction under 56 would become 0, meaning "skipped".)

**What this shows:** "tourists" is predicted **51 ms slower than an average word**, mostly because it is **long** (+28.9 ms) and **rare** (+25.3 ms).
The two biggest coefficients belong to word length and frequency, the two strongest features in the paper too.
The chosen Ridge strength is alpha = 100 ([lines 236-241](../phase1_replication/src/step4_system1.py#L236-L241)).

### ✅ Stored: `output/system1_predictions.csv` → `predicted_ffd` = **174.7** (real average: **173.6**)
### ✅ Model stored: `output/system1_models.pkl` → `["ffd"]` holds `model` (coefficients + intercept), `scaler` (means + stds), `threshold` (56), `features` (the 19 names)

---

# PART 5: SYSTEM 2 FEATURE VECTOR (step 5, built in memory, not saved)

Example: the **first pair** in `Dataset/Wiki/aligned-good(0.67).gz`:

- **WIKI (hard):** "Gingerbread was brought to Europe in 992 by the Armenian monk Gregory of Nicopolis …" (24 words)
- **SIMPLE (easy):** "Armenian monk Gregory of Nicopolis … brought ginger bread to Europe in 992 ." (22 words)

**Step A: predict FFD for every word** with System 1 ([step5 lines 48-72](../phase1_replication/src/step5_system2.py#L48-L72)).
Wikipedia sentences have no eye-tracking data and no parse, so only the word features are filled in:
`wlen`, `sent_len`, `wiki_freq`, `aoa_mean`, `aoa_std`. Everything else is 0 ([lines 109-129](../phase1_replication/src/step5_system2.py#L109-L129)).

- WIKI → 224.0, 88.0, 151.6, 71.4, 144.0, … (24 numbers). "Gingerbread" = 224 ms (long + rare)
- SIMPLE → 185.3, 145.0, 171.7, 71.6, 228.7, … (22 numbers)

**Step B: pad each to 60** with zeros ([lines 192-193](../phase1_replication/src/step5_system2.py#L192-L193)) and join → **120 numbers** ([line 195](../phase1_replication/src/step5_system2.py#L195)):
```
row 1: [224.0, 88.0, 151.6, … 0, 0 | 185.3, 145.0, 171.7, … 0, 0]   label 1  (first = WIKI = harder)
row 2: [185.3, 145.0, 171.7, … 0, 0 | 224.0, 88.0, 151.6, … 0, 0]   label 0  (swapped)
```
Every pair gives 2 rows ([lines 195-202](../phase1_replication/src/step5_system2.py#L195-L202)) → 117,712 pairs × 2 = **235,424 rows × 120 columns**.

**Step C: Extended model** adds 6 numbers ([lines 207-248](../phase1_replication/src/step5_system2.py#L207-L248)) → **126 columns**:

| extra feature | row 1 value |
|---|---|
| length of sentence 1 | 24 |
| length of sentence 2 | 22 |
| mean predicted FFD, sentence 1 | 147.87 |
| mean predicted FFD, sentence 2 | 153.95 |
| sum of surprisal, sentence 1 | 0.0 |
| sum of surprisal, sentence 2 | 0.0 |

> Note: the two surprisal sums are **always 0**, because `lex_surprisal` is not computed for Wikipedia sentences (it falls back to 0).
> So the Extended model's gain (74.15 → 75.26%) comes from **sentence length and mean FFD**.

**Step D: classify** with Logistic Regression, 10-fold cross-validation ([lines 304-318](../phase1_replication/src/step5_system2.py#L304-L318)) → **74.15%** (Base), **75.26%** (Extended).

---

# PART 6: GPT-2 SURPRISAL (Phase 2 → `output/gpt2_surprisal_gpt2_ctx.csv`)

**Meaning:** how unexpected the word is to GPT-2, a language model that reads **all the previous words**,
not just one like the bigram method.
**Formula:** `surprisal = −ln P(word | all previous words)`
**Input:** `output/dundee_sentences.txt` (sentence text) + `output/dundee_sent_map.json` (which `wnum` each word is).
**Code:** [compute_gpt2_surprisal.py lines 87-91](../phase1_replication/experiments/compute_gpt2_surprisal.py#L87-L91).

**By hand.** GPT-2's probability for each real next token in sentence 1:

| token | P(token \| everything before) | −ln P |
|---|---|---|
| are | 0.00016 | 8.7248 |
| **tourists** | **0.00001** | **11.6779** |
| ent | 0.00012 | 8.9960 |
| iced | 0.32635 | 1.1198 |
| **by** | **0.40686** | **0.8993** |
| these | 0.00137 | 6.5948 |
| attractions | 0.00808 | 4.8182 |
| threatening | 0.00001 | 12.0415 |
| their | 0.12269 | 2.0981 |
| very | 0.00867 | 4.7482 |
| existence | 0.49117 | 0.7110 |
| ? | 0.04260 | 3.1560 |

**Words split into pieces are added together** ([lines 96-116](../phase1_replication/experiments/compute_gpt2_surprisal.py#L96-L116)):
- enticed = "ent" + "iced" = 8.9960 + 1.1198 = **10.1158**
- existence? = "existence" + "?" = 0.7110 + 3.1560 = **3.867**

**Context:** with `--context-sentences 3`, the 3 previous sentences of the same text are put in front ([lines 50-63](../phase1_replication/experiments/compute_gpt2_surprisal.py#L50-L63)).
This is sentence 1 of text 1, so it has no previous sentences and the values are the same with or without context.
From sentence 2 onward, the context changes the values.

### ✅ Stored: `gpt2_surprisal_gpt2_ctx.csv` → row `1,2,11.6779` (tourists), `1,4,0.8993` (by)

### The key comparison: the word "by"

| Method | Looks at | "by" after "…enticed" |
|---|---|---|
| Bigram (`lex_surprisal`) | only "enticed" | **10.64** (surprising) |
| GPT-2 (`gpt2_surprisal`) | "Are tourists enticed" | **0.90** (expected: P = 41%) |

Readers find "by" easy here, so GPT-2 is right. That is why GPT-2 surprisal correlates with FFD at
**r = 0.465**, while parser surprisal manages only **r = −0.017**.

**Then** `experiments/train_system1_gpt2.py` swaps `syn_surprisal` for this column ([line 57](../phase1_replication/experiments/train_system1_gpt2.py#L57))
and retrains with the step 4 code → FFD R² 0.444 → **0.458**, RPD 0.555 → **0.582**, TD 0.536 → **0.568**.

---

## How I checked all of this

The script `hand_calc.py` (kept outside the project) recomputed every value above from the raw
`Dataset/` files and compared it with the stored CSVs. Every value matched to the last decimal.
It also found the two known issues (bwd_prob word order, h1–h8 alignment) and the always-0 surprisal sums in System 2.

# Recording Script — Phase 2 (GPT-2) + Phase 3 (Tamil)

Simple English. Real code explained line by line, not just output numbers.
This script is written to actually fill 30 minutes of talking — it does not rely on guessed pause time.

Open this file in VS Code and press **Ctrl+Shift+V** to preview it nicely while you record.

---

## Before You Start Recording

**1. Open these, in this order:**

| # | What | Why |
|---|---|---|
| 1 | `results/GPT2_Enhancement_and_Tamil_Plan.pptx` | Your slides — open it, go to Slide 1, do **not** start the slideshow yet |
| 2 | VS Code, with these files open as tabs: `src/step2_lexical_features.py`, `src/step3_syntactic_features.py`, `src/step4_system1.py`, `src/feature_demo.py`, `experiments/compute_gpt2_surprisal.py`, `experiments/train_system1_gpt2.py`, `experiments/gpt2_sentence_correlation.py`, `experiments/fixated_only_system1.py`, `experiments/check_wlen_r2_consistency.py` | You will show the **real code that computes each feature**, not just a script that prints results |
| 3 | `output/lexical_features.csv` and `output/syntactic_features.csv` | Extra tabs — you will point at real column values after explaining the code that made them |
| 4 | A terminal inside VS Code, already moved to the right folder: `cd "phase1_replication"` | So you don't waste recording time typing paths |
| 5 | Your screen recording tool (OBS, Zoom, PowerPoint's own "Record Slide Show", or Windows `Win+G`) | Record your **whole screen**, since you switch windows |

**2. Before you press record:** close notification apps, test your mic, keep this script open on a second screen or phone.

**3. This script is long on purpose.** A 30-minute recording needs about 3,800–4,300 spoken words at a careful, clear pace (roughly 130 words per minute when reading technical content aloud). This script is written to that length. You do **not** need to add filler or slow down artificially — just read it naturally, pause briefly after each code explanation, and it will land close to 30 minutes. If your own pace is faster, the live terminal runs and pauses for pointing will bring you back up to 30. If you must cut time, use the **"Cut here first"** markers.

**4. Screen cues used below:**
- 🖥️ **SLIDE** = PowerPoint, in slideshow mode
- 💻 **CODE** = VS Code, a specific file and line
- ⚫ **TERMINAL** = the black terminal window, running a command

---

# PART A — How the 17 Features Are Actually Computed *(code-heavy, ~14 min)*

**🖥️ SLIDE:** Slide 1 (Title) — this slide shows "Enhancing Reading-Time Prediction with GPT-2 Surprisal," with the subtitle "Replication of Singh et al. (2016) + GPT-2 Enhancement + Next Steps for Tamil."
**Say:** *"Hello. My project is titled 'Quantifying Sentence Complexity Based on Eye-Tracking Measures.' Today's video covers one part of that work — enhancing reading-time prediction using GPT-2, which you can see on this slide. It builds on my earlier replication of Singh et al. 2016, and I'll also share my next steps for Tamil. Before I show my new results, I'd like to walk you through my code — instead of just showing the final numbers, I'll try to explain, gently, line by line, how each of my 17 features is calculated. There are 6 lexical features and 11 syntactic or structure features. Let me start with those."*

### A1 — The raw data (no function yet, just the file format)
**💻 CODE:** none yet — say this while still on the slide, then move to code.
**Say:** *"Before any feature is calculated, I have raw eye-tracking data from 10 people reading the same newspaper texts. Each row says: which word, which text, the word's position, and how long the eye stayed on it, in milliseconds. If a person skipped a word, there is no row at all for that word from that person. I will explain why this detail matters, later in this video."*

### A2 — Averaging reading time across 10 people
**⚠️ FILE FOR THIS PART: `step4_system1.py`** — click that tab now (not step2, not step3), find the function called `load_data`, **lines 77–86**.
**Say:** *"This is the real code that turns 10 people's separate reading times into one number per word. This is inside a file called step4_system1.py. Look at line 79: `rt_sum = lex.groupby(['text_id','wnum'])[RT_MEASURES].sum()`. In plain words: this line groups all the rows that belong to the same word — same text, same position — and adds up their reading times. `groupby` is a pandas command that means 'put all matching rows into one bucket.' Then on line 80, I divide that total by the number of subjects, which is 10, to get the average. On line 82, `fillna(0.0)` means: if a word has no data at all for some reason, treat it as zero instead of leaving it blank, so the model never crashes on a missing value. This average number — one per word — is what every feature is trying to predict."*

### A3 — The 6 lexical features (real code from `step2_lexical_features.py`)

**Feature 1 — word length (wlen)**
**Say:** *"Feature 1 is word length — simply the number of letters. This one is not calculated by my code at all. It already exists as a column in the original Dundee data, so I just reuse it directly."*

**Feature 2 — sentence length (sent_len)**
**⚠️ SWITCH FILE NOW → `step2_lexical_features.py`** (you were on step4 for A2 — click the step2 tab), function `compute_sentence_lengths`, **lines 213–216**.
**Say:** *"Here is the actual function. Line 215: `sent_sizes = rt_df.groupby('sent_id').size()`. This groups every word by which sentence it belongs to, then `.size()` counts how many words landed in each group — that count becomes the sentence length. Line 216 merges that count back onto every single word row, so every word in a 12-word sentence gets the number 12 in its sent_len column. It's a two-line function, but it's doing a real group-and-count operation, not just a lookup."*

**Feature 3 — Wikipedia frequency (wiki_freq)**
**💻 CODE (still `step2_lexical_features.py`, no switch):** function `load_wiki_freq`, **lines 31–41**.
**Say:** *"Line 38 collects every unique word that appears anywhere in my data. Line 39 is the interesting one: `result = {w: math.log(word_frequency(w, 'en') + 1e-9) for w in unique_words}`. This is called a dictionary comprehension — it's a shortcut in Python for a loop that builds a dictionary in one line. In plain English, it says: for every word, ask the wordfreq library how common that word is in English, then take the logarithm of that number, and store it in a dictionary where the word is the key and the frequency score is the value. So later, when I need the frequency of the word 'tourists', I just look it up in this dictionary instantly, instead of asking the library again every time."*

**Features 4 and 5 — Age of Acquisition (aoa_mean, aoa_std)**
**💻 CODE (still `step2_lexical_features.py`, no switch):** function `load_aoa`, **lines 46–76**.
**Say:** *"This function reads a file of word-learning-age norms. Line 58, `for _, row in df.iterrows()`, means: go through this table one row at a time, top to bottom, like reading a spreadsheet. For each word, line 60 reads its average learning age. Lines 65 to 70 do something a bit more careful: since the file has several different studies measuring the same thing slightly differently, I collect all of those different measurements into a small list called `vals`, and then use their spread, their standard deviation, as my aoa_std feature — basically, how much disagreement there is between studies about when this word is learned. If a word is missing from this dictionary entirely, later code falls back to the average across all known words, so no word is ever left without a value."*

**Features 6 and 7 — forward and backward transition probability (fwd_prob, bwd_prob)**
**💻 CODE (still `step2_lexical_features.py`, no switch):** functions `build_bnc_bigrams` (**lines 97–173**) and `compute_transition_probs` (**lines 176–208**).
**Say:** *"This is the most involved part of the lexical features, so I will slow down here. First, `build_bnc_bigrams` reads through thousands of files from the British National Corpus — 100 million words of real English text. For every pair of words that appear next to each other, it counts how many times that exact pair happened. Look at lines 154 to 159: I use something called a defaultdict of defaultdicts — think of it as a dictionary of dictionaries. The outer dictionary's key is the first word, and its value is another dictionary counting every word that ever came right after it, and how many times. Line 158, `fwd_counts[w1][w2] += 1`, simply means: every time I see word1 followed by word2, add one to that pair's counter. Once all 100 million words are counted, `compute_transition_probs` turns raw counts into probabilities. Look at line 193: `fwd_p = (fwd_count + 1) / (prev_total + len(unigrams))`. This is called add-one smoothing — I add 1 to the count on top, and add the whole vocabulary size on the bottom. This trick means that even a word pair the model has never seen before still gets a small, safe probability instead of a probability of exactly zero, which would break the logarithm in the next step."*

*(A3 total is intentionally long — this is the densest code section. ~4–5 min.)*

### A4 — The 11 syntactic and structure features (real code from `step3_syntactic_features.py`)

**Features 8 and 9 — lexical and syntactic surprisal**
**⚠️ SWITCH FILE NOW → `step3_syntactic_features.py`** (you were on step2 — click the step3 tab), function `compute_lexical_surprisal`, **lines 94–109**.
**Say:** *"Surprisal means: how surprised should we be to see this word, given the word right before it? Line 104 looks up how many times this exact pair of words appeared in the BNC data I just described. Line 108 turns that into a probability, again using the same add-one smoothing trick. Line 109 takes the negative logarithm of that probability — that's the actual surprisal number. A common, expected word pair gives a small surprisal number. A rare, unexpected pair gives a large one. The second feature, syntactic surprisal, works on the same idea but uses grammar categories instead of exact words, and it comes from a separate, more complex tool called a left-corner parser — my code simply merges that tool's output in as an extra column, rather than calculating it inline, since it is a whole parser algorithm on its own."*

**Feature 10 — entropy reduction**
**💻 CODE (still `step3_syntactic_features.py`, no switch):** main function, **lines 294–301**.
**Say:** *"This feature asks: how much does this word narrow down what could come next, grammatically? Line 296 calculates the current uncertainty, called entropy, based on this word's grammar category. Line 300, `ent_red = max(0.0, prev_entropy - curr_entropy)`, subtracts the new uncertainty from the previous word's uncertainty. If uncertainty dropped, that difference is a positive number — a real entropy reduction. The `max(0.0, ...)` part means: if uncertainty actually went up instead of down, just record zero, since a negative reduction isn't a meaningful reading-difficulty signal."*

**Features 11 and 12 — embedding depth and embedding difference**
**💻 CODE (still `step3_syntactic_features.py`, no switch):** function `load_dependency_parse`, **lines 135–196**, focus on the while-loop at **lines 178–190**.
**Say:** *"Every sentence has a grammar tree, where each word points to the word it depends on — its 'head' — and the very top word, the root, has no head. To find how deep a word sits in this tree, the code climbs upward: line 187, `cur = head`, moves from the current word up to its head, then that head's head, and so on, one step per loop, until it reaches the root at depth zero. It keeps track of every word it passed on the way up in a list called `path`, at line 181. Once it reaches the root, lines 189 to 190 walk back down that same path, assigning depth 1, depth 2, depth 3, and so on, as it goes. This is a real tree-climbing algorithm, not a lookup — it has to walk the chain because depth isn't stored anywhere directly in the data."*

**Features 13 to 17 — the h1 through h8 structure flags**
**💻 CODE (still `step3_syntactic_features.py`, no switch):** function `compute_hierarchical_features`, **lines 201–241**.
**Say:** *"Once every word has a depth number, these last features just compare each word's depth to the word right before it. Lines 229 to 236 are eight simple yes-or-no questions: did the depth change at all, did it go up, did it go down, is this word at the very top of the tree, did it jump by more than one level up or down, and did it stay exactly the same. Each answer is stored as a 1 for yes or a 0 for no. It looks like a lot of code, but every line is the same simple comparison, just checking a different condition."*

**⚫ TERMINAL (now show real numbers for everything above):**
```
python src/feature_demo.py
```
**Say:** *"Now that I've explained how each feature is calculated, let me run the actual script and show you real numbers for real words from my data, so you can see the code and the output match."*
**Point at:** scroll through the printed sections for each feature as you name them, matching each one back to the function you just explained.

### A5 — From features to the R² result (real code from `step4_system1.py`)
**⚠️ SWITCH FILE NOW → `step4_system1.py`** (back to the file from A2 — click that tab), functions `split_by_sentence` (**lines 120–143**), `train_and_evaluate` (**lines 216–258**), `tune_threshold` (**lines 153–173**).
**Say:** *"Now the training code. First, `split_by_sentence` divides my data into training, tuning, and testing groups. Line 125 sets a fixed random seed, 42 — this means the split is random, but always the same random split every time I run it, so my results are reproducible. Lines 133 to 135 assign 60 percent of sentences to training, 20 percent to tuning, and the last 20 percent to testing — always splitting by whole sentences, never by individual words, so the model never sees part of a sentence during training and the rest during testing. Next, `train_and_evaluate`: line 232 creates something called a StandardScaler, which rescales every feature so they're all on a similar numeric scale — without this, a feature like sentence length, which can be a big number like 30, would unfairly dominate a feature like entropy reduction, which is usually a small decimal. Lines 236 to 241 try five different settings for the model, called Ridge regression, and keep whichever setting performs best on the tuning data. Finally, `tune_threshold`, lines 166 to 171, tests cutoff values from 50 to 200 milliseconds — any prediction below the cutoff gets changed to zero, because that usually means the model thinks the reader skipped the word. Whichever cutoff gives the best score on the tuning data is the one used on the final test."*

**Point at:** the FINAL RESULTS table from the earlier `feature_demo.py` run, or re-show it here.

**Part A total: ~14 minutes of real code explanation.**

---

# PART B — GPT-2 Enhancement *(code-heavy, ~13 min)*

### Slide 2 — Recap
**🖥️ SLIDE:** Slide 2
**Say:** *"So that is System 1 — it takes those 17 features and predicts 4 reading-time measures per word. On top of it sits System 2, which compares a Wikipedia sentence against its Simple Wikipedia version and decides which one is harder to read, using System 1's predictions as its input."*

### Slide 3 — Why GPT-2
**🖥️ SLIDE:** Slide 3
**Say:** *"I found that syntactic surprisal, one of my 17 features, barely helps at all — its connection to reading time is almost zero, minus 0.017. I checked the original research paper, and even their own version of this feature scores almost zero too, minus 0.067. So this isn't a mistake in my code — it's a real weak point in that particular method. That made me ask: instead of an older, rule-based grammar parser, could a modern AI language model guess word difficulty better?"*
**⚠️ SWITCH FILE NOW → `train_system1_gpt2.py`** (a new file, in the `experiments` folder, not `src`), **lines 76–80**.
**Say:** *"This loop prints the connection between reading time and both surprisal methods, in the exact same run, on the exact same data, so the comparison is fair."*

### Slide 4 — Method: explaining the actual GPT-2 code
**🖥️ SLIDE:** Slide 4
**Say:** *"Now let me explain exactly how I calculate this new feature, because this is genuinely new code I wrote, not just a library call."*
**⚠️ SWITCH FILE NOW → `compute_gpt2_surprisal.py`** (still in `experiments`, different file from the last slide), function `word_token_surprisals`, **lines 66–118**.
**Say:** *"Line 74 splits the sentence into separate words. Line 75 joins my earlier context — up to 3 previous sentences — onto the front of the current sentence, so GPT-2 isn't guessing blind at the very start. Line 77 feeds this whole piece of text into GPT-2's tokenizer, which breaks it into smaller pieces called tokens — a token can be a whole word or sometimes just part of a longer word. Lines 84 to 85 are the actual AI step: I run the GPT-2 model itself, and it returns a big table of numbers called logits, one row per token, containing a score for every possible next word in English. Line 87 turns those raw scores into real probabilities using a function called log-softmax. Line 90, `token_logp = log_probs[:-1].gather(...)`, is the key line — for each token, it looks up the probability the model actually gave to the token that really came next. Line 91 flips the sign to get surprisal: negative log probability. A word the model expected gets a small surprisal number; a word it didn't expect gets a large one — this is the exact same idea as the lexical surprisal feature from Part A, just calculated by a neural network instead of counting bigrams."*
**Say (the bug):** *"There's one more small part I'd like to mention, lines 106 to 116. Since one word can be split into several tokens by GPT-2's tokenizer, I need to add those pieces' surprisal values back together to get one number per whole word. While building this, I noticed a small bug: GPT-2's tokenizer sometimes attaches a blank space to the front of the next word's first token, instead of leaving it with the previous word. If I had matched tokens to words using where each token starts, I would sometimes have assigned the wrong word's surprisal to the wrong word, right at the boundary between my earlier context and the real sentence. Line 109 fixes this gently, by matching using where each token ends, instead of where it starts. I just wanted to share this small detail, since I think it's worth showing the small things I checked along the way."*
**⚠️ SWITCH FILE NOW → back to `train_system1_gpt2.py`**, **lines 27–31** and **59–63**.
**Say:** *"Line 30 adds this new GPT-2 number as an extra column onto my existing feature table. Lines 60 to 62 set up three versions of my model to compare fairly: one using only the old feature, one using both features together, and one that fully replaces the old feature with the new one."*

### Slide 5 — Results (main result, live)
**🖥️ SLIDE:** Slide 5
**⚫ TERMINAL:**
```
python experiments/train_system1_gpt2.py --surprisal-file output/gpt2_surprisal_gpt2_ctx.csv
```
**Say:** *"Here are the real numbers, calculated live. For two of the four measures, RPD and TD, my model now does better than the original paper. For FPD, I am very close, within 0.017. For FFD, I'm still a little behind, but it did improve — from 0.444 to 0.458 after adding GPT-2. This pattern makes sense: FFD is just the very first, quick glance at a word, mostly driven by simple things like word length. The other three measures include more of the full reading process, so a smarter predictability signal like GPT-2's has more room to help."*
**Point at:** each printed row, FFD, then FPD, RPD, TD, moving across the three number columns.

### Slide 6 — Making sure the new feature is trustworthy
**🖥️ SLIDE:** Slide 6
**⚫ TERMINAL:**
```
python experiments/gpt2_sentence_correlation.py
```
**⚠️ SWITCH FILE NOW → `gpt2_sentence_correlation.py`** (briefly), **lines 51–53**.
**Say:** *"Before trusting a new feature, I ran two checks. Look at lines 51 to 53 — I calculate three different ways of summarizing GPT-2's surprisal for a whole sentence: the mean, the maximum, and the sum. First check: is this new feature just copying my old syntactic surprisal feature? Their connection is almost zero, 0.014 — so no, it's genuinely new information. Second check, and this one is a mistake I almost made: if I add up, sum, surprisal across a whole sentence, that total ends up almost perfectly matching the sentence's word count — a connection of 0.914. That's not measuring difficulty at all, it's basically just counting words in disguise, since adding up more numbers naturally gives a bigger total for longer sentences. So I used the mean instead of the sum, which doesn't have this problem."*
**Point at:** the `syn_surprisal` row for 0.014, then the `sent_len` row under the `gpt2_sum` column for 0.914, then `gpt2_mean` next to it.

### Slide 7 — A wrong idea I tested and ruled out
**🖥️ SLIDE:** Slide 7
**⚠️ SWITCH FILE NOW → `fixated_only_system1.py`**, **line 60**.
**Say:** *"I had a theory: when a reader skips a word completely, that word's reading time becomes zero in my averaged data. So maybe word length wasn't really measuring reading difficulty — maybe it was secretly just measuring whether a word gets looked at at all, since longer words tend to get skipped less often. Line 60 tests this directly: `fix_df = df[df[rt_name] > 0].copy()` — this keeps only the words where the reading time is greater than zero, removing every skipped word, and retrains the whole model on just that smaller set."*
**⚫ TERMINAL:**
```
python experiments/fixated_only_system1.py
```
**Say:** *"The result: there was no difference at all — the scores are almost identical whether skipped words are included or not. Looking into why, I noticed that once you average across 10 readers, almost no word ends up with an average reading time of exactly zero — the smallest value across all 50,647 words is 5.1 milliseconds. So my original idea did not turn out to be true, and I wanted to include this part, because I think it's worth showing when an idea doesn't work out, and not only the parts that did."*
**Point at:** the printed comparison table — the all-words and fixated-only columns matching exactly.

### Slide 8 — System 2 still works well
**🖥️ SLIDE:** Slide 8
**Say:** *"One last piece — System 2, my readability classifier, still does a little better than the original paper on all four of its reported scores. Even though FFD alone is a bit behind, the final system built on top of these predictions still works well. So I think System 2 is in a good place for now, and I'd like to spend the rest of my time on the Tamil idea I'll explain next."*
**Point at:** the System 2 table already in the slide or in `README.md` — do not re-run this pipeline live, it takes too long.

**Part B total: ~13 minutes.**

---

# PART C — New Idea: Extending This Work to Tamil *(~5 min)* — Cut here first if you're short on time

### Slide 9 — Introducing Part 3
**🖥️ SLIDE:** Slide 9
**Say:** *"That was Phase 2. Now, if it's okay, I'd like to share an idea for Phase 3 — applying this same approach to the Tamil language. I should mention that this part is only a plan so far — there is no code or results yet."*

### Slide 10 — Why Tamil
**🖥️ SLIDE:** Slide 10
**Say:** *"I looked at existing research before suggesting this. As far as I could find, there is no eye-tracking reading dataset for Tamil yet, and no readability dataset either, even from the main research groups working on Tamil language processing. The closest similar project I found was for the Gujarati language, published in 2023. This suggests that this kind of project can be published for an Indian language — it just hasn't been done for Tamil yet. I did this search on my own, so it may not be a complete review, and I'd welcome your thoughts if you know of work I may have missed."*

### Slide 11 — The new idea
**🖥️ SLIDE:** Slide 11
**Say:** *"This is not simply copying my English code into Tamil — there's a real new question underneath it. I already showed that word length isn't a perfectly reliable difficulty measure, even in English. Tamil works very differently — one Tamil word can contain what English needs several separate words to say, by adding pieces onto the end of the word for tense, person, and grammar. So just counting letters is probably an even worse measure of difficulty in Tamil. My plan is to count meaningful word-parts instead, called morphemes, using a free, existing tool called ThamizhiMorph. Then I want to use that count to build and test a tool that simplifies Tamil sentences, checked against real people's opinions on which version is easier to read."*
**⚠️ SWITCH FILE (optional) → `check_wlen_r2_consistency.py`**, then run it:
```
python experiments/check_wlen_r2_consistency.py
```
**Say while showing this:** *"This is the same English-language finding I mentioned earlier — my own numbers stay perfectly consistent, but there's a real inconsistency in the original paper's own two tables for word length. This finding is the foundation this entire Tamil idea is built on."*

### Slide 12 — Why this is a good next step
**🖥️ SLIDE:** Slide 12
**Say:** *"I think this could be a good next step, for a few small reasons. First, I could reuse the GPT-2 code I already built and explained today — a Tamil version of GPT-2 already exists publicly. Second, the tools I would need already exist as free, open software, so this feels realistic to attempt, not just a hopeful idea. Third, this project could give two results: a research finding about whether word length works the same way across languages, and a working Tamil simplification tool. I also think this question is a little bigger than just Tamil — the same issue may show up in other languages built in a similar way, like Turkish, Finnish, and Korean."*

---

## Closing
**🖥️ SLIDE:** Last slide, or back to Slide 1
**Say:** *"That covers everything — my Phase 2 results, explained down to the actual code, and my plan for Phase 3. Thank you for watching. I'm happy to explain any part again or answer questions by email."*

---

## Time check

| Part | Content | Estimated time |
|---|---|---|
| Title | Intro | ~30 sec |
| Part A | 17 features, real code | ~14 min |
| Part B | GPT-2 code + results | ~13 min |
| Part C | Tamil proposal | ~5 min |
| Closing | | ~20 sec |
| **Total** | | **~32-33 min** |

This slightly overshoots 30 minutes on purpose, since your own reading speed will vary. If it runs long when you do a practice read-through:
1. **Cut first:** Part C — shorten slides 10–12 into one summary slide, saying "I have a written proposal for Tamil I'm happy to share separately" instead of all three slides in full.
2. **Cut second:** the "bug" explanation in Slide 4 (the tokenizer detail) — mention it in one sentence instead of walking through the fix.
3. **Never cut:** Part A — that is the section that proves you understand your own code.

Do one practice read-through out loud with a timer before your real recording, and adjust using the list above.

---

## If your professor replies with questions

| If asked... | Say... |
|---|---|
| Why Ridge regression, not a neural network? | To stay comparable with the original paper's method, and because it works well with a small number of features. |
| Could GPT-2's improvement be from memorized text? | Unlikely — GPT-2 is measuring word-by-word predictability from nearby context, not recalling memorized text. |
| Why only 3 sentences of context? | A practical choice to save computing time — testing more context is a natural next step. |
| Did you try combining both surprisal features, not just replacing one? | Yes — that is the "gpt2_added" version in the three-way comparison. *(Fill in the exact number before sending.)* |
| How long will the Tamil project take? | *(Decide your own estimate before sending.)* |

---

*Other versions in this folder: `GPT2_Enhancement_and_Tamil_Plan_Talking_Script.txt` (10-min tight version), `GPT2_Enhancement_and_Tamil_Plan_Talking_Script_30min.md` (live-meeting version with interruption handling and full Q&A).*

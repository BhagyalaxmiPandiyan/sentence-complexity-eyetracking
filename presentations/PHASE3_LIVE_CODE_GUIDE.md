# Phase 3: live code guide (with line numbers)

**How to use this in the meeting:**
1. Open this file in VS Code, and share your screen.
2. Click a line link, for example **[L140](../phase3_tamil_extension/src/step3_textbook_features.py#L140)**. VS Code opens the script at that exact line.
3. Point at the line and say the "Say" sentence.
4. Then open the **output file** and show a real row.

**The order:** for every step, say **Input → code lines → Output**.

---

## Step 0: download the textbooks
**Script:** [step0_fetch_textbooks.py](../phase3_tamil_extension/src/step0_fetch_textbooks.py)

### ❓ Why do we need this step?
- To test a difficulty measure, we need Tamil text where **we already know how hard it is**.
- In English, the paper had eye-tracking data. **In Tamil there is no eye-tracking data and no readability dataset.**
- **The idea:** Tamil Nadu government textbooks are written for each class. A Class 3 book is written for 8-year-olds and a Class 12 book for 17-year-olds. So **the class number tells us the difficulty**, and we get the labels for free.
- **The problem:** the official website was not working. The other websites that host the books also list unrelated books, so we can't trust a link blindly. **We must check every book.**

### ⚙️ How does it work? (5 simple steps)
1. For each class (3 to 12), take the download link.
2. Download the file.
3. Check that it is really a PDF.
4. Read its first 6 pages and look for the class name, for example "ஆறாம் வகுப்பு" for Class 6.
5. If the name is there, save the file as `class6.pdf`. If not, delete it and try the next link.

**Input:** download links. **Output:** 10 correct PDFs.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L22](../phase3_tamil_extension/src/step0_fetch_textbooks.py#L22) | `CANDIDATES = {...}` | "**Input:** the download links for each class, 3 to 12." |
| [L53](../phase3_tamil_extension/src/step0_fetch_textbooks.py#L53) | `def download(...)` | "This downloads one PDF." |
| [L57](../phase3_tamil_extension/src/step0_fetch_textbooks.py#L57) | `f.read(5) == b"%PDF-"` | "It checks that the file really is a PDF." |
| [L46](../phase3_tamil_extension/src/step0_fetch_textbooks.py#L46) | `ORDINAL = {... "ஆறாம் வகுப்பு" ...}` | "These are the class names printed on the title page." |
| [L66](../phase3_tamil_extension/src/step0_fetch_textbooks.py#L66) | `any(m in text for m in ORDINAL[n])` | "It checks that the book really says, for example, ஆறாம் வகுப்பு. If not, the book is thrown away." |
| [L87](../phase3_tamil_extension/src/step0_fetch_textbooks.py#L87) | `os.replace(tmp, final)` | "**Output:** a correct book is saved as class6.pdf." |

**Show the output:** the `Dataset/graded_textbooks/` folder, with 10 PDFs.

---

## Helper: fixing the broken PDF text
**Script:** [tamil_text.py](../phase3_tamil_extension/src/tamil_text.py)

### ❓ Why do we need this?
- When we copy text out of the textbook PDFs, **the Tamil letters come out broken**: அடித்்தளம் instead of அடித்தளம், and கிரா மத்தில் instead of கிராமத்தில்.
- If the words are broken, **every tool gives a wrong answer.** wordfreq can't find the word, Morfessor splits it wrongly, and GPT-2 finds it "surprising".
- So we must **repair the text first**, before any feature is calculated.

### ⚙️ How does it work?
1. Find doubled marks (்் or ொொ) and keep only one.
2. Find a word that was broken by a space. Join the pieces **only if** the joined word is a real Tamil word.

**Result:** real Tamil words went from **55% to 76.5%**.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L20](../phase3_tamil_extension/src/tamil_text.py#L20)–[L22](../phase3_tamil_extension/src/tamil_text.py#L22) | `_DOUBLE_PULLI...`, `_DOUBLE_SIGN` | "PDF text doubles the pulli and the vowel signs, like அடித்்தளம். These patterns find the doubled marks." |
| [L25](../phase3_tamil_extension/src/tamil_text.py#L25) | `def repair_pdf_text` | "This fixes them: அடித்்தளம் becomes அடித்தளம். Real words went from 55% to 76.5%." |
| [L33](../phase3_tamil_extension/src/tamil_text.py#L33) | `def merge_split_words` | "This joins split words back, like கிரா மத்தில் → கிராமத்தில்." |
| [L47](../phase3_tamil_extension/src/tamil_text.py#L47) | `known("".join(parts))` | "It only joins them if the result is a real word, so two real words are never glued together." |

---

## Step 1: take the sentences out of the PDFs
**Script:** [step1_extract_sentences.py](../phase3_tamil_extension/src/step1_extract_sentences.py)

### ❓ Why do we need this step?
- A PDF is **not** a list of sentences. It has poems, grammar lessons, exercises, page numbers, the national anthem and the table of contents.
- We want only **normal prose sentences**, the kind a student reads in a lesson.
- **Poetry is skipped** because it uses old, literary Tamil. A Class 6 poem may be harder than Class 12 prose, so it would confuse the test.
- **Exercises are skipped** because "எழுதுக", "____" and (அ) (ஆ) options are not real reading text.

### ⚙️ How does it work? (6 simple steps)
1. Read every page and repair the letters (using the helper above).
2. Skip the front pages (everything before the table of contents) and the pages with a broken font.
3. Find the lesson headings: **keep** உரைநடை உலகம் and விரிவானம், **skip** கவிதைப்பேழை and கற்கண்டு.
4. Cut the text into sentences at . ? !
5. Keep only clean sentences: 4–40 words, no broken characters, no exercises.
6. Remove repeated sentences and save.

**Input:** 10 PDFs. **Output:** `graded_sentences.csv` (2,351 sentences, each with its class).

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L32](../phase3_tamil_extension/src/step1_extract_sentences.py#L32) | `SECTION_START = {"கவிதைப்பேழை": "poetry", "உரைநடை உலகம்": "prose", ...}` | "Every lesson starts with a fixed heading. This is how I find the sections." |
| [L34](../phase3_tamil_extension/src/step1_extract_sentences.py#L34) | `KEEP = {"prose", "extra_reading"}` | "I keep only prose. I skip poetry, because it uses old Tamil." |
| [L35](../phase3_tamil_extension/src/step1_extract_sentences.py#L35) | `SECTION_END = ["மதிப்பீடு", ...]` | "The prose ends where the exercises start." |
| [L109](../phase3_tamil_extension/src/step1_extract_sentences.py#L109) | `toc = ... "பொருளடக்கம்"` | "I skip everything before the table of contents." |
| [L110](../phase3_tamil_extension/src/step1_extract_sentences.py#L110) | `page_known(t) >= 0.6` | "I skip pages where less than 60% of the words are real. Those pages have a broken font." |
| [L38](../phase3_tamil_extension/src/step1_extract_sentences.py#L38) | `SENT_SPLIT = ... [.?!]` | "A new sentence starts after a full stop, a question mark or an exclamation mark." |
| [L136](../phase3_tamil_extension/src/step1_extract_sentences.py#L136) | `for raw in SENT_SPLIT.split(...)` | "Here the text is cut into sentences." |
| [L75](../phase3_tamil_extension/src/step1_extract_sentences.py#L75) | `def is_clean(sent)` | "This checks whether a sentence is clean." |
| [L77](../phase3_tamil_extension/src/step1_extract_sentences.py#L77) | `MIN_WORDS <= len(words) <= MAX_WORDS` | "It must have 4 to 40 words." |
| [L79](../phase3_tamil_extension/src/step1_extract_sentences.py#L79) | `"�" in sent or ... INVALID_SEQ` | "No broken characters." |
| [L81](../phase3_tamil_extension/src/step1_extract_sentences.py#L81) | `EXERCISE ... INSTRUCTION_END` | "No exercise blanks (____), and no instructions like எழுதுக." |
| [L86](../phase3_tamil_extension/src/step1_extract_sentences.py#L86) | `len(w) <= 2 and w not in SHORT_WORDS` | "No broken word pieces." |
| [L143](../phase3_tamil_extension/src/step1_extract_sentences.py#L143) | `drop_duplicates` | "Repeated sentences are removed." |
| [L144](../phase3_tamil_extension/src/step1_extract_sentences.py#L144) | `df.to_csv(OUT)` | "**Output:** graded_sentences.csv, with 2,351 sentences." |

**Important point to say:** "I remove broken text by its symptoms, **never because a word is rare**. Rare words are what makes text hard."

**Show the output:** `output/graded_sentences.csv`, row 1: class 10, page 15, prose, முழுச்சுரையின் ஓடு…

---

## Step 2: make pairs
**Script:** [step2_build_pairs.py](../phase3_tamil_extension/src/step2_build_pairs.py)

### ❓ Why do we need this step?
- Our test question is **"Which of these two sentences is harder?"**. This is the **same idea as System 2** in the English paper.
- Comparing two sentences is easier and more reliable than giving one sentence an exact difficulty number.
- So we need **pairs of sentences where we already know the answer**: the sentence from the higher class is the harder one.
- **Why at least 3 classes apart?** A Class 5 and a Class 6 sentence can be equally hard, so the label isn't trustworthy. Class 3 vs Class 6 is clearly different.

### ⚙️ How does it work? (5 simple steps)
1. Put each **page** into train (80%) or test (20%). A page never appears in both, so the model is never tested on text it has seen.
2. Take two classes that are at least 3 apart, for example 4 and 9.
3. Pick random sentence pairs, at most 150 per class pair.
4. Randomly put the harder sentence first or second, so the model can't cheat by always saying "B".
5. Mark pairs with the same number of words. These are used to test morphology without the length effect.

**Input:** `graded_sentences.csv`. **Output:** `graded_pairs.csv` (8,004 pairs).

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L29](../phase3_tamil_extension/src/step2_build_pairs.py#L29) | `MIN_GAP = 3` | "Two sentences must be at least 3 classes apart." |
| [L30](../phase3_tamil_extension/src/step2_build_pairs.py#L30) | `PAIRS_PER_CLASS_PAIR = 150` | "At most 150 pairs per class pair, so the big Class 8 book doesn't dominate." |
| [L37](../phase3_tamil_extension/src/step2_build_pairs.py#L37) | `% 100 < TEST_FRACTION * 100` | "Train and test are split by **page**, so the model is never tested on a page it has seen." |
| [L51](../phase3_tamil_extension/src/step2_build_pairs.py#L51) | `if hi - lo < MIN_GAP: continue` | "Skip classes that are too close, like 5 and 6." |
| [L54](../phase3_tamil_extension/src/step2_build_pairs.py#L54) | `n = min(150, ...)` | "Pick random pairs." |
| [L59](../phase3_tamil_extension/src/step2_build_pairs.py#L59)–[L60](../phase3_tamil_extension/src/step2_build_pairs.py#L60) | `flip = ...`, `a, b = ...` | "Randomly put the harder sentence first or second, so the model can't cheat." |
| [L64](../phase3_tamil_extension/src/step2_build_pairs.py#L64) | `"harder": "a" if flip else "b"` | "This saves the right answer." |
| [L66](../phase3_tamil_extension/src/step2_build_pairs.py#L66) | `"length_matched": ... <= 1` | "This marks pairs of the same length, to test morphology without the length effect." |
| [L71](../phase3_tamil_extension/src/step2_build_pairs.py#L71) | `p.to_csv(OUT)` | "**Output:** graded_pairs.csv, with 8,004 pairs." |

**Show the output:** `output/graded_pairs.csv`, row 1: a Class 3 sentence against a Class 6 sentence, harder = b.

---

## Step 3: the 12 features ⭐ (spend the most time here)
**Script:** [step3_textbook_features.py](../phase3_tamil_extension/src/step3_textbook_features.py)

### ❓ Why do we need this step?
- **A computer can't read a sentence and "feel" whether it is hard.** It only understands numbers.
- So we run **12 tests** on every sentence, like a doctor's check-up: each test gives one number.
- **Group A (4 tests):** are the words hard? This is the English-style test from the paper.
- **Group B (6 tests):** are the word parts hard? This is **our new idea for Tamil**, because Tamil words are made of many parts.
- **Group C (2 tests):** is the sentence shape hard?
- Without these numbers, steps 6 to 9 have nothing to work with.

### ⚙️ How does it work? (5 simple steps)
1. Load the 4 tools: **wordfreq** (how common), **Tamil GPT-2** (how surprising), **Morfessor** (split into parts) and **Stanza** (grammar and tree).
2. GPT-2 gives every character a surprisal value.
3. For each word: count its characters, check how common it is, split it into parts, and get the surprisal of the root and of the endings.
4. Take the **average** over all the words of the sentence.
5. Stanza adds the grammar markings, the joining verbs, the tree depth and the word distance. Save one row per sentence.

**Input:** `graded_sentences.csv`. **Output:** `sentence_features.csv` (12 numbers per sentence).

### 💻 The code, line by line

### The 4 tools
| Line | What is there | Say |
|---|---|---|
| [L71](../phase3_tamil_extension/src/step3_textbook_features.py#L71) | `UnsupervisedMorphAnalyzer("ta")` | "**Morfessor** splits a word into parts." |
| [L72](../phase3_tamil_extension/src/step3_textbook_features.py#L72) | `stanza.Pipeline("ta", ...)` | "**Stanza** finds the grammar and the sentence tree." |
| [L74](../phase3_tamil_extension/src/step3_textbook_features.py#L74)–[L75](../phase3_tamil_extension/src/step3_textbook_features.py#L75) | `AutoTokenizer...`, `AutoModelForCausalLM...` | "**Tamil GPT-2** gives the surprisal." |
| [L143](../phase3_tamil_extension/src/step3_textbook_features.py#L143) | `zipf_frequency(w, "ta")` | "**wordfreq** tells how common a word is." |

### GPT-2 surprisal
| Line | What is there | Say |
|---|---|---|
| [L81](../phase3_tamil_extension/src/step3_textbook_features.py#L81) | `def char_surprisal` | "This gives every character a surprisal." |
| [L89](../phase3_tamil_extension/src/step3_textbook_features.py#L89)–[L90](../phase3_tamil_extension/src/step3_textbook_features.py#L90) | `log_softmax`, `surp = -logp ... / log(2)` | "Surprisal = minus log₂ of the probability. Unexpected means a big number." |
| [L93](../phase3_tamil_extension/src/step3_textbook_features.py#L93) | `per_char[end - 1] += s` | "GPT-2's pieces don't match Tamil word parts, so I give each piece's surprisal to its last character. Then I can add up any word, root or ending." |

### Group A and Group B, word by word
| Line | What is there | Feature | Say |
|---|---|---|---|
| [L133](../phase3_tamil_extension/src/step3_textbook_features.py#L133) | `def sentence_features` | | "This calculates all the features for one sentence." |
| [L140](../phase3_tamil_extension/src/step3_textbook_features.py#L140) | `lens.append(len(w))` | 2. word_len | "Count the characters of each word." |
| [L141](../phase3_tamil_extension/src/step3_textbook_features.py#L141) | `w_surp.append(per_char[...].sum())` | 4. word_surprisal | "Add up the surprisal of the word's characters." |
| [L143](../phase3_tamil_extension/src/step3_textbook_features.py#L143) | `zipfs.append(zipf_frequency(w))` | 3. word_zipf | "How common the word is." |
| [L144](../phase3_tamil_extension/src/step3_textbook_features.py#L144) | `morphs = seg.morph_analyze(w)` | | "Split the word, for example ஏரிகள் → ஏரி + கள்." |
| [L149](../phase3_tamil_extension/src/step3_textbook_features.py#L149) | `n_morph.append(len(morphs))` | 5. morphs_per_word | "Count the parts." |
| [L150](../phase3_tamil_extension/src/step3_textbook_features.py#L150) | `stem_end = start + len(morphs[0])` | | "Find where the root ends." |
| [L151](../phase3_tamil_extension/src/step3_textbook_features.py#L151) | `st_surp.append(...)` | 6. stem_surprisal | "Surprisal of the root only." |
| [L152](../phase3_tamil_extension/src/step3_textbook_features.py#L152) | `sf_surp.append(...)` | 7. suffix_surprisal | "Surprisal of the endings only." |
| [L155](../phase3_tamil_extension/src/step3_textbook_features.py#L155) | `"n_words": len(lens)` | 1. n_words | "Count the words." |
| [L156](../phase3_tamil_extension/src/step3_textbook_features.py#L156) | `"word_len": np.mean(lens)` | | "Then take the **average** over all the words." |
| [L163](../phase3_tamil_extension/src/step3_textbook_features.py#L163) | `f.update(parse_features(nlp(sentence)))` | | "Now add the Stanza features." |

### The Stanza features (the rest of group B, and group C)
| Line | What is there | Feature | Say |
|---|---|---|---|
| [L56](../phase3_tamil_extension/src/step3_textbook_features.py#L56) | `UNMARKED = {"Case=Nom", "Number=Sing", ...}` | | "These are default grammar values, so I don't count them." |
| [L60](../phase3_tamil_extension/src/step3_textbook_features.py#L60) | `NONFINITE = {"VerbForm=Part", "VerbForm=Conv", ...}` | | "These are the joining verb forms, like படித்து and வந்த." |
| [L98](../phase3_tamil_extension/src/step3_textbook_features.py#L98) | `def parse_features` | | "This reads Stanza's result." |
| [L106](../phase3_tamil_extension/src/step3_textbook_features.py#L106) | `marked += ...` | 8. feats_per_word | "Count the real grammar markings, like Case=Loc (in) and Number=Plur." |
| [L110](../phase3_tamil_extension/src/step3_textbook_features.py#L110) | `lemma_z.append(zipf_frequency(w.lemma))` | 9. lemma_zipf | "How common the dictionary form is: காட்டில் → காடு." |
| [L107](../phase3_tamil_extension/src/step3_textbook_features.py#L107) | `if NONFINITE & set(feats)` | 10. nonfinite_verbs | "Count the joining verbs." |
| [L118](../phase3_tamil_extension/src/step3_textbook_features.py#L118)–[L120](../phase3_tamil_extension/src/step3_textbook_features.py#L120) | `while h: ...`, `depth = max(...)` | 11. tree_depth | "Walk from each word up to the top of the tree. The longest walk is the depth." |
| [L122](../phase3_tamil_extension/src/step3_textbook_features.py#L122) | `dist.append(abs(w.id - w.head))` | 12. dep_distance | "The distance between a word and the word it connects to." |

### Saving
| Line | What is there | Say |
|---|---|---|
| [L171](../phase3_tamil_extension/src/step3_textbook_features.py#L171) | `for i, sent in enumerate(s.sentence)` | "**Input:** every sentence from graded_sentences.csv." |
| [L177](../phase3_tamil_extension/src/step3_textbook_features.py#L177) | `out.to_csv(OUT)` | "**Output:** sentence_features.csv." |

**Show the output:** `output/sentence_features.csv`, row 1 → `10, 8.5, 3.28, 17.64, 1.4, 14.08, 3.57, 0.5, 3.46, 2, 6, 3.09`

**Explain one value live:** "This sentence has 10 words with 85 characters in total, so word_len = 8.5. Morfessor found 14 parts in 10 words, so morphs_per_word = 1.4."

---

## Step 4: download StoryWeaver
**Script:** [step4_storyweaver_data.py](../phase3_tamil_extension/src/step4_storyweaver_data.py)

### ❓ Why do we need this step?
- After step 6 on the textbooks, we found that **the textbook labels are noisy.**
- The class is given to the **whole book**, not to each sentence. A Class 12 book also has simple sentences, and a Class 3 book can have a hard one.
- Even Class 3 vs Class 12 sentences were only about 52% separable, which is almost guessing.
- So we needed a **second, cleaner dataset**, to make sure our result isn't just caused by bad labels.
- **StoryWeaver** (Pratham Books) gives every children's book a **reading level from 1 to 4**, set by its editors. It is free to use (CC BY 4.0).

### ⚙️ How does it work? (4 simple steps)
1. Ask the StoryWeaver website for all Tamil books, 100 per page.
2. For each book, keep its **level** and its **short description**. The full story needs a login, but the description is written at the same level.
3. Cut the descriptions into sentences and remove repeats.
4. Save them.

**Input:** the StoryWeaver website. **Output:** `storyweaver_sentences.csv` (about 7,300 sentences).

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L32](../phase3_tamil_extension/src/step4_storyweaver_data.py#L32) | `API = "https://storyweaver.org.in/api/..."` | "**Input:** the StoryWeaver website. Every book has a reading level from 1 to 4." |
| [L45](../phase3_tamil_extension/src/step4_storyweaver_data.py#L45) | `books += [{"book_id", "level", "title", "description"}]` | "I take each book's level and its short description." |
| [L49](../phase3_tamil_extension/src/step4_storyweaver_data.py#L49) | `time.sleep(0.5)` | "It waits a little between requests, to be polite to the server." |
| [L68](../phase3_tamil_extension/src/step4_storyweaver_data.py#L68) | `for s in SPLIT.split(text)` | "It cuts the description into sentences." |
| [L77](../phase3_tamil_extension/src/step4_storyweaver_data.py#L77) | `out.to_csv(OUT)` | "**Output:** storyweaver_sentences.csv, about 7,300 sentences." |

**Why say it:** "The textbook labels are noisy, because the class is given per book. StoryWeaver is cleaner."

---

## Step 5: StoryWeaver features and pairs
**Script:** [step5_storyweaver_features.py](../phase3_tamil_extension/src/step5_storyweaver_features.py)

### ❓ Why do we need this step?
- The StoryWeaver sentences need the **same 12 numbers** and the **same kind of pairs** as the textbooks.
- Only then are the two datasets comparable. If the result is the same on both, we can trust it.

### ⚙️ How does it work? (3 simple steps)
1. **Reuse step 3's code** to calculate the 12 features. Nothing is rewritten, so the method is exactly the same.
2. Make pairs with levels at least 2 apart (1–3, 1–4, 2–4).
3. Split train and test by **book**, so one book's sentences are never in both.

**Input:** `storyweaver_sentences.csv`. **Output:** `storyweaver_features.csv` and `storyweaver_pairs.csv` (7,492 pairs).

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L28](../phase3_tamil_extension/src/step5_storyweaver_features.py#L28) | `from step3_textbook_features import ... sentence_features` | "I **reuse** step 3's code, so the features are calculated exactly the same way." |
| [L53](../phase3_tamil_extension/src/step5_storyweaver_features.py#L53) | `sentence_features(sent, ...)` | "Same 12 features." |
| [L36](../phase3_tamil_extension/src/step5_storyweaver_features.py#L36) | `MIN_GAP = 2` | "Levels must be at least 2 apart." |
| [L44](../phase3_tamil_extension/src/step5_storyweaver_features.py#L44) | `book_split` | "Train and test are split by **book**." |
| [L57](../phase3_tamil_extension/src/step5_storyweaver_features.py#L57), [L90](../phase3_tamil_extension/src/step5_storyweaver_features.py#L90) | `to_csv(FEATS)`, `to_csv(PAIRS)` | "**Output:** storyweaver_features.csv and storyweaver_pairs.csv (7,492 pairs)." |

---

## Step 6: which features work? → RESULT 1
**Script:** [step6_evaluate_metric.py](../phase3_tamil_extension/src/step6_evaluate_metric.py)

### ❓ Why do we need this step?
- This step answers the **main research question of Part A:** "Does morphology (group B) help find the harder sentence, **more than** simple word features (group A)?"
- We have the numbers (step 3) and the pairs with answers (step 2). Now we **test** them.

### ⚙️ How does it work? (5 simple steps)
1. For every pair, subtract: **features of B − features of A**.
2. The model (logistic regression) **learns** from the train pairs what these differences mean.
3. It **guesses** on the test pairs: "is B harder?"
4. Count how many guesses are right. That is the **accuracy**, where 50% would be guessing.
5. Repeat for group A only, B only, C only, and A + B, then compare them.

**Input:** the features and pairs files. **Output:** `results/summary.md` (accuracy for each group).

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L46](../phase3_tamil_extension/src/step6_evaluate_metric.py#L46) | `GROUPS = {"A": [...], "B": [...], "C": [...]}` | "The 12 features in 3 groups: words, word parts, sentence shape." |
| [L52](../phase3_tamil_extension/src/step6_evaluate_metric.py#L52) | `MODELS = {...}` | "I test each group alone and in combinations." |
| [L70](../phase3_tamil_extension/src/step6_evaluate_metric.py#L70)–[L71](../phase3_tamil_extension/src/step6_evaluate_metric.py#L71) | `Xa = ...`, `Xb = ...` | "**Input:** the features of sentence A and sentence B in each pair." |
| [L72](../phase3_tamil_extension/src/step6_evaluate_metric.py#L72) | `D = Xb - Xa` | "B minus A. This is the same idea as System 2 in Phase 1." |
| [L73](../phase3_tamil_extension/src/step6_evaluate_metric.py#L73) | `y = (p.harder == "b")` | "The right answer: 1 if B is harder." |
| [L130](../phase3_tamil_extension/src/step6_evaluate_metric.py#L130) | `LogisticRegression(fit_intercept=False)` | "The model learns from the train pairs. No intercept, so swapping A and B flips the answer, which is fair." |
| [L153](../phase3_tamil_extension/src/step6_evaluate_metric.py#L153)–[L154](../phase3_tamil_extension/src/step6_evaluate_metric.py#L154) | `pred = clf.predict(...)`, `c = (pred == y)` | "It predicts on the test pairs and checks whether each prediction is right." |
| [L78](../phase3_tamil_extension/src/step6_evaluate_metric.py#L78) | `class SentenceBootstrap` | "For the confidence interval, I resample **sentences**, not pairs, because one sentence is in many pairs." |
| [L87](../phase3_tamil_extension/src/step6_evaluate_metric.py#L87)–[L88](../phase3_tamil_extension/src/step6_evaluate_metric.py#L88) | `counts = rng.multinomial(...)` | "1,000 resamples." |
| [L190](../phase3_tamil_extension/src/step6_evaluate_metric.py#L190) | `open(... "summary.md")` | "**Output:** summary.md, with all the results." |

**Show the output:** `results/storyweaver/summary.md` → A = 63.8%, B = 60.5%, A + B = 63.4%.

---

## metric_model.py: one score per sentence
**Script:** [metric_model.py](../phase3_tamil_extension/src/metric_model.py)

### ❓ Why do we need this?
- Step 6 compares **two** sentences. But later we need **one difficulty number for each sentence**, to:
  - compare an original sentence with its simplified version (step 8)
  - compare our number with the readers' difficulty rating (step 9)
- We also want to compare **two scores**: the simple English-style one and our Tamil morphology one.

### ⚙️ How does it work? (3 simple steps)
1. Train on the StoryWeaver pairs, the same way as step 6.
2. The model learns **one weight per feature**, for example morphs_per_word gets +0.894.
3. **Score = each feature × its weight, all added up.** A higher score means harder.
   - **word score:** 4 features (group A)
   - **morph score:** 10 features (groups A + B)

**Input:** the StoryWeaver features and pairs. **Output:** two scoring functions, which steps 8 and 9 use.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L29](../phase3_tamil_extension/src/metric_model.py#L29) | `A = [4 features]` | "The word score uses these 4." |
| [L30](../phase3_tamil_extension/src/metric_model.py#L30) | `B = [6 features]` | "The morph score adds these 6." |
| [L32](../phase3_tamil_extension/src/metric_model.py#L32) | `METRICS = {"word": A, "morph": A + B}` | "So there are two scores to compare." |
| [L43](../phase3_tamil_extension/src/metric_model.py#L43) | `D = f.loc[sent_b] - f.loc[sent_a]` | "It's trained on the StoryWeaver pairs, the same way as step 6." |
| [L45](../phase3_tamil_extension/src/metric_model.py#L45)–[L46](../phase3_tamil_extension/src/metric_model.py#L46) | `LogisticRegression ... coef_` | "It learns one weight per feature." |
| [L54](../phase3_tamil_extension/src/metric_model.py#L54) | `features[cols] @ w` | "**Score = each feature × its weight, all added up.** Higher means harder." |

**Example to say:** "The Class 10 sentence gets +0.63 and the Class 3 sentence gets −1.18. So the model says the Class 10 sentence is harder."

---

## Step 7: pick 60 hard sentences
**Script:** [step7_select_complex.py](../phase3_tamil_extension/src/step7_select_complex.py)

### ❓ Why do we need this step?
- The project's **second goal** is to **reduce** difficulty, by making hard sentences easier with an AI.
- So first we need a set of **hard sentences** to simplify.
- **Why at random?** If we picked the sentences our own score calls "hardest", the test would favour that score. Random is fair to both scores.
- **Why prompt files?** We have no paid AI account, so we paste the prompts into a free AI chat. The script writes them ready to paste.

### ⚙️ How does it work? (4 simple steps)
1. Take only the hardest level: textbook Class 11–12 and StoryWeaver level 4.
2. Keep clean sentences of 10–30 words (no numbers, no English, no broken words).
3. Pick 30 from each **at random**, so 60 in total, named S01 to S60.
4. Write the AI instruction and the sentences into 3 prompt files.

**Input:** the two feature files. **Output:** `original.csv` (60 sentences) and `prompt_batch1.txt`, `prompt_batch2.txt`, `prompt_batch3.txt`.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L83](../phase3_tamil_extension/src/step7_select_complex.py#L83) | `t["class"] >= 11` | "**Input:** textbook sentences from Class 11 and 12 only." |
| [L84](../phase3_tamil_extension/src/step7_select_complex.py#L84) | `s.level == 4` | "And StoryWeaver level 4, the hardest." |
| [L88](../phase3_tamil_extension/src/step7_select_complex.py#L88) | `n_words.between(10, 30) & clean(...)` | "Only clean sentences with 10 to 30 words." |
| [L90](../phase3_tamil_extension/src/step7_select_complex.py#L90) | `d.sample(30, random_state=42)` | "30 from each, **chosen at random**, not by my score, so the test is fair." |
| [L35](../phase3_tamil_extension/src/step7_select_complex.py#L35) | `PROMPT = """You are an expert Tamil teacher..."""` | "This is the instruction I gave the AI: rewrite for a Class 3–5 child, keep the meaning, use common words." |
| [L96](../phase3_tamil_extension/src/step7_select_complex.py#L96) | `to_csv("original.csv")` | "**Output:** 60 sentences, S01 to S60." |
| [L100](../phase3_tamil_extension/src/step7_select_complex.py#L100) | `prompt_batch{N}.txt` | "And 3 prompt files to paste into the AI chat." |

**Show:** `Dataset/complex_sentences/prompt_batch1.txt`, then `simplified_claude-opus-5-5.txt` (the AI's answers).

---

## Step 8: score before and after, and make the forms
**Script:** [step8_score_simplified.py](../phase3_tamil_extension/src/step8_score_simplified.py)

### ❓ Why do we need this step?
- The AI has given us simplified sentences. Now we need to know **two things**:
  1. **Do our scores see the simplified version as easier?** The score should go down.
  2. **Do real people agree?** For that we need a survey (Google Forms).
- Making 60 survey questions by hand is slow and easy to get wrong. So the script **creates the forms automatically**.
- **Why random A/B?** If the original was always A, readers would notice the pattern and stop reading carefully.

### ⚙️ How does it work? (5 simple steps)
1. Read the AI's answers (lines like "S13 | …").
2. Calculate the 12 features for the original and the simplified version. If the AI split one sentence into two, take the **average**.
3. Give both versions a **word score** and a **morph score**.
4. Write a summary: did the scores go down?
5. Build the Google Forms. The original is randomly shown as A or B, and a secret answer key is saved.

**Input:** `original.csv` and the AI's answers. **Output:** `simplification_scores.csv`, `simplification_summary.md`, `make_google_form.gs` and `items_key.csv`.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L40](../phase3_tamil_extension/src/step8_score_simplified.py#L40) | `LINE = re.compile(... S\d{2} ...)` | "**Input:** the AI's answers. This reads lines like 'S13 \| …'." |
| [L194](../phase3_tamil_extension/src/step8_score_simplified.py#L194) | `metrics = fit_metrics()` | "It loads the two scores from metric_model." |
| [L203](../phase3_tamil_extension/src/step8_score_simplified.py#L203)–[L204](../phase3_tamil_extension/src/step8_score_simplified.py#L204) | `fo = text_features(original)`, `fs = text_features(simplified)` | "It calculates the 12 features for the original and the simplified sentence." |
| [L63](../phase3_tamil_extension/src/step8_score_simplified.py#L63) | `def text_features` | "If the AI split one sentence into two…" |
| [L66](../phase3_tamil_extension/src/step8_score_simplified.py#L66) | `parts = SENT_END.split(text)` | "…it cuts the text into its sentences…" |
| [L72](../phase3_tamil_extension/src/step8_score_simplified.py#L72) | `.mean()` | "…and takes the **average**. Otherwise two sentences would look harder just because they have more words." |
| [L211](../phase3_tamil_extension/src/step8_score_simplified.py#L211)–[L213](../phase3_tamil_extension/src/step8_score_simplified.py#L213) | `for side in ["orig", "simp"]: score(...)` | "Both versions get a word score and a morph score." |
| [L93](../phase3_tamil_extension/src/step8_score_simplified.py#L93) | `drop = score_orig - score_simp` | "If the drop is positive, the score says the text got easier." |
| [L131](../phase3_tamil_extension/src/step8_score_simplified.py#L131) | `a_is_orig = rng.random() < 0.5` | "In the form, the original is randomly shown as A or as B, so readers can't guess." |
| [L138](../phase3_tamil_extension/src/step8_score_simplified.py#L138) | `to_csv("items_key.csv")` | "The secret answer key: which side was the original." |
| [L142](../phase3_tamil_extension/src/step8_score_simplified.py#L142) | `make_google_form.gs` | "It writes code that **creates the Google Forms automatically**." |
| [L147](../phase3_tamil_extension/src/step8_score_simplified.py#L147) | `GS_TEMPLATE = ...` | "These are the 5 questions: easier? difficulty of A, difficulty of B, same meaning? errors?" |
| [L215](../phase3_tamil_extension/src/step8_score_simplified.py#L215) | `df.to_csv(OUT)` | "**Output:** simplification_scores.csv." |

**Show the output:** `output/simplification_scores.csv`, the S13 row → word score +0.11 → −0.64, morph score +0.60 → −0.24. "Both went down, so both say it got easier."

---

## Step 9: compare the readers with the scores → RESULT 2
**Script:** [step9_human_eval.py](../phase3_tamil_extension/src/step9_human_eval.py)

### ❓ Why do we need this step?
- In step 8, **both scores said all 60 simplifications are easier.** They agree on everything, so the scores alone **can't tell us which one is better.**
- Only **real Tamil readers** can tell us how hard a sentence really feels.
- So this step asks: **which score agrees more with real readers, word or morph?** This is the **main thesis question**.
- It also checks whether the AI's simplifications **really helped** and **kept the meaning**.

### ⚙️ How does it work? (5 simple steps)
1. Read the Google Form answers. Keep only native and fluent Tamil readers.
2. Use the answer key to turn the readers' "A/B" answers into "original/simplified".
3. Average the about 6 readers of each item.
4. Compare our scores with the readers' difficulty (Spearman correlation). Is morph better than word?
5. Check whether the readers agree with each other, and write the summary.

**Input:** `form1.csv`, `form2.csv`, `items_key.csv` and `simplification_scores.csv`. **Output:** `human_eval_summary.md`.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L50](../phase3_tamil_extension/src/step9_human_eval.py#L50) | `def read_responses` | "**Input:** form1.csv and form2.csv from Google Forms." |
| [L43](../phase3_tamil_extension/src/step9_human_eval.py#L43), [L63](../phase3_tamil_extension/src/step9_human_eval.py#L63) | `KEEP_LEVELS = ("Native", "Fluent")` | "Only native and fluent Tamil readers are kept." |
| [L73](../phase3_tamil_extension/src/step9_human_eval.py#L73) | `.merge(key, ...)` | "It joins the answers with the answer key." |
| [L76](../phase3_tamil_extension/src/step9_human_eval.py#L76)–[L78](../phase3_tamil_extension/src/step9_human_eval.py#L78) | `diff_orig = ...`, `diff_simp = ...` | "Readers answered about A and B. The key turns that into original and simplified." |
| [L129](../phase3_tamil_extension/src/step9_human_eval.py#L129) | `it = d.groupby("id").agg(...)` | "It averages the about 6 readers of each item." |
| [L91](../phase3_tamil_extension/src/step9_human_eval.py#L91) | `spearmanr(score, human)` | "**The main question:** does our score go up when the readers' difficulty goes up?" |
| [L99](../phase3_tamil_extension/src/step9_human_eval.py#L99) | `diff = draws["morph"] - draws["word"]` | "Is morph better than word? This is the difference, with a confidence interval." |
| [L111](../phase3_tamil_extension/src/step9_human_eval.py#L111) | `def rater_agreement` | "Do the readers agree with each other?" |
| [L126](../phase3_tamil_extension/src/step9_human_eval.py#L126), [L176](../phase3_tamil_extension/src/step9_human_eval.py#L176) | `d.to_csv(LONG)`, `open(SUMMARY)` | "**Output:** human_ratings_long.csv and human_eval_summary.md." |

**Show the output:** `results/human_eval_summary.md` → 49% vs 29%, 2.31 → 1.97, word +0.32, morph +0.28, difference −0.04 with a CI that includes 0.
⚠️ **Don't open** `human_ratings_long.csv` on screen. It has the readers' names.

---

## Step 10: Gemini as a judge → RESULT 3
**Script:** [step10_llm_judge.py](../phase3_tamil_extension/src/step10_llm_judge.py)

### ❓ Why do we need this step?
- Our human study is **small** (12 readers), and the readers did not agree much with each other.
- So we wanted a **second opinion**: we asked an AI (Gemini) the exact same questions.
- This also tests a useful question for researchers: **"Can an AI judge replace human readers?"**
- Gemini is **kept separate** and is never counted as a human reader.

### ⚙️ How does it work? (4 simple steps)
1. Read the **exact** A and B texts that the humans saw.
2. Send them to the free Gemini API with the same questions. Gemini must answer in a fixed format.
3. If the free limit is reached, wait and try again. Save the answers as you go.
4. Compare Gemini's answers with the average human.

**Input:** the forms file and the human answers. **Output:** `llm_judge_gemini-3.5-flash-lite.md`.

### 💻 The code, line by line

| Line | What is there | Say |
|---|---|---|
| [L80](../phase3_tamil_extension/src/step10_llm_judge.py#L80) | `forms = json.loads(... "var FORMS" ...)` | "**Input:** the exact A and B texts the humans saw, read from the forms file." |
| [L47](../phase3_tamil_extension/src/step10_llm_judge.py#L47) | `PROMPT = ...` | "The same instructions and the same questions as the human form." |
| [L59](../phase3_tamil_extension/src/step10_llm_judge.py#L59) | `SCHEMA = {...}` | "Gemini must answer in a fixed format: easier, difficulty, meaning, errors." |
| [L88](../phase3_tamil_extension/src/step10_llm_judge.py#L88) | `"temperature": 0` | "Temperature 0 gives the same answer every time." |
| [L91](../phase3_tamil_extension/src/step10_llm_judge.py#L91) | `requests.post(URL ...)` | "This sends the items to the free Gemini API." |
| [L92](../phase3_tamil_extension/src/step10_llm_judge.py#L92)–[L96](../phase3_tamil_extension/src/step10_llm_judge.py#L96) | `status_code == 429`, `time.sleep(60...)` | "If the free limit is reached, it waits and tries again." |
| [L114](../phase3_tamil_extension/src/step10_llm_judge.py#L114)–[L117](../phase3_tamil_extension/src/step10_llm_judge.py#L117) | `cache = ...`, `todo = ...` | "Answers are saved, so running it again only asks the missing ones." |
| [L173](../phase3_tamil_extension/src/step10_llm_judge.py#L173) | `spearmanr(gemini, humans)` | "It compares Gemini with the average human." |

**Show the output:** `results/llm_judge_gemini-3.5-flash-lite.md` → Gemini vs humans +0.30, but it says 90% got easier (humans: 49%).

---

# The final output: open these at the end

| Result | File | One line to say |
|---|---|---|
| 1 | [results/storyweaver/summary.md](../phase3_tamil_extension/results/storyweaver/summary.md) | "Word-level is best at 63.8%. Morphology adds nothing extra." |
| 2 | [results/simplification_summary.md](../phase3_tamil_extension/results/simplification_summary.md) | "Both scores say all 60 AI simplifications are easier." |
| 3 | [results/human_eval_summary.md](../phase3_tamil_extension/results/human_eval_summary.md) | "Readers agree the text got easier. Morph is not better than word." |
| 4 | [results/llm_judge_gemini-3.5-flash-lite.md](../phase3_tamil_extension/results/llm_judge_gemini-3.5-flash-lite.md) | "Gemini helps, but it can't replace humans." |

**Final sentence:**
> "Morphology does carry difficulty information in Tamil, but word length already contains it. The pair test and the human study give the same answer."

---

## If you have only 5 minutes, show just these 6 lines

1. [step1 L34](../phase3_tamil_extension/src/step1_extract_sentences.py#L34): keep only prose
2. [step2 L29](../phase3_tamil_extension/src/step2_build_pairs.py#L29): pairs at least 3 classes apart
3. [step3 L144](../phase3_tamil_extension/src/step3_textbook_features.py#L144): Morfessor splits the word into parts
4. [step6 L72](../phase3_tamil_extension/src/step6_evaluate_metric.py#L72): features of B − features of A
5. [metric_model L54](../phase3_tamil_extension/src/metric_model.py#L54): score = features × weights
6. [step9 L91](../phase3_tamil_extension/src/step9_human_eval.py#L91): compare the score with the readers

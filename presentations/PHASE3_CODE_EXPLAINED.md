# Phase 3 code, step by step: input → code → output

For every step you get:
- **INPUT:** which file goes in, with a real example
- **CODE:** the important lines from the script, with a plain-English explanation
- **OUTPUT:** which file comes out, with a real example

All the scripts are in `phase3_tamil_extension/src/`. Run them from the `phase3_tamil_extension` folder, for example `python src/step1_extract_sentences.py`.

```
PART A   step0 → step1 → step2 → step3 → (step4 → step5) → step6
PART B   step7 → [AI chat, by hand] → step8
PART C   [Google Forms, by hand] → step9 → step10
```

---

# PART A: build the data and test the features

## Step 0: download the textbooks
**File:** `src/step0_fetch_textbooks.py`

### INPUT
A list of Google Drive links to the TN Tamil textbooks, written inside the code:

```python
CANDIDATES = {
    3:  ("Term 1", ["1LK0e6FLB3KdaVCnMwhiMYKhCqa5O6YBs", ...], "https://www.tntextbooks.net/..."),
    ...
    12: ("Full year", ["1syhbydqEPDllh66-RvSuvbqAfJmCQncx", ...], "https://www.tntextbooks.net/..."),
}
```
**In simple words:** for each class (3 to 12), the code has one or more possible download links.

### CODE
**1. Download the file.**
```python
def download(file_id, path):
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"
```
It downloads the file, then checks that the first 5 bytes are `%PDF-`, which shows it really is a PDF.

**2. Check that it is the right class.**
```python
ORDINAL = {3: ["மூன்றாம் வகுப்பு"], 6: ["ஆறாம் வகுப்பு"], ... 12: ["இரண்டாம் ஆண்டு", "XII Std"]}

def names_class(path, n, pages=6):
    text = repair_pdf_text(...first 6 pages...)
    return any(m in text for m in ORDINAL[n])
```
It reads the first 6 pages. If the Class 6 book does not say "ஆறாம் வகுப்பு", it is the wrong book, so it is deleted.

**3. Keep the right file.**
```python
if match:
    os.replace(tmp, final)        # rename to class6.pdf
```

### OUTPUT
`Dataset/graded_textbooks/class3.pdf … class12.pdf` (10 PDFs) and `manifest.csv`, a list of what was downloaded.

---

## Helper: fixing broken Tamil text
**File:** `src/tamil_text.py`

When we copy text out of the textbook PDFs, the Tamil letters come out broken. This helper fixes them, and step 0 and step 1 use it.

| Broken (from the PDF) | Fixed |
|---|---|
| அடித்்தளம் | அடித்தளம் (the pulli ் was doubled) |
| தொொடக்்கமும் | தொடக்கமும் (the vowel sign was doubled) |
| கிரா மத்தில் | கிராமத்தில் (one word was split by a space) |

```python
def repair_pdf_text(text):
    text = _DOUBLE_PULLI_CONS.sub(...)   # ்் + same consonant twice → one
    text = _DOUBLE_PULLI.sub(PULLI, text)  # ்் → ்
    return _DOUBLE_SIGN.sub(...)         # ொொ → ொ
```
**Result:** real Tamil words went from **55% to 76.5%**.

`merge_split_words()` joins split words back together. It only joins them if the joined word is a real Tamil word and the pieces are not, so two real words are never glued together.

---

## Step 1: take the sentences out of the PDFs
**File:** `src/step1_extract_sentences.py`

### INPUT
The 10 PDFs from step 0.

### CODE
**1. Read every page and fix the text.**
```python
pages = [repair_pdf_text(pdf[i].get_textpage().get_text_range()) for i in range(len(pdf))]
```

**2. Skip bad pages.**
```python
toc = next(i for i, t in enumerate(pages) if "பொருளடக்கம்" in t)       # table of contents
usable = [(i, t) for i, t in enumerate(pages) if i > toc and page_known(t) >= 0.6]
```
- Pages before the table of contents (the preface and anthems) are skipped.
- Pages where less than 60% of the words are real are skipped, because the font is broken on those pages.

**3. Find the sections by their headings.**
```python
SECTION_START = {"கவிதைப்பேழை": "poetry", "உரைநடை உலகம்": "prose",
                 "விரிவானம்": "extra_reading", "கற்கண்டு": "grammar"}
KEEP = {"prose", "extra_reading"}
SECTION_END = ["மதிப்பீடு", "கற்பவை கற்றபின்", "நூல் வெளி"]
```
Every Tamil textbook lesson starts with a fixed heading:
- **KEEP:** உரைநடை உலகம் (prose) and விரிவானம் (stories)
- **SKIP:** கவிதைப்பேழை (poetry, which uses old Tamil) and கற்கண்டு (grammar)
- The prose ends where the exercises (மதிப்பீடு) start.

**4. Cut the text into sentences.**
```python
SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")      # a new sentence starts after . ? !
```

**5. Keep only clean sentences** (`is_clean()`):
```python
if not 4 <= len(words) <= 40:            return False   # too short or too long
if "�" in sent or INVALID_SEQ.search(sent): return False   # broken characters
if EXERCISE.search(sent):                 return False   # blanks "____", options (அ) (ஆ)
if INSTRUCTION_END.search(sent):          return False   # "எழுதுக", "கூறுக" = exercise instructions
```

**Important rule:** we remove broken text by its **symptoms** (broken characters, blanks), **never because a word is rare**. Rare words are exactly what makes text hard, so removing them would make the test unfair.

**6. Save the sentences.**
```python
df = pd.DataFrame(rows, columns=["class", "page", "section", "sentence", "n_words"])
df = df.drop_duplicates(subset="sentence")
df.to_csv(OUT)
```

### OUTPUT
`output/graded_sentences.csv`, with **2,351 sentences**.

Real row:
| class | page | section | sentence | n_words |
|---|---|---|---|---|
| 10 | 15 | prose | முழுச்சுரையின் ஓடு, குடுக்கை என்றும் தேங்காய் நெற்றின் மேற்பகுதி, மட்டை என்றும் வழங்கப்படுகின்றன. | 10 |

---

## Step 2: make pairs
**File:** `src/step2_build_pairs.py`

### INPUT
`output/graded_sentences.csv` from step 1.

### CODE
**1. Split into train and test by page.**
```python
def page_split(cls, page):
    return "test" if zlib.crc32(f"{cls}-{page}".encode()) % 100 < 20 else "train"
```
About 20% of the **pages** go to test. All sentences from one page stay on the same side, so the model is never tested on a page it has already seen.

**2. Pair classes that are at least 3 apart.**
```python
MIN_GAP = 3
for lo in classes:
    for hi in classes:
        if hi - lo < MIN_GAP: continue        # skip Class 5 vs Class 6 (too close)
```
Neighbouring classes are too similar to trust which one is harder, so we use only pairs like Class 3 vs 6, or Class 4 vs 12.

**3. Pick random sentence pairs, and shuffle which side is harder.**
```python
n = min(150, len(easy) * len(hard))     # at most 150 pairs per class pair
flip = rng.random() < 0.5
a, b = (h, e) if flip else (e, h)
"harder": "a" if flip else "b"
```
The harder sentence is randomly put in place A or place B, so a model can't cheat by always saying "B".

**4. Mark pairs of the same length.**
```python
"length_matched": abs(a.n_words - b.n_words) <= 1
```
These pairs are used later to test whether morphology helps **when length is equal**.

### OUTPUT
`output/graded_pairs.csv`, with **8,004 pairs**.

Real row:
| sent_a | class_a | sent_b | class_b | harder | gap | length_matched | split |
|---|---|---|---|---|---|---|---|
| கரடி எந்தெந்த விலங்குகள் தகுதியற்றவை எனக் கூறியது? | 3 | இவர் 1920ஆம் ஆண்டு நாடகம் ஒன்றை எழுதினார். | 6 | b | 3 | True | train |

---

## Step 3: calculate the 12 features ⭐
**File:** `src/step3_textbook_features.py`

### INPUT
`output/graded_sentences.csv` (2,351 sentences).

### CODE
**1. Load the 4 tools** (`load_tools()`):
```python
seg = unsupervised_morph.UnsupervisedMorphAnalyzer("ta")              # Morfessor: splits words into parts
nlp = stanza.Pipeline("ta", processors="tokenize,mwt,pos,lemma,depparse")  # Stanza: grammar + tree
tok = AutoTokenizer.from_pretrained("abinayam/gpt-2-tamil")          # Tamil GPT-2
lm  = AutoModelForCausalLM.from_pretrained("abinayam/gpt-2-tamil")
```
**wordfreq** (how common a word is) is the fourth tool. It is used directly.

**2. GPT-2 surprisal for every character** (`char_surprisal()`):
```python
logp = torch.log_softmax(logits, -1)
surp = -logp[...] / math.log(2)          # surprisal = −log₂ P(token)
for s, (_, end) in zip(surp, enc.offset_mapping):
    per_char[end - 1] += s               # give the token's surprisal to its last character
```
Tamil GPT-2 cuts a word into small pieces (tokens), and those pieces don't match the real word parts. So each piece's surprisal is given to **one character**. Later we can add up the characters of any word, root or ending.

**3. Word features and morphology features, word by word** (`sentence_features()`):
```python
for m in WORD.finditer(sentence):
    w = m.group()
    lens.append(len(w))                                   # feature 2: word length
    w_surp.append(per_char[start:end].sum())              # feature 4: word surprisal
    zipfs.append(zipf_frequency(w, "ta"))                 # feature 3: how common
    morphs = seg.morph_analyze(w)                         # Morfessor split
    n_morph.append(len(morphs))                           # feature 5: number of parts
    stem_end = start + len(morphs[0])
    st_surp.append(per_char[start:stem_end].sum())        # feature 6: surprisal of the root
    sf_surp.append(per_char[stem_end:end].sum())          # feature 7: surprisal of the endings
```
Then it takes the **average** over all the words:
```python
"n_words": len(lens),                       # feature 1
"word_len": np.mean(lens), ...
```

**4. Grammar and tree features from Stanza** (`parse_features()`):
```python
UNMARKED = {"Case=Nom", "Number=Sing", "Person=3", ...}         # defaults: not counted
NONFINITE = {"VerbForm=Part", "VerbForm=Conv", "VerbForm=Inf", ...}

marked += sum(1 for f in feats if f not in UNMARKED)    # feature 8: grammar markings
if NONFINITE & set(feats): nonfinite += 1               # feature 10: joining verbs
lemma_z.append(zipf_frequency(w.lemma, "ta"))           # feature 9: how common the dictionary form is

# feature 11: tree depth = steps from the word up to the top
while h: h = head[h]; d += 1
# feature 12: distance between a word and the word it connects to
dist.append(abs(w.id - w.head))
```

### Worked example: how one row is made
**Sentence:** முழுச்சுரையின் ஓடு, குடுக்கை என்றும் தேங்காய் நெற்றின் மேற்பகுதி, மட்டை என்றும் வழங்கப்படுகின்றன.

| Word | Length | Zipf (how common) | Morfessor parts |
|---|---|---|---|
| முழுச்சுரையின் | 14 | 0.0 | முழுச் + சுரை + யின் (3) |
| ஓடு | 3 | 4.17 | 1 |
| குடுக்கை | 8 | 0.0 | 1 |
| என்றும் | 7 | 6.05 | 1 |
| தேங்காய் | 8 | 4.63 | 1 |
| நெற்றின் | 8 | 0.0 | நெற்றி + ன் (2) |
| மேற்பகுதி | 9 | 3.93 | 1 |
| மட்டை | 5 | 3.92 | 1 |
| என்றும் | 7 | 6.05 | 1 |
| வழங்கப்படுகின்றன | 16 | 4.07 | வழங்க + ப்படுகின்றன (2) |
| **Average** | **85 ÷ 10 = 8.5** | **3.28** | **14 ÷ 10 = 1.4** |

### OUTPUT
`output/sentence_features.csv`, one row per sentence: the class, the sentence, and the 12 numbers.

| n_words | word_len | word_zipf | word_surprisal | morphs_per_word | stem_surprisal | suffix_surprisal | feats_per_word | lemma_zipf | nonfinite_verbs | tree_depth | dep_distance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 8.5 | 3.28 | 17.64 | 1.4 | 14.08 | 3.57 | 0.5 | 3.46 | 2 | 6 | 3.09 |

---

## Step 4: download the StoryWeaver data
**File:** `src/step4_storyweaver_data.py`

**Why:** the textbook labels were too noisy, because the class is given to the whole book, not to each sentence. StoryWeaver gives every book a **reading level from 1 to 4**.

### INPUT
The StoryWeaver website, through its public search API:
```python
API = "https://storyweaver.org.in/api/v1/books-search?languages%5B%5D=Tamil&per_page=100&page={}"
```

### CODE
**1. Download the list of Tamil books, page by page.**
```python
books += [{"book_id": b["id"], "level": b["level"], "title": b["title"],
           "description": b.get("description")} for b in d["data"]]
time.sleep(0.5)        # wait a little between requests, to be polite to the server
```
The full story needs a login, so we use each book's **short description**. It is written at the same level as the book.

**2. Cut the descriptions into sentences, and remove repeats.**
```python
for s in SPLIT.split(text):
    if len(s.split()) < 3 or s in seen: continue
    rows.append({"book_id": ..., "level": int(b["level"]), "title": ..., "sentence": s})
```

### OUTPUT
`output/storyweaver_sentences.csv`, with about **7,300 sentences**.

Real row:
| book_id | level | title | sentence |
|---|---|---|---|
| 378 | 1 | என் நண்பர்கள் | எனக்கு பல நண்பர்கள் உள்ளனர். |

---

## Step 5: StoryWeaver features and pairs
**File:** `src/step5_storyweaver_features.py`

### INPUT
`output/storyweaver_sentences.csv`.

### CODE
**1. Reuse step 3's code.** Nothing is copied:
```python
from step3_textbook_features import load_tools, sentence_features
rows.append(sentence_features(sent, seg, nlp, tok, lm))
```
So the 12 features are calculated **exactly the same way** as for the textbooks.

**2. Make the pairs, the same way as step 2, with two differences:**
```python
MIN_GAP = 2                     # levels 1–3, 1–4 and 2–4
def book_split(book_id): ...    # train/test split by BOOK
```

### OUTPUT
- `output/storyweaver_features.csv`: the 12 features per sentence (about 7,080 sentences)
- `output/storyweaver_pairs.csv`: **7,492 pairs**

---

## Step 6: which features can find the harder sentence? → RESULT 1
**File:** `src/step6_evaluate_metric.py`

### INPUT
- the features: `output/sentence_features.csv` (or StoryWeaver's)
- the pairs: `output/graded_pairs.csv` (or StoryWeaver's)

### CODE
**1. Put the features into 3 groups.**
```python
GROUPS = {
    "A": ["n_words", "word_len", "word_zipf", "word_surprisal"],
    "B": ["morphs_per_word", "stem_surprisal", "suffix_surprisal",
          "feats_per_word", "lemma_zipf", "nonfinite_verbs"],
    "C": ["tree_depth", "dep_distance"],
}
```

**2. For each pair: features of B − features of A.**
```python
Xa = f.loc[p.sent_a, all_feats]
Xb = f.loc[p.sent_b, all_feats]
D  = Xb - Xa                           # the difference
y  = (p.harder == "b")                 # the answer: 1 if B is harder
```
**Example:** if B has 10 words and A has 4, then the difference is +6. A big positive difference suggests B is harder.

**3. Learn on train and test on test, for each group.**
```python
clf = LogisticRegression(fit_intercept=False).fit(Xtr / scale, y[tr])
pred = clf.predict(D[cols] / scale)
correct = (pred == y)
accuracy = correct[test].mean()
```
`fit_intercept=False` means no bias. If you swap A and B, the answer also flips, which is fair.

**4. Confidence intervals: is the result reliable?**
```python
class SentenceBootstrap: ...   # resample SENTENCES 1,000 times, not pairs
```
One sentence appears in many pairs. If we resampled pairs, the result would look more certain than it really is.

**5. Write all the results.**

### OUTPUT
- `results/summary.md` (textbooks) and `results/storyweaver/summary.md`
- `model_table.csv`, `feature_table.csv`, `comparison_table.csv`, `coefficients.csv`, and the figure `fig_gap.png`

**Result 1 (StoryWeaver):**
| Model | Accuracy | Same-length pairs |
|---|---|---|
| A: word-level | **63.8%** | 57.3% |
| B: morphology | 60.5% | **59.7%** |
| A + B | 63.4% | 57.3% |

**Meaning:** morphology works by itself, but **adding it to word-level gives nothing extra**.

---

## metric_model.py: one difficulty score per sentence
**File:** `src/metric_model.py`

### INPUT
`output/storyweaver_features.csv` and `output/storyweaver_pairs.csv` (train pairs only).

### CODE
```python
A = ["n_words", "word_len", "word_zipf", "word_surprisal"]
B = ["morphs_per_word", "stem_surprisal", "suffix_surprisal", "feats_per_word", "lemma_zipf", "nonfinite_verbs"]
METRICS = {"word": A, "morph": A + B}

clf = LogisticRegression(fit_intercept=False).fit(D / scale, y)   # same as step 6
weights = clf.coef_[0] / scale

def score(features, metrics):
    return features[cols] @ w          # each feature × its weight, all added up
```

### OUTPUT
No file of its own. It gives 2 scores that steps 8 and 9 use:
- **word score**, from 4 features
- **morph score**, from 10 features (its biggest weight is morphs_per_word = +0.894)

**Example:** the Class 10 sentence above gets a morph score of **+0.63**, and the Class 3 sentence "தண்ணீர் நமது உயிர் நீர்!" gets **−1.18**. Higher means harder.

---

# PART B: simplify the hard sentences

## Step 7: pick 60 hard sentences and write the AI prompts
**File:** `src/step7_select_complex.py`

### INPUT
`output/sentence_features.csv` and `output/storyweaver_features.csv`.

### CODE
**1. Take only the hardest level of each dataset.**
```python
t = t[t["class"] >= 11]         # textbooks: Class 11–12
s = s[s.level == 4]             # StoryWeaver: level 4
```

**2. Keep only clean sentences of 10–30 words** (no numbers, no English, no quotation marks, no broken words).

**3. Pick 30 from each at random.**
```python
d.sample(30, random_state=42)
```
It picks **at random, not by our score**, so the test is fair to both scores.

**4. Write the AI instructions (the prompt).**
```python
PROMPT = """You are an expert Tamil teacher. Rewrite each Tamil sentence below so that a child
in Class 3-5 (about 8-10 years old) can read and understand it easily.
Rules: 1. Keep the same meaning ... 2. Use common, everyday Tamil words ...
3. Use shorter words ... 4. You may split one long sentence into two or three ...
Output format: S01 | <simplified Tamil text>"""
```

### OUTPUT
- `Dataset/complex_sentences/original.csv`: 60 sentences with ids S01 to S60
- `prompt_batch1.txt`, `prompt_batch2.txt`, `prompt_batch3.txt`: 20 sentences each, ready to paste

Real row from `original.csv`:
| id | source | level | sentence |
|---|---|---|---|
| S01 | textbook | 12 | இதில் அவளைப் பார்க்க வந்தவர், அவள் விருப்பத்திற்கு மாறாக அங்கிருந்து புறப்பட்டுச் செல்கிறார் என்ற நிகழ்வு, அப்பெண்ணின் முகபாவனை மூலமே சொல்லப்படுகிறது. |

---

## Done by hand: the AI simplifies

### INPUT
`prompt_batch1.txt`, `prompt_batch2.txt` and `prompt_batch3.txt`.

### WHAT WE DID
We pasted each prompt into a free AI chat (Claude) and saved its answers into one text file.

### OUTPUT
`Dataset/complex_sentences/simplified_claude-opus-5-5.txt`
```
S13 | ஒரு பெரிய காடு இருந்தது. அதன் ஒரு தூரமான பகுதியில் நிறைய ஏரிகளும் குளங்களும் ஓடைகளும் இருந்தன.
```

---

## Step 8: score before and after, and make the Google Forms
**File:** `src/step8_score_simplified.py`

### INPUT
- `Dataset/complex_sentences/original.csv`: the 60 originals
- `Dataset/complex_sentences/simplified_claude-opus-5-5.txt`: the AI's answers

### CODE
**1. Read the AI's answers.**
```python
LINE = re.compile(r"^[\s*#>-]*(S\d{2})[\s*]*[|:\-–]\s*(.+?)\s*$")
```
It finds lines like `S13 | ...`, even if the AI added extra symbols such as `**`.

**2. Calculate the 12 features. If the AI split one sentence into two, take the average.**
```python
def text_features(text, tools, cache):
    parts = SENT_END.split(text)                       # split into sentences
    rows = [sentence_features(p, *tools) for p in parts]   # same step 3 code
    return pd.DataFrame(rows).mean()                   # AVERAGE
```
**Why the average?** Two short sentences together have more words than one. Without averaging, a split would look **harder**, which is wrong.

**3. Give both versions a word score and a morph score.**
```python
for side in ["orig", "simp"]:
    score(X, metrics)       # from metric_model.py
```

**4. Write the summary.** Did the score go **down** (meaning easier)?
```python
drop = score_orig - score_simp       # > 0 means the simplified version is easier
```

**5. Make the Google Forms automatically.**
```python
a_is_orig = rng.random() < 0.5                         # randomly put the original as A or B
rows.append({"form": ..., "item": k, "id": r.id, "A_is": "original" or "simplified"})
```
It writes a **Google Apps Script** (`make_google_form.gs`). Pasting it into script.google.com creates the forms with all 5 questions. It also saves a secret answer key (`items_key.csv`), so we know which side was the original.

### OUTPUT
| File | What is in it |
|---|---|
| `output/simplification_scores.csv` | 60 rows: original, simplified, the 12 features before and after, both scores before and after |
| `results/simplification_summary.md` | did each score see the text as easier? |
| `Dataset/human_study/make_google_form.gs` | the code that creates the 2 forms (30 items each) |
| `Dataset/human_study/items_key.csv` | the answer key: which side was the original |

Real example, S13:
| | Original | Simplified |
|---|---|---|
| words per sentence | 11 | 6.5 (2 sentences, averaged) |
| parts per word | 1.45 | 1.28 |
| word score | +0.11 | **−0.64** |
| morph score | +0.60 | **−0.24** |

Both scores went down, so both say the simplified version is easier.

Real rows from `items_key.csv`:
| form | item | id | A_is |
|---|---|---|---|
| 1 | 1 | S27 | simplified |
| 1 | 2 | S36 | original |

---

# PART C: check with real people

## Done by hand: the human study

### INPUT
The 2 Google Forms, made from `make_google_form.gs`.

### WHAT WE DID
12 native Tamil readers filled in the forms. For each item, they saw sentence A and sentence B and answered:
1. Which is easier? (A / B / Same)
2. How difficult is A? (1–5)
3. How difficult is B? (1–5)
4. Do A and B mean the same thing? (1–5)
5. Which one has errors? (Neither / A / B / Both)

Then we downloaded the answers from Google Forms as CSV files.

### OUTPUT
`Dataset/human_study/responses/form1.csv` and `form2.csv`

---

## Step 9: compare the readers with our scores → RESULT 2
**File:** `src/step9_human_eval.py`

### INPUT
- `Dataset/human_study/responses/form1.csv` and `form2.csv`: the readers' answers
- `Dataset/human_study/items_key.csv`: which side was the original
- `output/simplification_scores.csv`: our scores

### CODE
**1. Keep only native or fluent readers.**
```python
if not str(resp[level_col]).startswith(("Native", "Fluent")): continue
```

**2. Use the answer key to change "A/B" into "original/simplified".**
```python
a_orig = d.A_is == "original"
d["diff_orig"] = np.where(a_orig, d.difficulty_a, d.difficulty_b)
d["diff_simp"] = np.where(a_orig, d.difficulty_b, d.difficulty_a)
```
Readers answered about "A" and "B". The key tells us which one was the original.

**3. Average over the about 6 readers of each item.**
```python
it = d.groupby("id").agg(h_orig=("diff_orig", "mean"), h_simp=("diff_simp", "mean"),
                         pick_simp=("easier", lambda e: (e == "simplified").mean()), ...)
```

**4. The main question: which score agrees with the readers?**
```python
spearmanr(our_score, human_difficulty)     # +1 = perfect agreement, 0 = no relation
```
It is calculated for both scores, and also the **difference** (morph − word), with a 95% confidence interval from 2,000 resamples.

**5. Do the readers agree with each other?** (`rater_agreement()`)

### OUTPUT
- `output/human_ratings_long.csv`: one row per reader per item (360 rows)
- `results/human_eval_summary.md`: **the Phase C result**

**Result 2:**
| Question | Answer |
|---|---|
| Readers picked the simplified version as easier | **49%** (the original: 29%) |
| Difficulty, original → simplified | **2.31 → 1.97** |
| Meaning kept (1–5) | 3.72 |
| Agreement with readers: word score | **+0.32** |
| Agreement with readers: morph score | **+0.28** |
| Morph − word | −0.04, with a 95% CI from −0.11 to +0.02, which includes 0 → **morph is not better** |
| Readers agreeing with each other | about 0.05 (low) |

---

## Step 10: ask Gemini the same questions → RESULT 3
**File:** `src/step10_llm_judge.py`

### INPUT
- `Dataset/human_study/make_google_form.gs`: the **exact** A and B texts the readers saw
- `items_key.csv`, `human_ratings_long.csv` and `simplification_scores.csv`
- a free Gemini API key (`GEMINI_API_KEY`)

### CODE
**1. Read the exact items from the forms file.**
```python
forms = json.loads(re.search(r"var FORMS = (\[.*?\n\]);", src, re.S).group(1))
```

**2. Send them to Gemini, which must answer in a fixed format.**
```python
body = {"contents": [...prompt with items...],
        "generationConfig": {"temperature": 0,                    # same answer every time
                             "responseMimeType": "application/json",
                             "responseSchema": SCHEMA}}           # must answer easier/difficulty/meaning/errors
r = requests.post(URL.format(model), headers={"x-goog-api-key": key}, json=body)
```

**3. Free limit:** if Gemini says "too many requests", the code waits and tries again. Answers are **saved as it goes** (a cache), so running it again doesn't waste the daily limit.

**4. Compare Gemini with the humans.** Gemini is kept separate and never counted as a human.

### OUTPUT
- `output/llm_judge_gemini-3.5-flash-lite.csv`: Gemini's answers
- `results/llm_judge_gemini-3.5-flash-lite.md`: the comparison

**Result 3:** Gemini agrees with the average human at **+0.30**, while one human agrees with the others at only +0.11. But Gemini is too positive: it says **90%** of the simplifications are easier (humans: 49%) and finds **0%** errors (humans: 27%).

---

# THE FINAL OUTPUT

| # | Final file | What it shows |
|---|---|---|
| 1 | `results/storyweaver/summary.md` and `results/summary.md` | Word-level features are best. Morphology adds nothing extra. |
| 2 | `results/simplification_summary.md` | Both scores say all 60 AI simplifications are easier. |
| 3 | `results/human_eval_summary.md` | Readers agree the text got easier. Morph is not better than word. |
| 4 | `results/llm_judge_gemini-3.5-flash-lite.md` | An AI judge can help, but it can't replace humans. |

### The one final answer of Phase 3
> **Morphology does carry difficulty information in Tamil, but word length already contains it.** Two completely different tests (graded pairs in step 6, real readers in step 9) give the same answer.

### How to run everything again
```
cd phase3_tamil_extension
python src/step0_fetch_textbooks.py
python src/step1_extract_sentences.py
python src/step2_build_pairs.py
python src/step3_textbook_features.py          (~10 min)
python src/step4_storyweaver_data.py
python src/step5_storyweaver_features.py       (~1 hour)
python src/step6_evaluate_metric.py
python src/step6_evaluate_metric.py storyweaver
python src/step7_select_complex.py
   → paste prompts into the AI chat, save simplified_<name>.txt
python src/step8_score_simplified.py
   → run make_google_form.gs, collect answers, save form1.csv / form2.csv
python src/step9_human_eval.py
python src/step10_llm_judge.py
```

# Speaking Script: MTP Internal Evaluation (20 Oct 2026)

**Total time:** about 25 minutes of talking + 5 minutes of questions.
**How to use this:** read each "Say" part out loud 3–4 times today. Don't learn it word by word. Learn the **order of ideas**. The "If he asks" boxes are for questions.

---

## Slide 1: Cover (30 seconds)

**Say:**
> Good morning, sir. My project is about **measuring how hard a sentence is to read**, and **making hard sentences easier**.
>
> It has three parts. First, I repeated an English research paper by Singh and others from 2016. Second, I improved it using GPT-2. Third, I took the same idea to Tamil.
>
> I will take about 25 minutes, and then I am happy to take questions.

⚠️ **Before tomorrow:** replace "[Professor's name]" on this slide with your guide's name.

---

## Slide 2: Which sentence is harder? (1 minute)

**Say:**
> Look at these two sentences. Most people feel **sentence B is harder**. It has longer words, rarer words like "notwithstanding" and "ratified", and a more complex structure.
>
> But how can a computer know this? That is the question of my project.
>
> I have two goals. **First, measure difficulty.** The paper's idea is that when a sentence is hard, people's eyes stay longer on the words. So if we can predict how long people look at each word, we can tell how hard the sentence is.
>
> **Second, reduce difficulty.** Rewrite hard sentences in a simpler way, and check with real readers if they really became easier.

---

## Slide 3: Three phases (1 minute)

**Say:**
> The work has three phases.
>
> **Phase 1:** I rebuilt the Singh et al. paper on English, from scratch, in Python. This is done.
>
> **Phase 2:** In the paper, one feature, syntactic surprisal, did not work. So I replaced it with GPT-2. This is also done.
>
> **Phase 3:** I moved to Tamil. Tamil words are very long, because they are built from many small parts. I made a Tamil difficulty measure, simplified sentences with an AI model, and tested them with native Tamil readers. The study is done; the write-up is next.

---

## Slide 4: Two systems (5–6 minutes) ⭐ MOST IMPORTANT SLIDE

This is where you show you **understood the paper**. Go slowly.

### Part A: the data and the two systems (1.5 minutes)

**Say:**
> The paper uses the **Dundee eye-tracking corpus**. 10 people read newspaper text while a machine recorded their eyes. It has 50,647 words. For every word, we know how long each person looked at it.
>
> The paper has **two systems**.
>
> **System 1** predicts reading time.
> - **Input:** one word, described by numbers called features. For example, how long the word is and how common it is.
> - **Output:** how many milliseconds a person looks at that word.
> - It is a linear regression model.
>
> It predicts four kinds of reading time:
> - **FFD**, first fixation: the very first look at the word.
> - **FPD**, first pass: all looks at the word before the eyes move forward.
> - **RPD**, regression path: from the first look until the eyes go past the word, including going back.
> - **TD**, total duration: all the time spent on the word, even coming back later.
>
> **System 2** predicts which sentence is harder.
> - **Input:** two sentences.
> - **Output:** which one is harder.
> - It takes System 1's predicted reading time for every word in both sentences, and a classifier decides.
> - It is tested on 117,712 pairs from normal Wikipedia and Simple Wikipedia. Normal Wikipedia is the harder one, so we already know the right answer.
>
> **Why this is clever:** we need the eye-tracker only once, to train System 1. After that, we can judge any sentence without an eye-tracker.

### Part B: how each feature is calculated (3–4 minutes)

**Say:**
> Now I'll explain how each feature is calculated. I'll use one real sentence from the data: *"Are tourists enticed by these attractions threatening their very existence?"*
>
> **First, the simple word features.**
> 1. **Word length:** count the letters. "attractions" has 11.
> 2. **Sentence length:** count the words. This sentence has 10.
> 3. **Word frequency:** how common the word is, taken from a large word-frequency list and written as a log. "Are" is −5.2, very common. "enticed" is −14.4, rare.
> 4. **Age of acquisition:** the age at which people usually learn the word, from the Kuperman list. "by" is learned at 5 years, "enticed" at 12.5 years.
> 5. **AoA standard deviation:** how much different AoA lists disagree about that age.
> 6. **Forward and backward probability:** how often this word comes after the previous word, and before the next word. I counted this in the British National Corpus, 100 million words. "enticed" after "tourists" is very unlikely.
>
> The simple rule: long, rare, late-learned and unexpected words take more reading time.
>
> **Now the syntactic features.** These are about grammar and structure.
> 7. **Lexical surprisal:** how surprising the word is after the previous word: minus log of the probability. Surprising words are read more slowly.
> 8. **Syntactic surprisal:** how surprising the **grammar** is at this word. I used the same parser the paper used, the left-corner parser. It learned English grammar from the Penn Treebank, which is newspaper sentences with their grammar trees already marked.
>    - The parser reads the sentence one word at a time.
>    - At each word, it keeps all the possible grammar trees, each with a probability.
>    - If the new word does not fit the trees it expected, the surprisal is high.
>    - Example: "Are" gets a high value, 35.8, because newspaper sentences rarely start as a question. "by" gets 0.2, because the grammar expects it there.
> 9. **Entropy reduction:** how much the uncertainty about the next word drops at this word. After "their", a noun must come, so the uncertainty drops.
> 10. **Embedding depth:** think of the sentence as a tree with the main verb at the top. Count how many steps the word is from the top. "enticed" is the main verb, so 0. "by" is 2. "their" is 4. Deeper words are harder to keep in memory.
> 11. **Embedding difference:** this word's depth minus the previous word's depth. For "by", 2 minus 0 is +2, so the reader goes deeper.
> 12. **Hierarchical features, h1 to h8:** small yes/no checks. Did the depth go up, go down, stay the same, or jump a lot?
>
> So in total, each word has 12 groups of features, which is 20 numbers. These go into System 1.

> 💡 **If he says "show me":** open a terminal and run `python src/feature_demo.py`. It prints how each feature is calculated, step by step.

---

## Slide 5: Phase 1 + 2 architecture (2 minutes)

**Say:**
> This is the full pipeline. There is one Python script per step.
>
> **Step 1** reads the Dundee eye-tracking files and works out four reading times for every word.
> **Step 2** makes word features: length, how common the word is, the age children learn it, and how predictable it is from the word before.
> **Step 3** makes grammar features from a parser.
> **Step 4 is System 1:** a Ridge regression that learns reading time from all 17 features.
> **Step 5 is System 2:** for each Wikipedia and Simple Wikipedia pair, it uses the predicted reading times to say which sentence is harder. It reaches 75.26%.
>
> **Phase 2 changes only one thing** (orange box): in step 3, the parser surprisal had almost no link to reading time, r = −0.017. I replaced it with GPT-2 surprisal, r = 0.465, and retrained step 4 with the same code.

---

## Slide 6: Phase 1 results (2 minutes)

**Say:**
> These are my replication results.
>
> First, what R² means: it tells how much of the reading-time differences the model can explain. 1 would be perfect.
>
> **On the left, System 1.**
> - TD: 0.536, which is **better** than the paper's 0.516.
> - FPD and RPD are close to the paper.
> - Only FFD, the first look, is clearly lower: 0.444 against 0.649.
>
> **On the right, System 2**, the final goal. It **beats the paper on all four models**. The best is 75.26%, against the paper's 75.21%. 50% would be guessing.
>
> So System 2 works fully, even though System 1's FFD is lower.

**If asked "why is your FFD lower?" (backup answer, no slide):**
> First, two quick terms. **r** tells how strongly two things move together: 0 means no link, 1 means a perfect link. **R²** tells how much of the reading time the model explains. With only one feature, R² is simply r × r.
>
> I did not just accept the FFD gap. I tested three possible reasons.
>
> **One:** maybe my syntactic surprisal was computed wrongly. So I ran the real parser the paper used, inside Linux, on all 50,647 words. Its r with reading time is still almost zero, −0.017. The paper's own Table 1 also shows almost zero, −0.067. So my code is not the problem.
>
> **Two:** maybe words that people skipped were counted as zero time. I checked: after averaging over 10 readers, **no word has zero time**. The smallest is 5.1 milliseconds. So this is not the reason.
>
> **Three, my best lead:** the paper's own two tables do not agree. When you use only one feature, R² must equal r × r. Table 1 says r = 0.765 for word length, so R² should be about 0.585. But Table 2 says 0.267. These cannot both be true. My numbers do match: my r is 0.67, 0.67 × 0.67 = 0.45, and my model's R² is 0.447. So my pipeline is consistent, and the gap most likely comes from the paper.

---

## Slide 7: GPT-2 (3 minutes) ⭐ "Why GPT-2" slide

**Say:**
> **Why did I enhance the paper?** Because syntactic surprisal, the paper's main grammar feature, carried **no signal** at all. One important feature was not working.
>
> **Why GPT-2?**
> - The grammar parser only knows grammar rules.
> - The word-pair feature only looks **one word back**.
> - **GPT-2 looks at all the words before**, so it understands the context. It knows which words are expected.
> - Research on reading shows that people slow down on words they did not expect. GPT-2 measures exactly that.
>
> A simple example: in "tourists enticed **by**", the old word-pair method says "by" is surprising, 10.6. GPT-2 says it is almost certain, 0.9, because "enticed by" is a common phrase. GPT-2 is right.
>
> **How I did it:**
> 1. **Input:** the Dundee sentences, plus the 3 previous sentences as context.
> 2. GPT-2 gives a probability for each word. Surprisal is minus log of that probability.
> 3. **Output:** one GPT-2 surprisal number for every word.
> 4. I replaced the old syntactic surprisal with this number and trained System 1 again.
>
> **Result:** its correlation with reading time is **0.465**, against almost zero for the parser. All four measures improved. **RPD and TD now beat the paper**, and FPD is very close.
>
> **Two more findings:** giving 3 previous sentences as context helped the most. And a **bigger GPT-2 was worse**. This matches a published paper by Oh and Schuler, 2023: bigger language models are not always closer to how humans read.
>
> I also checked that GPT-2 adds new information. It correlates only 0.014 with the old syntactic surprisal, so it is not a copy of an existing feature.

---

## Slide 8: Why Tamil (1.5 minutes)

**Say:**
> Now Phase 3. Why Tamil?
>
> Tamil builds words by joining many parts. This one word, வீடுகளிலிருந்து, means "from the houses": house, plus plural, plus "in", plus "from". One Tamil word is like a whole English phrase.
>
> In English, word length was the strongest feature. But in Tamil, a word can be long just because it has many endings. So my **hypothesis** was: in Tamil, counting letters is the wrong unit; counting **word parts, called morphemes**, should measure difficulty better.
>
> Also, there is no Tamil eye-tracking data and no Tamil readability dataset. So I had to build my own.

---

## Slide 9: Phase 3 architecture (1.5 minutes)

**Say:**
> Phase 3 has three stages, from left to right.
>
> **A, Metric:** I built two graded Tamil datasets, school textbooks and StoryWeaver stories. For each sentence I compute word-level, morphology and syntax features, and a model learns to pick the harder sentence. This gives me two metrics: word-level and morphology.
>
> **B, Simplify:** I picked 60 hard sentences. An LLM rewrote each one in simple Tamil, and both metrics scored the original and the simplified version.
>
> **C, Check:** 12 native Tamil readers rated the same pairs. I checked which metric agrees better with the readers. Gemini was an extra judge, reported separately, never counted as a human.

---

## Slide 10: Tamil data and features (1.5 minutes)

**Say:**
> I built two datasets where we already know which text is harder.
>
> **First, Tamil Nadu state textbooks**, Class 3 to 12. I took 2,351 prose sentences. I skipped poetry, because it uses old Tamil. I made 8,004 pairs where one sentence is at least 3 classes harder.
>
> **Second, StoryWeaver**, children's stories with reading levels 1 to 4.
>
> For every sentence I calculate three groups of features:
> - **A, word-level:** the same idea as English: word length, frequency, and Tamil GPT-2 surprisal.
> - **B, morphology:** my new idea. How many parts each word has, grammar markings like case, and verb forms. Tools: Morfessor splits the words, and Stanza finds the grammar. Stanza is about 95% correct on Tamil case.
> - **C, syntax:** sentence structure from the Stanza parser.

---

## Slide 11: Tamil metric results (2 minutes)

**Say:**
> The test is: given two sentences, can the model pick the harder one? 50% is guessing.
>
> **Word-level features do best overall**: 63.8% on StoryWeaver.
>
> **Morphology alone is also clearly above guessing**, and it is the **best when both sentences have the same length**. So morphology does carry real information.
>
> **But adding morphology on top of word-level gives nothing.** The reason: in Tamil, long words are exactly the words with many parts. So word length already contains the morphology information.
>
> So my hypothesis is **not supported**. This is an honest negative result. Also, textbook grades are given per book, not per sentence, so those labels are noisy.

---

## Slide 12: LLM simplification (1.5 minutes)

**Say:**
> Now the second goal: reducing difficulty.
>
> I picked 60 hard sentences **at random**: 30 from Class 11 and 12 textbooks, and 30 from the hardest StoryWeaver level. Random, so the test does not favour either metric.
>
> An AI language model rewrote each sentence in simple Tamil.
>
> The results:
> - Sentences became about **half as long**, from 12.8 to 6.8 words.
> - Words have **fewer parts**.
> - There are far **fewer non-finite verbs**. These are the chained verb forms that join many clauses into one sentence.
>
> **Both metrics say every simplification is easier.** So they agree on everything, and I can't use this alone to tell which metric is better. That's why I needed real readers.

---

## Slide 13: Human study (2 minutes)

**Say:**
> 12 native Tamil readers answered two Google Forms. Each pair got about 6 ratings.
>
> **Left side: did the simplification help?** Yes.
> - Readers picked the simplified version as easier 49% of the time; only 29% picked the original.
> - Difficulty dropped from 2.31 to 1.97, on a scale of 1 to 5.
> - Meaning was mostly kept: 3.72 out of 5.
>
> **Right side: which metric agrees with readers?**
> - Word-level: 0.32. Morphology: 0.28. Both agree with readers.
> - The difference is −0.04, and its confidence interval includes zero.
> - So **morphology is not better.**
>
> This is the same answer as slide 10, but from a completely different kind of evidence: real people instead of textbook labels.

---

## Slide 14: Gemini as a judge (1 minute)

**Say:**
> I also asked Gemini, through its free API, the **exact same questions** the humans saw. I report it separately; it is never counted as a human.
>
> Gemini agrees with the average human at 0.30. That is more than one human agrees with the others, 0.11. So it behaves like a reasonable reader.
>
> **But it is too optimistic.** It says the simplified version is easier 90% of the time, and it finds no errors at all. Humans found errors in 27%.
>
> So an AI judge can **support** a human study, but it cannot **replace** it.

---

## Slide 15: Four findings (1 minute)

**Say:**
> To summarise, four findings:
> 1. The replication works. System 2 beats the paper on all four models.
> 2. GPT-2 surprisal is a real reading-time signal, and two measures now beat the paper.
> 3. In Tamil, morphology carries difficulty information, but word-level features already capture it. Two different tests agree on this.
> 4. AI simplification really helps native readers, and both metrics can detect it.

---

## Slide 16: Next steps and limitations (1 minute)

**Say:**
> What is left:
> - finish the second Gemini run
> - build the demo toolkit for the final evaluation, with an English tab and a Tamil tab
> - write the thesis
>
> Limitations, to be honest:
> - Only 12 raters, and they did not agree much with each other. This is common for difficulty ratings, but it weakens the result.
> - Textbook grades are per book, so the labels are noisy.
> - I tested only one simplification model.
> - The FFD gap is not fully explained, but I have a strong lead.
>
> Thank you, sir. I'm happy to take questions.

---

# Questions he may ask: short answers

| Question | Answer |
|---|---|
| What is the input and output? | **System 1:** a word with its features as input; reading time in milliseconds as output. **System 2:** two sentences as input; which one is harder as output. |
| Why linear regression? | The paper used it. It is simple, and you can see how much each feature matters. |
| What is surprisal? | How unexpected a word is: minus log of its probability. A low probability gives high surprisal. |
| Why does GPT-2 work better than the parser? | The parser only knows grammar. GPT-2 knows real word patterns and uses the whole context. |
| Why not a bigger model like GPT-3 or a larger GPT-2? | I tried the larger GPT-2 and it was worse. Published research says the same. Also, GPT-2 is free and runs on my laptop. |
| Why was FFD lower? | I tested three reasons. The best lead is that the paper's own two tables do not agree with each other. |
| How did you split the data? | By **sentence**: 60% train, 20% dev, 20% test. A sentence is never in both train and test. |
| What is R²? | How much of the differences in reading time the model explains. 1 is perfect. |
| Why average over 10 readers? | Readers differ a lot. One read "Are" in 102 ms, another in 216 ms. Averaging gives one stable time per word. |
| Did you write the code yourself? | "I designed every step and wrote it with AI coding help. I can explain every step and its output." |
| Why is h5 always 0? | It needs the list of children for each word in the tree, which I did not compute. I say this honestly. |

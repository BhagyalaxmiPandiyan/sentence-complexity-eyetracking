# Evaluating Morphology-Aware Sentence Complexity and LLM Simplification for Tamil

**Author(s):** [Add author name(s)]  
**Affiliation:** [Add institution and department]

## Abstract

Readability methods developed for English often use word length, frequency, and language-model surprisal as proxies for sentence difficulty. Their transfer to Tamil is not straightforward: Tamil is agglutinative, and grammatical information commonly appears as suffixes attached to a lexical stem. This study evaluates whether explicit morphological features improve sentence-difficulty ranking over a word-level baseline, and whether either metric agrees with readers when evaluating large-language-model (LLM) simplifications. We extract 12 lexical, morphological, and syntactic features from Tamil State Board textbook sentences and StoryWeaver book descriptions. Pairwise logistic-regression models are evaluated against grade or publisher-assigned reading-level labels, with book- or page-level train/test separation. The word-level model achieves 57.5% test accuracy on textbook pairs and 63.8% on StoryWeaver pairs. Adding morphological features changes these scores to 57.9% and 63.4%, respectively; neither gain is statistically distinguishable from zero. In a second experiment, an LLM simplified 60 Tamil sentences. Both the word-level and morphology-augmented scores ranked all 60 rewrites as easier, but native readers selected the simplified version as easier on 49% of items and rated its mean difficulty 1.97 versus 2.31 for the original. Across the 120 texts, correlations with mean human difficulty were modest for both metrics (Spearman's ρ = .323 for word-level and .281 for morphology-augmented); the paired difference was not significant. Correlations between metric score changes and reader-rated changes were close to zero. These pilot results show that the evaluated morphological features carry some signal in isolation but do not yet demonstrate improved prediction beyond the word-level baseline. They also highlight the importance of validating automatic simplification scores against human judgments. Finally, a webcam eye-tracking pilot (five native Tamil readers who passed the comprehension check, 100 Tamil news sentences) gives the same picture with reading behaviour: word-level features predict held-out total reading time (R² = .103 vs .027 for position controls), morphological features alone also predict it (R² = .071), but adding them to the word-level features does not help (ΔR² = −.004). At the sentence level, the word-level metric tracks reading time more closely than the morphology-augmented metric (ρ = .851 vs .682), mainly because it weights sentence length more heavily.

**Keywords:** Tamil, sentence complexity, morphology, readability, text simplification, human evaluation

## 1. Introduction

Sentence complexity is relevant to reading support, educational materials, and automatic text simplification. A common approach is to represent a sentence using measurable linguistic features and learn a model that ranks or classifies texts by difficulty. Singh et al. (2016) demonstrated a two-stage approach for English: predict word reading times from linguistic features and use these predictions to rank sentence pairs. The earlier phases of this project replicated that work and explored language-model surprisal as an additional feature.

This paper extends the project to Tamil. In an agglutinative language, a surface word may contain a stem and several grammatical suffixes. Consequently, character length alone can conflate lexical complexity with grammatical information. A morphology-aware representation could potentially distinguish these sources of difficulty. No Tamil eye-tracking corpus comparable to the Dundee corpus was available, so the main experiments evaluate sentence rankings against graded-text labels and compare automatic scores with judgments from Tamil readers. In addition, we collected a small browser-based webcam eye-tracking pilot to test the original reading-time prediction task on Tamil at feasibility scale.

The study addresses three questions:

1. Do morphological features improve the prediction of Tamil sentence difficulty over a word-level feature baseline?
2. Do either of these metrics track native or fluent Tamil readers' judgments of LLM-generated simplifications?
3. In a webcam eye-tracking pilot, do morphological features add to word-level features when predicting Tamil word and sentence reading times?

The results are mixed. Morphological features show some standalone association with reading level, particularly on StoryWeaver, but do not yield a reliable improvement when added to the word-level model. The human study likewise finds no evidence that the morphology-augmented score aligns with readers better than the baseline. We therefore treat this work as an exploratory evaluation rather than evidence that morphology is irrelevant to Tamil reading.

## 2. Related Work

Readability and sentence-complexity research has used lexical frequency, word length, syntactic structure, and language-model probabilities, among other predictors. Singh et al. (2016) use reading-time measures and linguistic features to quantify sentence complexity. Their work motivates the English-style word-level baseline in the present study, but no Tamil eye-tracking corpus of comparable size exists; our own webcam pilot (Section 3.4) is far smaller.

Text simplification systems are commonly evaluated through a combination of automatic measures and human judgments. Automatic scores can indicate that a rewrite is shorter or contains more frequent words, but they do not by themselves establish that the rewrite is easier to understand, preserves meaning, or is grammatically correct. This distinction is especially important when the generator may restructure or split a sentence.

The present study contributes an empirical comparison of word-level and morphology-augmented features for Tamil using two graded text sources, followed by a native-reader evaluation of generated simplifications. The results are limited to the datasets, feature extractors, and reader sample described below; we do not claim that this is the first Tamil readability or simplification study.

## 3. Data and Methods

### 3.1 Graded Tamil text

We use two datasets with labels assigned at the source-book level:

* **Tamil State Board textbooks:** 2,351 extracted sentences from Class 3–12 textbooks. Class 3–7 material is from Term 1 textbooks. Sentence-pair labels are based on class level, not sentence-level reader judgments.
* **StoryWeaver:** 7,080 sentences retained for feature analysis from Tamil book descriptions at reading levels 1–4. The descriptions are short publisher-provided blurbs; the reading level is assigned to the book rather than independently to each sentence.

For textbooks, pairs are drawn from classes separated by at least three grades. For StoryWeaver, pairs are drawn from reading levels separated by at least two levels. Pair order is randomized, and the label identifies which member comes from the higher grade or reading level. Pair construction is capped by grade/level combination to reduce domination by the largest groups. An 80/20 train/test split is made by textbook page or StoryWeaver book, respectively, to reduce leakage between sentences from the same source. This produces 8,004 textbook pairs (4,176 train; 3,828 test) and 7,492 StoryWeaver pairs (5,994 train; 1,498 test). A length-matched subset includes pairs whose sentence word counts differ by no more than one.

These labels are proxies for sentence difficulty. In particular, sentences from the same textbook or book can vary considerably in difficulty, and an assigned reading level is not a direct measurement of an individual reader's comprehension.

### 3.2 Sentence features and models

Each sentence is represented with 12 features in three groups:

* **A — word-level baseline (4 features):** number of words, mean word length, mean word-form frequency, and mean word surprisal.
* **B — morphology-aware (6 features):** mean morphemes per word, stem surprisal, suffix surprisal, marked morphological features per word, lemma frequency, and count of non-finite verbal forms.
* **C — syntax (2 features):** dependency-tree depth and mean dependency distance.

Tamil word forms are segmented with the pretrained Tamil Morfessor model distributed with IndicNLP resources. Stanza supplies tokenization, morphological analyses, lemmas, and dependency parses. Word and morpheme surprisal are derived from the `abinayam/gpt-2-tamil` language model. Because its subword tokens do not align with Tamil morpheme boundaries, token surprisal is assigned to character spans and aggregated over each word or segmented morpheme span. Thus, the reported morphology-level surprisal is an alignment-based estimate, not a language model trained directly on morpheme tokens.

Pairwise logistic-regression models are fit to scaled feature differences between the two sentences in each training pair. Reported test accuracy is the proportion of test pairs for which the model selects the higher-grade or higher-level member as harder. Chance performance is 50%. We compare A, B, C, A+B, A+C, and A+B+C. Test-set confidence intervals and paired comparisons use 1,000 sentence-bootstrap draws.

### 3.3 LLM simplification and reader study

For the simplification experiment, 30 eligible sentences were sampled from Class 11–12 textbooks and 30 from StoryWeaver level 4. Eligibility required 10–30 words and passed heuristic filters for extraction artifacts; the selected items were not individually verified by a Tamil linguist. One LLM system (Claude Opus 5.5, run through Claude Code on 13 October 2026) rewrote the sentences using the same prompt: simplify for a Class 3–5 child, preserve meaning, prefer common everyday Tamil, and allow splitting into shorter sentences. Its output was not produced by a trained simplification model. When a rewrite contained multiple sentences, its sentence-level feature values were averaged before scoring, so a split did not automatically increase the score merely by increasing the number of output sentences.

The two simplification metrics are sentence scores derived from pairwise models trained on StoryWeaver training pairs: **word** uses feature group A; **morph** uses A+B. Higher scores indicate greater predicted difficulty. For each original/rewrite pair, we compute the original score minus the rewrite score; a positive difference means the metric considers the rewrite easier.

Twelve native or fluent Tamil readers rated the 60 pairs through two forms, with a mean of six reader ratings per item. The original and simplified sentences were displayed in randomized A/B order. Readers rated the difficulty of each sentence on a five-point scale (1 = very easy, 5 = very hard), indicated which sentence was easier (A, B, or same), rated meaning preservation, and reported perceived errors. They were not told which sentence was generated. The analysis compares metric scores with mean human difficulty at the text level and compares score differences with human difficulty differences at the pair level. Confidence intervals and metric differences use 2,000 item-bootstrap draws, resampling both texts in a pair together.

An LLM judge was also evaluated as a separate exploratory analysis. Its responses are not pooled with the human ratings.

### 3.4 Webcam eye-tracking pilot

**Materials.** One hundred sentences of 6–16 words were selected from UD Tamil-TTB, which contains Tamil news sentences from Dinamani (Ramasamy and Žabokrtský, 2012). Sentences with digits, Latin letters, brackets, or quotation marks were excluded. Candidates were scored on word count, mean dependency distance, tree depth, and mean word length, cut into easy, medium, and hard thirds, and sampled 33/33/34 from these bands. Twenty sentences were followed by a yes/no probe ("Was this word in the sentence?").

**Procedure.** Readers took part in their own browser. Gaze was estimated from the laptop webcam with WebGazer (Papoutsaki et al., 2016) at about 30 Hz after a 13-point click calibration, with a 9-point validation check at the start and at three breaks (recalibration if the median error exceeded 150 px). Each sentence was shown whole at 30 px, in random order; the reader pressed ENTER when finished. Fixations were detected from the gaze samples and assigned to word boxes, and first fixation duration (FFD), first-pass duration (FPD), regression-path duration (RPD), and total duration (TD) were computed with the same rules as our Phase 1 Dundee pipeline. A sentence's gaze data were marked unusable when most gaze fell off the text. Only gaze coordinates were stored, not webcam images.

**Readers.** Developer test sessions and registrations that produced no reading data are not analysed. Seven reading sessions remained. Two different readers had entered the same participant code; their sessions were separated by date (the later one is listed as R05). One reader took part twice under different codes: his first session (50 sentences, 58% probe accuracy) failed the pre-set 80% rule, and from his second session only the 50 sentences he had not already read were used (88% probe accuracy), so every analysed reading is a first reading. One further reader answered 75% of the probes correctly and was excluded under the 80% rule; a sensitivity analysis that includes this reader is reported in Section 4.4. Five readers were included (all native Tamil speakers, aged 16–39); four read all 100 sentences, with probe accuracy between 80% and 95%. Their median validation errors were 114–144 px, while the median word on screen was 153 px wide (middle half 111–202 px); webcam gaze error is therefore about one word.

**Analysis.** Following Phase 1, each word's reading-time measure is the sum over readers whose gaze on that sentence was usable, divided by the number of such readers, with skipped words counted as 0 ms. We compute the step-3 features per word: word length in aksharas, word-form frequency, and GPT-2 surprisal (A); and morpheme count, stem and suffix surprisal, marked UD features, lemma frequency, and a non-finite-verb indicator (B). All models include position controls (word position, first or last word on a line, sentence length). Linear regressions are evaluated by 10-fold cross-validation grouped by sentence, with 2,000 sentence-bootstrap draws for confidence intervals. At the sentence level, reading time is the time to the ENTER press (valid even when gaze is poor), log-transformed, trimmed at 3 SD, z-scored within reader, and averaged. It is compared with the word and morph metric scores of Section 3.3, which were trained on StoryWeaver and never on these data.

## 4. Results

### 4.1 Graded-text pair ranking

**Table 1. Test-pair accuracy by feature group.** Values in brackets are 95% confidence intervals. Length-matched accuracy is shown separately.

| Dataset | Model | Test accuracy (95% CI) | Length-matched accuracy |
|---|---|---:|---:|
| Textbooks | A: word-level | 57.5% [50.8, 64.1] | 49.0% |
| Textbooks | B: morphology | 51.9% [44.7, 58.8] | 48.2% |
| Textbooks | C: syntax | 62.5% [55.7, 68.7] | 55.0% |
| Textbooks | A+B | 57.9% [50.5, 64.6] | 49.5% |
| Textbooks | A+C | 57.6% [50.6, 63.9] | 49.8% |
| Textbooks | A+B+C | 57.7% [50.2, 64.0] | 49.4% |
| StoryWeaver | A: word-level | 63.8% [57.7, 69.7] | 57.3% |
| StoryWeaver | B: morphology | 60.5% [54.6, 66.5] | 59.7% |
| StoryWeaver | C: syntax | 59.8% [53.3, 66.2] | 51.6% |
| StoryWeaver | A+B | 63.4% [57.5, 69.4] | 57.3% |
| StoryWeaver | A+C | 64.3% [58.3, 70.2] | 59.4% |
| StoryWeaver | A+B+C | 63.3% [57.3, 69.0] | 57.1% |

On the textbook test set, the word-level model scores 57.5%, while A+B scores 57.9%. The paired-bootstrap gain is 0.3 percentage points (95% CI: −4.2 to +4.9; one-sided p = .437). On StoryWeaver, A scores 63.8% and A+B scores 63.4%; the gain is −0.3 points (95% CI: −4.0 to +3.8; p = .587). These results do not establish an improvement from adding the evaluated morphological features.

Morphology-only performance is above chance on StoryWeaver overall (60.5%), and its 59.7% accuracy on the length-matched subset is higher than the corresponding word-level score of 57.3%. On textbooks, however, the morphology-only model is near chance. The sentence-level textbook labels appear especially noisy: even the word-level model is near chance on length-matched textbook pairs (49.0%). In both datasets, confidence intervals for the reported models overlap substantially.

### 4.2 LLM-generated simplifications

Both metrics assigned a lower average difficulty score to every one of the 60 LLM rewrites. The mean score drop was 0.874 for the word-level metric (95% CI: 0.750–1.009) and 0.911 for the morphology-augmented metric (95% CI: 0.780–1.038).

The mean number of words decreased from 12.75 to 6.79. Mean morphemes per word decreased from 1.44 to 1.34, non-finite verb counts from 2.52 to 0.98, and mean dependency-tree depth from 5.48 to 3.82. The rewrites averaged 2.22 sentences, indicating that the generator frequently split source sentences. Word-level surprisal increased from 12.17 to 12.70 bits, demonstrating that not all features moved in the direction conventionally associated with lower difficulty.

### 4.3 Comparison with human judgments

Readers selected the simplified text as easier on 49% of items and the original on 29%; the remaining responses selected neither/equally easy. Mean rated difficulty decreased from 2.31 for originals to 1.97 for rewrites. Mean meaning-preservation rating was 3.72/5. Readers marked perceived errors in 33% of originals and 27% of rewrites.

At the text level (120 texts), the word-level metric correlated with mean human difficulty at ρ = .323 (95% CI: .149–.484); the morphology-augmented metric correlated at ρ = .281 (95% CI: .103–.452). The difference (morph minus word) was −.042 (95% CI: −.106 to .021), providing no evidence that the morphology-augmented metric was better.

At the pair level, correlations between metric score drop and reader-rated difficulty drop were close to zero: ρ = .039 for the word-level metric (95% CI: −.238–.298) and ρ = .036 for the morphology-augmented metric (95% CI: −.253–.304). The between-metric difference was also close to zero. Rater agreement was low in both forms (mean pairwise Spearman's ρ = .054 and .058), which limits the precision of conclusions about reader-level preferences.

In the separate LLM-judge analysis, Gemini selected the rewrite as easier on 90% of items, compared with 49% for readers. Its agreement with mean human difficulty was ρ = .297 across the 120 texts, but it rated meaning preservation more positively and identified fewer errors than the human panel. This result is exploratory and does not replace the human evaluation.

### 4.4 Webcam eye-tracking pilot

Word-level agreement between readers was low (mean pairwise Spearman's ρ = .10 for FFD to .20 for TD, over 336 words per pair read with usable gaze on average), which sets a low ceiling for any feature model. Sentence reading times agreed much better (mean pairwise ρ = .64). After excluding sentences with unusable gaze, 1,021 words from 97 sentences remained (2.90 readers per word on average; 33% of word readings were skips).

**Table 2. Cross-validated R² for word reading times.** Brackets are 95% sentence-bootstrap intervals. All models include the position controls.

| Measure | Controls only | A: word-level | B: morphology | A+B |
|---|---:|---:|---:|---:|
| FFD | .100 [.060, .137] | .105 [.064, .144] | .099 [.061, .137] | .103 [.064, .141] |
| FPD | .087 [.048, .123] | .141 [.089, .193] | .120 [.072, .168] | .137 [.084, .190] |
| RPD | .037 [.011, .064] | .132 [.071, .189] | .094 [.050, .139] | .130 [.069, .188] |
| TD | .027 [−.004, .076] | .103 [.032, .208] | .071 [.028, .141] | .099 [.030, .202] |

For FPD, RPD, and TD, both feature groups improved on the controls: word-level features added .054 to .095 R² and morphological features .033 to .057, all with intervals above zero. Adding morphology to word-level features did not help for any measure (ΔR² between −.002 and −.004; all intervals include zero). FFD was not predicted beyond the controls, consistent with the 30 Hz sampling rate and one-word gaze error, which make the first fixation the least reliable measure. In the full A+B model for TD, word length had the largest coefficient (+118 ms per SD), followed by line-position and word-position controls (about ±45 ms per SD). Morpheme count correlated with TD (ρ = .18) but not as strongly as word length (ρ = .32).

At the sentence level, the word-level metric correlated with mean reading time at ρ = .851 (95% CI: .789–.895), and the morphology-augmented metric at ρ = .682 (.542–.780). The difference (−.169, CI −.285 to −.081) favours the word-level metric. Sentence length alone reached ρ = .883, so this result mainly reflects length. With reading time per word as the target (length removed), neither metric correlated reliably on its own (word ρ = −.050, CI −.246 to .152; morph ρ = .084, CI −.129 to .295), although the morph metric was slightly closer (difference +.133, CI +.012 to +.249); given the small sample and the null correlations, we treat this as a hint for future work rather than a finding. Mean reading time rose across the pre-defined bands: 7.2 s for easy, 9.4 s for medium, and 11.9 s for hard sentences.

**Sensitivity analysis.** Including the reader excluded at 75% probe accuracy (six readers) gives the same pattern: TD R² = .045 (controls), .153 (A), .106 (B), and .146 (A+B), with ΔR² for A+B over A of −.007 (CI −.015 to .000); sentence-level ρ = .864 (word) and .693 (morph).

## 5. Discussion

The central result is that the explicit morphological features tested here do not consistently improve ranking when added to the word-level baseline. This is a more cautious conclusion than the initial hypothesis that morphemes would be a better unit of analysis. On StoryWeaver, the morphology-only model performs above chance and is comparatively strong on length-matched pairs. Yet that signal does not translate into a reliable gain for A+B over A. On textbooks, all models perform weakly on length-matched pairs, consistent with grade labels being an imperfect proxy for sentence-level difficulty.

One possible explanation is redundancy: word count, word length, and language-model surprisal may already encode some of the variation that the morphological variables capture. Another is measurement error. The morphological representation depends on an unsupervised segmenter and automatic morphological analyses; the surprisal-to-morpheme alignment is approximate. Either issue could obscure a true contribution. The current experiments do not identify which explanation is responsible.

The simplification study shows why automatic score changes should not be treated as a substitute for reader evaluation. Both metrics consistently score the generated texts as easier, and readers report a modest mean difficulty decrease. However, readers do not consistently choose the rewrite as easier, the text-level correlations are modest, and pair-level correlations between metric changes and human-rated changes are close to zero. Reductions in sentence length and syntactic depth may therefore make rewrites look simpler to these feature models without reliably predicting which individual rewrite a reader will find easier. The observed meaning-preservation and error ratings also indicate that simplification quality involves dimensions beyond measured complexity.

The eye-tracking pilot points the same way with a different kind of evidence. Morphological features predict real reading times when used alone, so they are not noise. However, they add nothing once word length, frequency, and surprisal are in the model, which is the redundancy pattern seen in the graded-text and rating experiments. In Tamil, a word with more suffixes is also a longer word, so word length may already capture much of what morpheme count measures. At the sentence level, both metrics mostly measure length: reading time grows with the number of words, and neither metric reliably predicts reading time per word (the morph metric is only slightly closer). The word-level advantage at the sentence level should not be over-read, because it comes almost entirely from the larger weight that metric gives to sentence length.

The weak inter-rater correlations counsel against strong claims about individual item preferences. The human results support describing a small average rating difference in this sample, not claiming that the generated version is universally easier or that either metric is validated as a stand-alone readability measure.

## 6. Limitations

This study has several limitations:

1. **Proxy labels:** textbook grades and StoryWeaver reading levels apply to whole books, not individual sentences. Pairwise accuracy therefore measures agreement with source-level labels rather than ground-truth comprehension.
2. **Pilot-scale eye-tracking:** the reading-time experiment has five included readers (one contributing 50 sentences) and uses webcam tracking whose error (about 114–144 px) is close to one word width. Word-level reader agreement is low (ρ ≈ .10–.20), so the absolute R² values are far below lab eye-tracking results such as the Dundee-based Phase 1 replication, and the comparison between feature groups has limited power. It is a feasibility test of the original paper's System 1 on Tamil, not a replication at the original scale.
3. **Feature-tool quality:** Morfessor segmentation, Stanza analyses, frequency estimates, and the Tamil GPT-2 model may contain errors or domain mismatch. GPT-2 subword surprisal is aligned to characters, not generated from a morpheme-tokenized model.
4. **Small, low-agreement human study:** 60 items and 12 readers provide pilot-scale evidence. Low inter-rater correlations and only six ratings per item on average limit statistical power.
5. **Single simplification system:** all 60 rewrites came from one LLM system. We did not isolate the effect of prompting, decoding settings, or model choice.
6. **Meaning and correctness:** reader ratings are subjective and do not constitute expert linguistic annotation. The study does not establish semantic equivalence or grammatical correctness.
7. **Selection and generalization:** the tested sentences and genres are limited. Results should not be generalized to all Tamil registers, reader populations, or educational contexts.

## 7. Conclusion

We compared word-level, morphology-aware, and syntactic features for ranking Tamil sentences from two graded text sources, then compared two sentence-difficulty scores with reader judgments of 60 LLM-generated rewrites. Morphological features alone provide a measurable signal on StoryWeaver, including on length-matched pairs, but adding them to the word-level model does not yield a statistically reliable accuracy gain in either dataset. Both automatic scores rate every generated rewrite as easier, yet neither tracks item-level reader-rated difficulty changes well, and the morphology-augmented metric does not outperform the baseline in the human comparison. A webcam eye-tracking pilot shows the same pattern in reading behaviour: morphological features predict word reading times alone, but do not add to word-level features.

The findings do not show that Tamil morphology is unimportant. They show that these particular features, tools, labels, and sample sizes are insufficient to establish that morphology-aware scoring improves on the tested baseline. Future work should prioritize larger reading-time datasets (more readers, ideally with lab-grade eye tracking or larger text sizes), stronger validated morphological analyses, a larger and more diverse human panel, and multiple simplification systems. Until then, the metrics are best treated as exploratory signals to accompany—not replace—human evaluation.

## Data, ethics, and reproducibility

The analysis scripts and aggregate result tables are maintained in the project repository. Tamil State Board textbook sentences should not be redistributed; users should obtain source texts through their authorized channels and reproduce extraction locally. StoryWeaver materials are subject to the source's licensing terms. The reader study collected initials and self-reported Tamil reading proficiency. The eye-tracking pilot stored participant codes, age, gender, native-language status, screen size, and gaze coordinates; no webcam images were stored. UD Tamil-TTB is licensed CC BY-NC-SA 3.0. Consent and institutional ethics requirements should be confirmed with the project supervisor before external publication.

## References

1. Singh, A. D., Mehta, P., Husain, S., and Rajkumar, R. (2016). “Quantifying sentence complexity based on eye-tracking measures.” *Proceedings of the Workshop on Computational Linguistics for Linguistic Complexity*, 202–212.
2. Qi, P., Zhang, Y., Zhang, Y., Bolton, J., and Manning, C. D. (2020). “Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.” *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, 101–108.
3. Pratham Books. *StoryWeaver*. https://storyweaver.org.in/ (Tamil book descriptions and reading-level metadata; accessed for this project).
4. Indic NLP Library. Tamil morphological segmentation resources, including the pretrained Morfessor model used in this project. https://github.com/anoopkunchukuttan/indic_nlp_library
5. Abinayam. *GPT-2 Tamil* model. https://huggingface.co/abinayam/gpt-2-tamil
6. Ramasamy, L., and Žabokrtský, Z. (2012). “Prague Dependency Style Treebank for Tamil.” *Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC 2012)*, 1888–1894.
7. Papoutsaki, A., Sangkloy, P., Laskey, J., Daskalova, N., Huang, J., and Hays, J. (2016). “WebGazer: Scalable Webcam Eye Tracking Using User Interactions.” *Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI 2016)*, 3839–3845.
8. Claude Opus 5.5 was used as the single LLM simplification system recorded for this experiment. Model/version and prompting details should be retained with any final submission or replication.

---

**Drafting note:** This manuscript is based on the results currently stored in the project. Before submission, add author and institutional details; verify software/model citations and access dates; confirm consent and ethics documentation; and have the supervisor review the interpretation and reference list.

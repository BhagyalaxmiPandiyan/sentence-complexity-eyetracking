# Teleprompter Script

7 clips, matches the Camtasia recording plan.

> **How to use:** Open this file in VS Code and press `Ctrl+Shift+V` (Open Preview) — put the preview pane on your second monitor and zoom in with `Ctrl+=` a few times for a teleprompter-sized view. Or paste each clip's spoken text into Camtasia's own Script/Teleprompter panel (Record window → "Script" tab in Camtasia 2023) one clip at a time. Read at a slow, even pace — pause at the blank lines, they're natural breath points.

---

## Clip 1 — Slides 1–2

*[ACTION: PowerPoint Slide Show mode, start on Slide 1]*

Good [morning / afternoon].

Today I want to cover two things: what I found after taking the replication further with GPT-2, and a proposal for what I'd like to work on next — extending this to Tamil.

I'll show the code and live numbers as I go, not just the slides.

*[ADVANCE TO SLIDE 2]*

Quick recap since we last spoke.

System 1 predicts four eye-tracking reading-time measures per word — FFD, FPD, RPD, TD — from 17 linguistic features.

System 2 takes those predictions and classifies which of two parallel sentences, Wikipedia versus Simple Wikipedia, is harder to read.

Both are fully replicated on the Dundee corpus — 50,647 words, 2,378 sentences.

*[END CLIP 1]*

---

## Clip 2 — Phase 1 live demo (feature_demo.py)

*[ACTION: SWITCH TO TERMINAL]*
*[ACTION: RUN — `python src/feature_demo.py`
(run it once, unnarrated, BEFORE you hit record — it finishes in a few seconds since the data is already cached; then scroll the terminal back to the top of its output so you can scroll down through it live during the real take)]*

I want to make this concrete instead of just quoting numbers, so let me trace one real word through the whole pipeline.

*[SCROLL TO "PART 1: Raw Dundee Corpus" SECTION]*

This is one of the raw eye-tracking files — Subject A reading Text 1. Every row is a word this subject actually looked at, with FDUR, the fixation duration in milliseconds.

If a subject skipped a word, there's simply no row for it — that matters in a moment.

*[SCROLL TO "PART 3: Computing Lexical Features" SECTION]*

Take the word 'tourists'. Word length is 8. It's in a 10-word sentence. Its Wikipedia frequency is quite low — it's a moderately rare word. Its age of acquisition is 8.76 — most people learn this word around age 9. All of these come from real external sources: the Dundee corpus itself, a 30,000-word AoA database, and the British National Corpus.

*[SCROLL TO "PART 5: System 1" RESULTS TABLE, near the bottom]*

And here's where all 17 features land: FPD, RPD, and TD all land within 0.04 R-squared of the paper — TD actually beats it. FFD is the one gap — 0.445 against the paper's 0.649 — and I root-caused that gap by testing every reasonable way of averaging across the 10 subjects. It traces to an undocumented preprocessing choice in the original paper, not a defect in my pipeline.

That's Phase 1. Now let me show you what I did next.

*[END CLIP 2]*

---

## Clip 3 — Slides 3–4

*[ACTION: BACK TO POWERPOINT, SLIDE 3]*

The paper's syntactic surprisal feature — from an incremental left-corner parser — turned out to carry almost no signal for FFD.

I measured r equals minus 0.017.

And I cross-checked this against the paper's own Table 1: they report minus 0.067 for the same feature in their own data.

So this isn't a bug in my pipeline — it's a real property of that feature.

That raised the question: could a modern language model's surprisal do better than a hand-built parser's?

*[ADVANCE TO SLIDE 4]*

The method: compute GPT-2 surprisal per word — how unpredictable each word is given what came before, including up to 3 preceding sentences as context — then swap it in for syn_surprisal in the same 19-feature model I already had.

Same 60/20/20 sentence-level split, same Ridge regression, no other part of the pipeline touched.

And I tested this across all four reading-time measures, not just FFD.

*[END CLIP 3]*

---

## Clip 4 — Slide 5 live demo

*[ACTION: SWITCH TO TERMINAL]*

Here are the numbers, re-run live so they're directly reproducible.

*[ACTION: RUN — `python experiments/train_system1_gpt2.py --surprisal-file output/gpt2_surprisal_gpt2_ctx.csv`]*

*[WAIT FOR TABLE TO PRINT, POINT AT ROWS AS YOU SAY THIS:]*

RPD and TD now beat the paper outright.

FPD is within 0.017.

FFD is the one that still trails — but it improved too, from 0.444 baseline to 0.458 with GPT-2.

The pattern makes sense: FFD is a single brief first glance, the other three integrate more reading behavior, so they have more room for a richer predictability signal like GPT-2's to help.

*[END CLIP 4]*

---

## Clip 5 — Slide 6

*[ACTION: BACK TO POWERPOINT, SLIDE 6]*

Before I trusted this feature, I wanted to gently check it wasn't secretly just a duplicate of something I already had.

Its correlation with syn_surprisal comes out to about 0.014, which is close enough to zero that it seems to be picking up something genuinely new.

It does correlate a little more with lexical surprisal and frequency, which makes sense — they're all somewhat related, predictability-flavored features.

One small thing I caught along the way: if you sum up GPT-2 surprisal across a whole sentence, that total ends up correlating quite strongly — around 0.914 — with sentence length. So it was really just counting words in disguise.

To avoid that, I used the mean instead, which keeps it a much cleaner predictability signal.

*[END CLIP 5]*

---

## Clip 6 — Slide 7 live demo

*[ACTION: SWITCH TO TERMINAL]*

I want to show one negative result, because I think it matters for how you judge the rest of this.

I had a theory: words a subject skips get a reading time of zero, so word length might be secretly acting as a proxy for whether a word gets fixated at all, not for how long.

I tested this directly by filtering to only fixated words and retraining.

*[ACTION: RUN — `python experiments/fixated_only_system1.py`]*

*[WAIT FOR OUTPUT]*

Result: it made zero difference.

It turns out at the word-averaged level, across 10 subjects, essentially no word has a reading time of exactly zero — minimum FFD across all 50,647 words is 5.1 milliseconds.

So that theory is disproven, not just unconfirmed.

I corrected my own documentation once I found this, rather than leaving the wrong explanation in.

*[END CLIP 6]*

---

## Clip 7 — Slides 8–12 + closing

*[ACTION: BACK TO POWERPOINT, SLIDE 8]*

Last piece of Phase 2: System 2, the readability classifier, still beats the paper on all four numbers — Base RT and Extended RT, both scoring methods.

So even though FFD alone hasn't caught up to the paper, the downstream task that actually uses these predictions is solid.

I'd consider System 2 effectively closed at this point — further effort there has diminishing returns.

*[ADVANCE TO SLIDE 9]*

That's Phase 2. Now I want to propose Phase 3: taking this pipeline to Tamil.

*[ADVANCE TO SLIDE 10]*

Before proposing this I checked the literature so I wasn't duplicating existing work.

There's no Tamil eye-tracking-while-reading corpus — even MultiplEYE, the main multilingual eye-tracking effort, doesn't cover Tamil yet.

There's no Tamil readability or complexity dataset or classifier in DravidianLangTech, the main Tamil NLP workshop.

The closest precedent I found is a 2023 machine-learning readability model for Gujarati — that tells me this kind of approach is publishable for an Indic language, it just hasn't been done for Tamil specifically.

So this is a genuine gap, and I already have a working pipeline that's a natural fit for it.

*[ADVANCE TO SLIDE 11]*

This isn't a straight port to a new language — there's a real new question here.

From our own results, word length is already an unstable complexity proxy in English — that's the Table 1 versus Table 2 inconsistency I found in the original paper.

Tamil is agglutinative: one word can carry what English spreads across a full clause, with case, tense, and agreement all stacked onto a single word.

So word length is likely an even worse proxy there — morphemes, not characters, may be the right unit of complexity.

The plan is to build a morphology-aware complexity metric using an open-source Tamil morphological analyzer called ThamizhiMorph, and use that to drive and evaluate an LLM-based Tamil sentence simplifier, validated against human judgments.

*[ADVANCE TO SLIDE 12]*

Why I think this is the right next step: it reuses the GPT-2 surprisal pipeline I already built, not starting from scratch.

Every missing piece — a Tamil GPT-2 model, the ThamizhiMorph analyzer — already has a real, open-source option identified, so this is achievable, not speculative.

And it produces two separate deliverables: a testable linguistic claim about whether word-length-based complexity metrics transfer to agglutinative languages, and a working simplification system as an application.

It also answers a bigger question than just Tamil — the same issue applies to Turkish, Finnish, and Korean.

*[CLOSING — no slide]*

That's Phase 2 complete and Phase 3 proposed.

I'm happy to run any of the scripts again, or go deeper into whichever part you want — the GPT-2 context-window mechanics, the negative result, or the Tamil plan's feasibility.

*[END CLIP 7 — END OF RECORDING]*

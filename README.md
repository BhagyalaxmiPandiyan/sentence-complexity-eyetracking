# Measuring and Reducing Sentence Difficulty

M.Tech project. It has three phases and one toolkit:

1. **Phase 1:** replicate Singh et al. (2016) on English eye-tracking data.
2. **Phase 2:** improve the reading-time model with GPT-2 surprisal.
3. **Phase 3:** extend the work to Tamil with a morphology-aware difficulty metric, LLM simplification, and a native-speaker study.

The **toolkit** shows each phase live, on sentences the user types in.

## ▶ How to run (start here)

Everything runs on this laptop with `C:\Python313\python.exe`. Just double-click:

| Double-click | What happens | Time |
|---|---|---|
| **`START_TOOLKIT.bat`** | Starts the live demo and opens it in the browser at http://127.0.0.1:7860. Keep the black window open; close it to stop. | browser opens in ~20 s; Phase 2 and 3 ready after 1–2 min |
| **`RUN_PIPELINE.bat`** | A menu that re-runs the pipeline steps and prints the thesis numbers (see below). | 10 s – 4 min per option |

**`RUN_PIPELINE.bat` menu**

| # | Option | Prints | Time |
|---|---|---|---|
| 1 | Start the toolkit | – | – |
| 2 | Phase 1 · System 1 (step 4) | ablation table + test R²: FFD 0.444, FPD 0.558, RPD 0.555, TD 0.536 | ~10 s |
| 3 | Phase 1 · System 2 (step 5) | accuracy 74.15% / 75.26% (paper 73.82% / 75.21%) | 30–40 min |
| 4 | Phase 2 · GPT-2 surprisal | R² with GPT-2: 0.458 / 0.583 / 0.582 / 0.568 | ~10 s |
| 5 | Phase 3 · word vs morph metric (step 6) | StoryWeaver: word 63.8%, morph alone 60.5%, word + morph 63.4% | ~10 s |
| 6 | Phase 3 · webcam eye-tracking (step 11) | TD R²: word 0.103, morph 0.071, both 0.099 | ~4 min |
| 7 | Phase 3 · OasisSimp external test (step 12) | word metric 95.6%, morph metric 93.4% | ~15 s |
| 8 | Open the thesis PDF | – | – |
| 9 | Open the replication guide | – | – |

You can also run one option directly from a terminal, e.g. `RUN_PIPELINE.bat 2`. Re-running
re-creates the results files with the same numbers; the saved System 1 models used by the
toolkit are backed up and restored around option 2.

The same steps by hand, from raw data, with every command: **[REPLICATION.md](REPLICATION.md)**.
The thesis: **[thesis_report/](thesis_report/)** (PDF and Word, plus the source to rebuild it).

## Folder map

```
Final One/
├── README.md                   ← you are here
├── START_TOOLKIT.bat           ▶ double-click: live demo in the browser
├── RUN_PIPELINE.bat            ▶ double-click: re-run steps, print thesis numbers
├── REPLICATION.md              every step from raw data, with expected numbers
├── phase1_replication/         Phase 1 + Phase 2 (English)
│   ├── Research Paper 1.pdf    the paper being replicated
│   ├── src/                    Phase 1 pipeline, step1 → step5
│   ├── experiments/            Phase 2 (GPT-2) + FFD-gap investigations
│   ├── results/                summary.md + diagrams/
│   ├── output/                 generated CSVs and trained models
│   └── Dataset/                Dundee, BNC, AoA, Penn Treebank, Wiki pairs
├── phase3_tamil_extension/     Phase 3 (Tamil)
│   ├── Tamil_Readability_Thesis_Proposal.pdf
│   ├── src/                    pipeline, step0 → step12
│   ├── experiments/            feasibility pilots (tools, segmenters)
│   ├── results/                result tables (*.md, *.csv, figures)
│   ├── output/                 generated CSVs
│   ├── Dataset/                textbooks, StoryWeaver, 60 test sentences, human study
│   └── tools/                  UD Tamil treebank, IndicNLP resources
├── toolkit/                    live demo app (Gradio): app.py + one file per phase
├── Tool/                       webcam eye-tracking study apps (TamilReadingWeb = the web version used)
├── thesis_report/              MTP thesis (PDF, DOCX) + source/ to rebuild it
├── presentations/              slides for evaluations (old_presentations/ = earlier decks)
└── _archive/                   old backup and side material, not part of the thesis
```

## Phase 1: Replication (English)

**Question:** can we rebuild the paper's two systems and match its numbers?

- **System 1** predicts how long readers look at each word (4 measures: FFD, FPD, RPD, TD). It uses word length, frequency, age of acquisition and surprisal, learned from the Dundee eye-tracking corpus (10 readers, 50,647 words).
- **System 2** uses those predicted reading times to decide which of two sentences is harder (117,712 Wikipedia / Simple Wikipedia pairs).

| System 2 model | Ours | Paper |
|---|---|---|
| Base RT, pairwise | **74.15%** | 73.82% |
| Extended RT, pairwise | **75.26%** | 75.21% |

System 2 beats the paper on all 4 models. System 1 is close on FPD, RPD and TD. FFD stays lower (0.444 vs 0.649); the paper's own Table 1 and Table 2 disagree on this (see `experiments/check_wlen_r2_consistency.py`).

Run: `phase1_replication/src/step1_parse_dundee.py` → … → `step5_system2.py`.
Full tables: [phase1_replication/results/summary.md](phase1_replication/results/summary.md).

## Phase 2: GPT-2 enhancement (English)

**Question:** is GPT-2 surprisal a better predictor than the paper's parser-based surprisal?

| Measure | Before | + GPT-2 | Paper |
|---|---|---|---|
| FFD | 0.444 | 0.458 | 0.649 |
| FPD | 0.558 | 0.583 | 0.600 |
| RPD | 0.555 | **0.582** | 0.570 |
| TD | 0.536 | **0.568** | 0.516 |

GPT-2 surprisal correlates with FFD at r = 0.465, compared with −0.017 for parser surprisal. RPD and TD now beat the paper.

Run: `phase1_replication/experiments/compute_gpt2_surprisal.py`, then `train_system1_gpt2.py`.

## Phase 3: Tamil extension

**Question:** in Tamil, where words are built from many parts, do morphemes measure difficulty better than word length?

| Step | What | Result |
|---|---|---|
| A · Metric | Word-level vs morphology features on textbooks (8,004 pairs) and StoryWeaver | Morphology has signal (60.5%) but adds nothing on top of word-level (63.8%) |
| B · Simplification | 60 hard sentences simplified by an LLM | 12.8 → 6.8 words; both metrics judge 100% easier |
| C · Human study | 12 native readers, 6 ratings per pair | Difficulty 2.31 → 1.97; word ρ = +0.32, morph ρ = +0.28, no significant difference |
| C · LLM judge | Gemini as an extra judge, reported separately | Agrees with humans (+0.30) but is over-optimistic |
| C2 · Eye-tracking | Own webcam study (Tool/TamilReadingWeb), 100 Tamil sentences, 5 readers | TD R²: word 0.103, morph 0.071, word + morph 0.099 (no gain); sentence reading time ρ word 0.85, morph 0.68 |
| External test | OasisSimp Tamil: 1,904 human-simplified pairs (Liu et al., LREC 2026) | Word metric picks the complex sentence 95.6% (morph 93.4%, significantly lower) |
| D · Paper | `phase3_tamil_extension/Phase3_Research_Paper.md` (+ PDF) | Done |

Run: `phase3_tamil_extension/src/step0_…` → `step12_oasissimp.py` (details in [phase3_tamil_extension/README.md](phase3_tamil_extension/README.md)).

## Toolkit

```
cd toolkit
C:\Python313\python.exe app.py
```

Opens at `http://127.0.0.1:7860` (plus a temporary public link). Use the system Python 3.13:
its Stanza version matches the Tamil models. The Tamil and GPT-2 models load in the
background in about 1 minute.

| Tab | What the user does | Phase |
|---|---|---|
| English · Phase 1 | Types two sentences → which is harder (System 2) | 1 |
| English · Phase 2 (GPT-2) | Types one sentence → GPT-2 surprisal + predicted FFD/FPD/RPD/TD per word, with and without GPT-2 | 2 |
| Tamil · Phase 3 | ① compare two sentences · ② analyse one sentence (both metrics, morpheme breakdown) · ③ simplify with Gemini or load a study pair → compare + human ratings | 3 |

Details: [toolkit/README.md](toolkit/README.md).

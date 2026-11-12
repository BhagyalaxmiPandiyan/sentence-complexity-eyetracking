# End-to-end replication guide

This file explains how to rebuild every number in the thesis from raw data, in order.
**Quick way:** double-click `RUN_PIPELINE.bat` (re-runs the main steps and prints their numbers)
or `START_TOOLKIT.bat` (live demo); see the main [README](README.md).
Each step lists the command to run, the time it takes, the files it writes and the numbers
it should print. The folder-level READMEs have more detail:
[phase1_replication/README.md](phase1_replication/README.md),
[phase3_tamil_extension/README.md](phase3_tamil_extension/README.md),
[toolkit/README.md](toolkit/README.md).

```
 Phase 1 (English)          Phase 2 (English)            Phase 3 (Tamil)
 Dundee ─► features ─►      GPT-2 surprisal ─►           textbooks / StoryWeaver ─► features ─►
 System 1 (reading time)    System 1 again               word vs morph metric ─► LLM simplification ─►
 ─► System 2 (Wiki pairs)   (FFD/FPD/RPD/TD)             human ratings ─► webcam eye-tracking ─► OasisSimp
                    \___________________ toolkit (live demo) ___________________/
```

---

## 0. Environment

| Item | Version used | Used for |
|---|---|---|
| Windows 11 (or WSL Ubuntu for Phase 1) | — | all |
| Python | 3.13 (`C:\Python313\python.exe`) | all steps |
| Stanza | **1.14** (Tamil models downloaded with 1.14; 1.15 refuses them) | Phase 3, toolkit |
| PyTorch | CPU build | GPT-2 (English and Tamil) |
| Node.js + Microsoft Word | Node 20+, docx 9.5.1 | rebuilding the thesis only |

```bash
git clone https://github.com/BhagyalaxmiPandiyan/sentence-complexity-eyetracking.git "Final One"
cd "Final One"
C:\Python313\python.exe -m pip install -r phase1_replication/requirements.txt
C:\Python313\python.exe -m pip install -r phase3_tamil_extension/requirements.txt
C:\Python313\python.exe -m pip install -r toolkit/requirements.txt
C:\Python313\python.exe -m pip install stanza==1.14.0 indic-nlp-library openpyxl
C:\Python313\python.exe -c "import stanza; stanza.download('ta')"
```

Models downloaded automatically on first use (internet needed once): `gpt2` (English GPT-2
small) and `abinayam/gpt-2-tamil` from Hugging Face.

## 1. Data

Some datasets are licensed or personal, so they are **not** in the repository. Put each one
at the path shown.

| Dataset | Path | In repo? | Where to get it |
|---|---|---|---|
| Dundee eye-tracking corpus | `phase1_replication/Dataset/dundee_corpus/` | yes | Kennedy et al. (2003), on request from the authors |
| Age-of-acquisition norms | `phase1_replication/Dataset/AoA.csv` | yes | Kuperman et al. (2012) |
| Wikipedia / Simple Wikipedia pairs | `phase1_replication/Dataset/Wiki/aligned-good(0.67).gz` | yes | Hwang et al. (2015) |
| British National Corpus (XML) | `phase1_replication/Dataset/BNC/` | no (4.4 GB) | http://www.natcorp.ox.ac.uk/ |
| Penn Treebank 3 | `phase1_replication/Dataset/penn_treebank_3/` | no (LDC licence) | LDC99T42 |
| TN state Tamil textbooks | `phase3_tamil_extension/Dataset/graded_textbooks/` | no (copyright) | downloaded by `step0_fetch_textbooks.py` |
| StoryWeaver Tamil books | `phase3_tamil_extension/output/storyweaver_*.csv` | yes | re-downloaded by `step4_storyweaver_data.py` |
| UD Tamil treebank + IndicNLP resources | `phase3_tamil_extension/tools/UD_Tamil-TTB/`, `tools/indic_nlp_resources/` | no | `git clone https://github.com/UniversalDependencies/UD_Tamil-TTB` and `git clone https://github.com/anoopkunchukuttan/indic_nlp_resources` into `tools/` |
| Human rating responses | `phase3_tamil_extension/Dataset/human_study/responses/` | no (personal data) | own Google Form study |
| Webcam eye-tracking data | `phase3_tamil_extension/Tamil Reading Study data.xlsx` | no (personal data) | own study, `Tool/TamilReadingWeb/` |
| OasisSimp Tamil | `phase3_tamil_extension/Dataset/oasissimp/OasisSimp/OasisSimp-TA/` | no (no licence file) | https://oasissimpdataset.github.io/OasisSimp.zip, then unzip |

---

## 2. Phase 1: replicate Singh et al. (2016) (English)

Run from `phase1_replication/`. Each step reads from and writes to `output/`.

| # | Command | Time | Writes | Check |
|---|---|---|---|---|
| 1 | `python src/step1_parse_dundee.py` | ~1 min | `dundee_rt.csv` | ~307K word×reader rows, 10 readers |
| 2 | `python src/step2_lexical_features.py` | ~20 min (BNC) | `lexical_features.csv` | word length, frequency, AoA, bigram probabilities |
| 3 | `python src/step3_syntactic_features.py` | ~1 h | `syntactic_features.csv` | surprisal, embedding depth, entropy reduction |
| 4 | `python src/step4_system1.py` | ~10 s | `system1_models.pkl`, `system1_predictions.csv` | test R²: FFD 0.444, FPD 0.558, RPD 0.555, TD 0.536 (thesis Table 4.1: 0.447 / 0.561 / 0.558 / 0.538 from an earlier feature version, see Section 4) |
| 5 | `python src/step5_system2.py` | 30–40 min | accuracy table | Base RT 74.15%, Extended RT 75.26% (paper 73.82 / 75.21) |

Extra check on the FFD gap with the paper (thesis Section 4.3):
`python experiments/check_wlen_r2_consistency.py`.

## 3. Phase 2: GPT-2 surprisal (English)

Still in `phase1_replication/`. This phase needs the output of Phase 1 steps 1–4.

| # | Command | Time | Writes | Check |
|---|---|---|---|---|
| 6 | `python experiments/compute_gpt2_surprisal.py --context-sentences 3 --context-tokens 300` | ~10 min CPU | `output/gpt2_surprisal_gpt2_ctx.csv` | one surprisal value per Dundee word |
| 7 | `python experiments/train_system1_gpt2.py --surprisal-file output/gpt2_surprisal_gpt2_ctx.csv` | ~3 min | printed table | R²: FFD 0.458, FPD 0.583, RPD 0.582, TD 0.568 |
| 8 | `python experiments/gpt2_sentence_correlation.py` | ~1 min | printed table | `gpt2_sum` ≈ sentence length (r = 0.91), so don't use it |

## 4. Phase 3: Tamil extension

Run from `phase3_tamil_extension/` with `C:\Python313\python.exe`, written `py` below.

**A. Difficulty metric**

| # | Command | Time | Writes | Check |
|---|---|---|---|---|
| 9 | `py src/step0_fetch_textbooks.py` | ~10 min | `Dataset/graded_textbooks/` | Class 3–12 PDFs |
| 10 | `py src/step1_extract_sentences.py` | ~2 min | `output/graded_sentences.csv` | 2,351 sentences |
| 11 | `py src/step2_build_pairs.py` | seconds | `output/graded_pairs.csv` | 8,004 pairs |
| 12 | `py src/step3_textbook_features.py` | ~40 min | `output/sentence_features.csv` | 12 features per sentence |
| 13 | `py src/step4_storyweaver_data.py` | ~1 min | `output/storyweaver_sentences.csv` | reading levels 1–4 |
| 14 | `py src/step5_storyweaver_features.py` | ~1.5 h | `output/storyweaver_features.csv`, `storyweaver_pairs.csv` | (already in repo, so you can skip this step) |
| 15 | `py src/step6_evaluate_metric.py` and `py src/step6_evaluate_metric.py storyweaver` | ~2 min | `results/summary.md`, `results/storyweaver/summary.md` | StoryWeaver: word 63.8%, morph alone 60.5%, word+morph 63.4% |

**B. LLM simplification**

| # | Command | Writes | Check |
|---|---|---|---|
| 16 | `py src/step7_select_complex.py` | `Dataset/complex_sentences/original.csv`, `prompt_batch*.txt` | 60 complex sentences |
| — | paste each prompt into the LLM, save the replies as `Dataset/complex_sentences/simplified_<system>.txt` (prompt in thesis Appendix B) | | |
| 17 | `py src/step8_score_simplified.py` | `output/simplification_scores.csv`, `results/simplification_summary.md`, Google Form items | 12.8 → 6.8 words; both metrics say 100% easier |

**C. Human study, LLM judge, eye-tracking, external test**

| # | Command | Writes | Check |
|---|---|---|---|
| 18 | `py src/step9_human_eval.py` | `results/human_eval_summary.md` | 12 raters; difficulty 2.31 → 1.97; ρ word +0.32, morph +0.28 |
| 19 | `py src/step10_llm_judge.py` (needs `GEMINI_API_KEY`) | `results/llm_judge_*.md` | Gemini agrees with humans at about +0.30 |
| 20 | `py src/step11_eyetracking.py` | `results/eyetracking_summary.md`, `output/eyetracking_*.csv` | 5 readers, 97 sentences; TD R² word 0.103, morph 0.071, word+morph 0.099 |
| 20b | `py src/step11_eyetracking.py --with=R03` | `results/eyetracking_summary_with_R03.md` | sensitivity check with the 6th reader |
| 21 | `py src/step12_oasissimp.py` | `results/oasissimp_summary.md` | 1,904 pairs; word 95.6%, morph 93.4% (length-matched 90.5% vs 83.3%) |

Step 20 needs the exported Google Sheet from the eye-tracking web app. To run the study
again, follow [Tool/TamilReadingWeb/HOW_TO_SHARE.md](Tool/TamilReadingWeb/HOW_TO_SHARE.md).

---

## 5. Toolkit (live demo)

```bash
cd toolkit
C:\Python313\python.exe train_system2.py     # once: Phase 1 System 2 model (needs Phase 1 outputs)
C:\Python313\python.exe train_phase2.py      # once: Phase 2 models (~1 min)
C:\Python313\python.exe build_bnc_db.py     # once: BNC word-pair lookup for Phase 1 ② (~1 min)
C:\Python313\python.exe app.py               # opens http://127.0.0.1:7860
```

The trained models are already in `toolkit/models/`, so you only need `app.py`. Wait about
1 minute for the Tamil and GPT-2 models to load. Live Tamil simplification needs a
`GEMINI_API_KEY` environment variable. Without one, use the saved study pairs.

## 6. Thesis document

The source is in [thesis_report/source/](thesis_report/source/README.md). Figures come from
`make_figs.py`. The document is built with docx-js, then Word updates the contents pages
and exports the PDF:

```bash
cd thesis_report/source
npm install docx@9.5.1
python make_figs.py fig
node build.js thesis_raw.docx
powershell -ExecutionPolicy Bypass -File finalize.ps1 -In "%CD%\thesis_raw.docx" -OutDocx "%CD%\thesis.docx" -OutPdf "%CD%\thesis.pdf"
```

## 7. Where each thesis result comes from

| Thesis item | Produced by |
|---|---|
| Tables 4.x (System 1, System 2) | steps 4, 5 |
| GPT-2 tables (Chapter 4) | steps 6–8 |
| Tamil metric tables (Sections 5.1–5.3) | step 15 |
| Simplification and human study (Sections 5.4–5.5) | steps 17–19 |
| Eye-tracking (Section 5.6) | steps 20, 20b |
| OasisSimp (Section 5.7, Table 5.11) | step 21 |
| Toolkit (Chapter 6) | `toolkit/app.py` |

# Sentence Difficulty Toolkit (English + Tamil)

A self-contained web demo of **System 2** from the thesis (see
[`phase1_replication/`](../phase1_replication)): given two sentences, predicts
which one is harder to read.

It reuses the same pipeline that matched/beat the paper's numbers
(74.15% / 75.26% accuracy, see
[`phase1_replication/results/summary.md`](../phase1_replication/results/summary.md)):
System 1 predicts per-word reading time (FFD) from lexical features, and
System 2 (Logistic Regression) classifies which sentence is more complex
using those predicted reading times plus sentence-level features.

## Run locally

```
pip install -r requirements.txt
C:\Python313\python.exe app.py            # local only: http://127.0.0.1:7860
C:\Python313\python.exe app.py --share    # also a temporary public link (~72 hours)
```

The page opens in light mode so it projects well. Wait about a minute for the Tamil and
GPT-2 models to load.

## What each tab shows

| Tab | Input | What happens on screen |
|---|---|---|
| **Home** | — | The project story and the key result of each phase |
| **Phase 1 · English** | two sentences | Predicted first-fixation time for every word (bars), the word features, and System 2's verdict, plus replication results vs the paper |
| **Phase 2 · GPT-2** | one sentence | GPT-2 surprisal per word, total reading time with vs without GPT-2 (bars), all four measures, R² table |
| **Phase 3 · Tamil → ① How we did it** | — | The 8 steps of the Tamil work (data, features, metrics, tests, simplification, human study, eye-tracking, OasisSimp) with results |
| **Phase 3 · Tamil → ② Analyse a sentence** | one Tamil sentence | StoryWeaver level for both metrics, a card per word (Morfessor parts, root, grammar marks, GPT-2 surprisal, frequency, dependency), the 12 features vs typical level-1 and level-4 sentences, and the full proof tables |
| **Phase 3 · Tamil → ③ Which is harder?** | two Tamil sentences | Verdict with confidence, both metrics, feature comparison |
| **Phase 3 · Tamil → ④ Simplify** | a Tamil sentence or a study pair | Gemini simplification, level before → after for both metrics, feature changes, and the native readers' ratings |

`ui.py` holds the page design and the explanation pages; `tamil.py` and `phase2.py` the logic.

## Deploy permanently (free, Hugging Face Spaces)

For a permanent public link instead of a 72-hour temporary one:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** → choose the **Gradio** SDK → name it (e.g.
   `readability-classifier`).
3. Upload the contents of this folder (`app.py`, `requirements.txt`,
   `models/`, `data/`) to the new Space — either via the web UI's "Upload
   files", or:
   ```
   git clone https://huggingface.co/spaces/<your-username>/readability-classifier
   cp -r app.py requirements.txt models data <cloned-folder>/
   cd <cloned-folder>
   git add . && git commit -m "Add demo" && git push
   ```
4. The Space builds automatically and gives you a permanent URL:
   `https://huggingface.co/spaces/<your-username>/readability-classifier`

## Regenerating the model

`models/system2_model.pkl` is produced by `train_system2.py`, which fits
the final classifier on the full 117,712-pair dataset (one-time, not part
of the deployed app). Only needed if the underlying System 1/2 pipeline in
`phase1_replication/` changes:

```
python train_system2.py
```

This also copies the latest `system1_models.pkl` and `AoA.csv` into this
folder, keeping the demo self-contained.

## English · Phase 2 (GPT-2) tab

Type one English sentence: GPT-2 small gives each word's surprisal (thesis function
`phase1_replication/experiments/compute_gpt2_surprisal.word_token_surprisals`), and System 1
predicts FFD, FPD, RPD and TD per word with and without GPT-2 surprisal. Models are in
`models/phase2_models.pkl`, made by `train_phase2.py` (same Dundee data, split and Ridge +
skip-threshold training as `step4_system1.py`; features the app can compute live: word length,
sentence length, frequency, age of acquisition, + GPT-2 surprisal). Test R² without → with GPT-2:
FFD 0.447 → 0.450, FPD 0.560 → 0.574, RPD 0.557 → 0.573, TD 0.538 → 0.557. Needs
`phase1_replication/` next to `toolkit/`.

## Tamil tab (Phase 3)

The **Tamil · Phase 3** tab uses the thesis code in `../phase3_tamil_extension/src`
(`step3_textbook_features.py` for the 12 features, `metric_model.py` for the two metrics),
imported by `tamil.py`, so its scores are identical to the thesis
(`output/simplification_scores.csv`). Unlike the English tab it is **not self-contained**:
it needs the `phase3_tamil_extension/` folder next to `toolkit/`.

Like the English tab, it takes two sentences and says which is more complex, with a confidence.
The verdict comes from the word-level metric (the thesis's best): score(A) - score(B) is the pair
model's log-odds that A is harder, turned into a probability. The morphology-aware metric is shown
as agree / disagree. The tab has three sections: ① compare two sentences, ② analyse one sentence (both metrics, StoryWeaver level, morpheme breakdown, features), ③ simplify with Gemini (live, needs GEMINI_API_KEY) or load one of the 30 StoryWeaver study pairs with its human ratings, then compare original vs simplified with both metrics.

Examples are StoryWeaver study pairs only (textbook sentences are government copyright).
The Tamil models (Stanza, Tamil GPT-2) load in the background in about 60 s after start-up.

## Files

```
toolkit/
├── app.py               # Gradio app: English tab + Tamil tab
├── phase2.py            # Phase 2 tab logic (GPT-2 surprisal + reading times)
├── tamil.py             # Tamil tab logic (imports phase3_tamil_extension/src)
├── train_phase2.py      # One-time: trains models/phase2_models.pkl
├── train_system2.py     # One-time: trains + saves the final model (run locally, not deployed)
├── requirements.txt
├── models/
│   ├── system1_models.pkl   # Reading-time prediction model (copied from phase1_replication/output/)
│   ├── system2_model.pkl    # Readability classifier (generated by train_system2.py)
│   └── phase2_models.pkl    # Phase 2 reading-time models (generated by train_phase2.py)
└── data/
    └── AoA.csv           # Age-of-Acquisition norms (copied from phase1_replication/Dataset/)
```

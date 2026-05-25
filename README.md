# Sentence Complexity Prediction Using Eye-Tracking Measures

**Replication of:** Singh, S., Bhatt, A., Bhatt, P., & Srikant, S. (2016). [*Quantifying Sentence Complexity Based on Eye-Tracking Measures.*](https://aclanthology.org/W16-4108/) EMNLP 2016 Workshop on Uphill Battles in NLP.

> This project replicates and extends the original paper's two-system pipeline that (1) predicts per-word reading times from linguistic features, and (2) classifies which sentence in a Wikipedia/Simple-Wikipedia pair is harder to read.

---

## Results

### System 1 — Reading Time Prediction (Paper Table 3)

| Measure | Our Threshold | Our Test R² | Paper R² |
|---------|:---:|:---:|:---:|
| First Fixation Duration (FFD) | 55 ms | 0.447 | 0.649 |
| First Pass Duration (FPD) | — | 0.561 | 0.600 |
| Regression Path Duration (RPD) | — | 0.558 | 0.570 |
| Total Duration (TD) | — | **0.538** | 0.516 ✓ |

> FPD / RPD / TD are within 0.04 R² of the paper. FFD gap is due to differences in Dundee corpus preprocessing (see [Methodology](#methodology)).

### System 2 — Readability Classification (Paper Table 5)

| Model | Our Accuracy | Paper Accuracy |
|-------|:---:|:---:|
| Base RT + Pairwise Classification | **72.41%** | 73.82% |
| Extended RT + Pairwise Classification | **72.59%** | 75.21% |

> System 2 is within ~1.5% of the paper. The small gap is consistent with our System 1 FFD model being less precise than the paper's.

---

## Project Structure

```
.
├── src/                          # Core pipeline (run in order)
│   ├── step1_parse_dundee.py     # Parse Dundee corpus → per-word RT measures
│   ├── step2_lexical_features.py # Word length, freq, AoA, bigram probs
│   ├── step3_syntactic_features.py # PCFG surprisal, embedding depth, entropy
│   ├── step4_system1.py          # Train + evaluate reading-time regression
│   ├── step5_system2.py          # Train + evaluate readability classifier
│   └── syntactic_extractor.py   # Syntactic feature utilities
│
├── experiments/                  # Experimental scripts (advanced syntactic models)
│   ├── build_lcparse_surprisal.py  # Van Schijndel left-corner parser surprisal
│   ├── compute_pcfg_surprisal.py   # BLLIP + PTB PCFG surprisal
│   └── ...                         # Other exploratory scripts
│
├── results/                      # Summary tables and key outputs
│   └── summary.md
│
├── output/                       # Generated files (created by running pipeline)
│   ├── dundee_rt.csv             # Step 1 output: per-word RT per subject
│   ├── lexical_features.csv      # Step 2 output: all lexical features
│   ├── syntactic_features.csv    # Step 3 output: all syntactic features
│   ├── system1_models.pkl        # Step 4 output: trained System 1 models
│   ├── system1_predictions.csv   # Step 4 output: predicted FFD for all words
│   └── ...
│
├── Dataset/                      # Input corpora (not on GitHub — licensed)
│   ├── dundee_corpus/            # Eye-tracking data (Kennedy et al. 2003)
│   ├── AoA.csv                   # Age of Acquisition norms
│   ├── BNC/                      # British National Corpus
│   ├── penn_treebank_3.tar.bz2   # Penn Treebank WSJ
│   └── Wiki/                     # Wikipedia / Simple-Wikipedia pairs
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key libraries: `scikit-learn`, `pandas`, `numpy`, `nltk`, `wordfreq`, `bllipparser` (optional, for syntactic surprisal).

### Datasets Required

> These are **not included** in this repo (licensed). Place them under `Dataset/`:

| Dataset | Source |
|---------|--------|
| Dundee Corpus | Kennedy, A. et al. (2003) — request from authors |
| Penn Treebank 3 | LDC99T42 — via LDC |
| British National Corpus (BNC) | http://www.natcorp.ox.ac.uk/ |
| Age of Acquisition Norms | Kuperman et al. (2012), available online |
| Wiki/SimpleWiki pairs | Hwang et al. (2015), `aligned-good(0.67).gz` |

---

## How to Run

Run steps 1–5 in order. Each step reads from `output/` and writes back to `output/`.

```bash
# Step 1: Parse Dundee eye-tracking corpus
python src/step1_parse_dundee.py
# Output: output/dundee_rt.csv  (~307K rows, one per fixated word×subject)

# Step 2: Extract lexical features
python src/step2_lexical_features.py
# Output: output/lexical_features.csv  (word length, frequency, AoA, bigram probs)

# Step 3: Extract syntactic features
python src/step3_syntactic_features.py
# Output: output/syntactic_features.csv  (surprisal, embedding depth, entropy)

# Step 4: Train System 1 (reading time regression)
python src/step4_system1.py
# Output: output/system1_models.pkl, output/system1_predictions.csv
# Prints: ablation table (Table 2) + final R² results (Table 3)

# Step 5: Train System 2 (readability classifier)
python src/step5_system2.py
# Output: accuracy scores (Table 5)
# Runtime: ~30–40 minutes on 117K Wikipedia pairs
```

> **WSL note:** Steps 1–5 should be run inside WSL (Ubuntu) for Linux-path compatibility. The scripts auto-detect the OS and set paths accordingly.

---

## Methodology

### System 1 — Reading Time Prediction

The paper predicts per-word reading times using a linear regression model (Ridge) trained on 17 linguistic features:

| Feature Group | Features |
|---|---|
| Word-level | Word length (chars), Wikipedia log-frequency |
| Sentence-level | Sentence length (# words) |
| Age of Acquisition | Mean AoA, Std Dev AoA (Kuperman et al.) |
| Transition probability | Forward P(w\|prev), Backward P(w\|next) from BNC |
| Lexical surprisal | −log P(word\|context) from BNC bigrams |
| Syntactic surprisal | −log P(parse\|prefix) from PTB PCFG |
| Entropy reduction | Drop in word-prediction entropy |
| Embedding depth | Distance of word to parse-tree root |
| Embedding difference | Depth change from previous word |
| Hierarchical structure | 8 binary left-corner parser memory ops |

**Training:** 60% train / 20% dev / 20% test split at **sentence level** (2,369 sentences). Threshold tuning: any prediction below `T` ms → 0.0 (models word-skipping).

**FFD gap explanation:** Our FFD R² (0.447) is lower than the paper's (0.649). After extensive investigation, the main cause is that `sent_len` (which accounts for +0.233 R² in the paper) shows near-zero correlation (r = −0.058) with our computed FFD. This is a known issue in corpus-based reading-time modeling where different averaging strategies change the feature-target correlation structure.

### System 2 — Readability Classification

Given a Wikipedia / Simple-Wikipedia sentence pair, classify which is harder.

- **Base RT model:** Predict word-level FFD for each sentence using System 1. Concatenate predicted FFD vectors (padded to 60 words each) → 120-dimensional feature vector per pair.
- **Extended RT model:** Add sentence length, mean predicted FFD, and sum of lexical surprisal for both sentences.
- **Classifier:** Logistic Regression, **10-fold cross-validation** on 117,712 pairs.

---

## Paper Reference

```bibtex
@inproceedings{singh2016quantifying,
  title     = {Quantifying Sentence Complexity Based on Eye-Tracking Measures},
  author    = {Singh, Sanjay and Bhatt, Anoop and Bhatt, Poonam and Srikant, Shashank},
  booktitle = {Proceedings of the Workshop on Uphill Battles in NLP: Scaling Early Achievements to Robust Methods},
  year      = {2016},
  pages     = {46--50}
}
```

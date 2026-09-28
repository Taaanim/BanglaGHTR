# BanglaGHTR

**Bangla Grapheme-based Handwritten Text Recognition**

An end-to-end deep learning system for recognizing Bengali handwritten text — from a full paragraph image down to the final transcribed string.

---

## Full Paragraph Recognition Pipeline

The primary contribution of this project is a **full paragraph recognition system** that takes an unconstrained handwritten Bengali paragraph image and returns the complete transcribed text. No manual line segmentation is required.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FULL PARAGRAPH PIPELINE                                  │
│                                                                             │
│  Input: Handwritten Bengali Paragraph Image (any size)                      │
│                              │                                              │
│                ┌─────────────▼──────────────┐                               │
│                │   1. PREPROCESSING         │                               │
│                │   Grayscale conversion     │                               │
│                │   Otsu / Adaptive binarize │                               │
│                │   Hough-based deskewing    │                               │
│                └─────────────┬──────────────┘                               │
│                              │                                              │
│                ┌─────────────▼──────────────┐                               │
│                │   2. LINE SEGMENTATION     │                               │
│                │   Horizontal projection    │                               │
│                │   profiling (ink density   │                               │
│                │   per row) → valley        │                               │
│                │   detection → line boxes   │                               │
│                └─────────────┬──────────────┘                               │
│                              │  N line images                               │
│                ┌─────────────▼──────────────┐                               │
│                │   3. PER-LINE RECOGNITION  │                               │
│                │   BanglaGHTR model         │                               │
│                │   (see architecture below) │                               │
│                └─────────────┬──────────────┘                               │
│                              │  N (text, confidence) pairs                  │
│                ┌─────────────▼──────────────┐                               │
│                │   4. POST-PROCESSING       │                               │
│                │   LM-aware ensemble        │                               │
│                │   selector picks best      │                               │
│                │   decode per line          │                               │
│                └─────────────┬──────────────┘                               │
│                              │                                              │
│  Output: Full paragraph text + per-line confidence analytics                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recognition Model Architecture

Each segmented text line is recognized by **BanglaGHTR** — a 22.6M parameter model built with three Bengali-specific inductive biases:

```
Input Image [H=64px, variable width]
│
├─ STAGE 1: ConvNeXt Visual Stem (4 stages)
│   Stage 1:  [1, 64, W]   → [96,  32, W/2]   patch embed + residual block
│   Stage 2:  [96, 32, W/2] → [192, 16, W/2]   downsample + 2× residual blocks
│   Stage 3:  [192,16, W/2] → [384,  8, W/2]   downsample + 3× residual blocks
│   Stage 4:  [384, 8, W/2] → [384,  4, W/4]   downsample + 1× residual block
│
│   Dual vertical collapse:
│     MaxPool(H→1) → [B, 384, T]   ← captures matra headline (peak ink per column)
│     AvgPool(H→1) → [B, 384, T]   ← captures general ink energy per column
│     Concat + Linear(768→384) + LayerNorm → [B, T, 384]
│     (max_feat cached for Matra Attention)
│
├─ STAGE 2: Matra-Aware Attention Module
│   Structural gating using max-pool features from the stem.
│   The matra (Bengali headline) is physically encoded in max-pool activations.
│
│   gate = Sigmoid(Linear(max_feat))        ← structurally-derived, not learned
│   x    = x + gate × MultiHeadAttention(x) ← amplifies at matra positions
│   x    = x + FFN(LayerNorm(x))
│   Output: [B, T, 384]
│
├─ STAGE 3: Grapheme Mixture-of-Experts
│   3-way expert routing for Bengali grapheme categories:
│
│   Router: DepthwiseConv1d(kernel=3) → Linear → Softmax
│           (3-frame context — sees current token + neighbors)
│
│   Expert A (w_main):  Regular consonants    → direct pass-through
│   Expert B (w_diac):  Diacritics (কার, ফোলা) → DilatedConv1d(dilation=2) + FFN
│   Expert C (w_conj):  Conjuncts (যুক্তাক্ষর)  → 2× capacity FFN
│
│   fused = w_main × x + w_diac × expert_B(x) + w_conj × expert_C(x)
│   + Switch Transformer load-balancing auxiliary loss (prevents expert collapse)
│   Output: [B, T, 384]
│
├─ STAGE 4: Transformer Encoder (6 layers, pre-norm)
│   d_model=384, num_heads=8, ffn_dim=1536, dropout=0.1
│   Sinusoidal positional encoding (generalizes to unseen line widths)
│   Global context over the full character sequence
│   Output: encoder_out [B, T, 384]
│
├─ STAGE 5A: CTC Decoder
│   Linear(384 → vocab_size) + LogSoftmax → [T, B, vocab]
│   Trained with CTCLoss(blank=0, zero_infinity=True)
│   Input lengths: valid_widths / 4  (excludes padded regions)
│   Inference: prefix beam search (width=5) or greedy
│   Confidence: calibrated via per-frame peak certainty × top-2 margin × non-blank fraction
│
└─ STAGE 5B: Attention Decoder (4 causal layers + cross-attention)
    Causal self-attention (future positions masked)
    Cross-attention to encoder_out
    Teacher forcing during training, label smoothing ε=0.05
    Inference: autoregressive greedy with margin-based confidence
    Confidence: logit margin (top-1 − top-2) per step, geometric mean, repetition penalty
```

---

## Post-Processing: LM-Aware Best Prediction Selection

After each line is decoded, three decode paths run in parallel and a meta-ensemble selector picks the best:

```
Decode Paths:
  [1] CTC Beam Search   (beam_width=5)  → ctc_beam_text,   ctc_beam_conf
  [2] CTC Greedy                        → ctc_greedy_text, ctc_greedy_conf
  [3] Attention Greedy (margin-scored)  → attn_text,       attn_conf

Selector (src/postprocess/selector.py):

  Combined_Score = w_model(0.55) × model_conf
                 + w_lm(0.35)    × lm_score
                 + w_agree(0.10) × agreement_bonus
                 − penalties

  Penalties applied:
    Repetition penalty   — detects hallucinations: "কককককক"
    Length-coverage      — penalizes truncated outputs vs longest non-degenerate
    Novel N-gram         — penalizes trigrams not seen in any other decoder (hallucination)
    Prefix-superset      — penalizes outputs that are a truncated prefix of another
    Odd-one-out          — penalizes when two decoders agree and one dissents

  BanglaLM (src/postprocess/bangla_lm.py):
    Zero-dependency Bengali language model (no KenLM required)
    Tier 1: Unicode validity (no orphaned hasant, no double matra)
    Tier 2: Vocab coverage (OOV fraction penalty)
    Tier 3: Character bigram score (Laplace-smoothed, rule-derived priors)
    Tier 4: KenLM n-gram (optional — loaded if installed + path provided)
    Returns score in [0, 1]

  Output: best_text, best_source, selection_reason, per-decoder diagnostics
```

---

## Training Pipeline

Three-stage curriculum training:

| Stage | Objective | Dataset | Epochs |
|-------|-----------|---------|--------|
| 1. Character Pretraining | 122-class isolated character classification | Ekush (367K images) | 20 |
| 2. Supervised Hybrid HTR | Joint CTC + Attention loss | BN-HTRd (9,894 lines) | 60 |
| 3. SCST RL Refinement | Direct CER/WER optimization via REINFORCE | BN-HTRd | 10 |

**Stage 2 loss:** `L = 0.4 × L_CTC + 0.6 × L_Attention + 0.01 × L_MoE_AuxLoad`

**Stage 3 reward:** `R = 0.7 × (1−CER) + 0.2 × (1−WER) + 0.1 × (1−BG-CER)`

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| CER | Character Error Rate — edit distance at Unicode codepoint level |
| WER | Word Error Rate — edit distance at whitespace-tokenized word level |
| BG-CER | Bangla Grapheme CER — edit distance at grapheme cluster (akshar) level |
| NED | Normalized Edit Distance in [0,1], higher is better |
| Accuracy | Exact match fraction |

**BG-CER** is linguistically correct for Bengali: a 3-codepoint conjunct like 'ক্ষ' counts as 1 recognition unit, not 3.

---

## Project Structure

```
BanglaGHTR/
├── src/
│   ├── models/
│   │   ├── banghtr_x.py           # BanglaGHTR unified model class (BANGHTR_X_V2)
│   │   ├── vision/backbones.py    # ConvNeXt visual stem (4-stage, dual pool)
│   │   ├── grapheme/
│   │   │   ├── matra_attention.py # Matra-Aware Attention Module
│   │   │   └── diacritic_expert.py # Grapheme MoE (router + 3 experts)
│   │   ├── encoder/               # Pre-norm Transformer encoder
│   │   └── decoder/               # CTC decoder + Attention decoder (beam search)
│   ├── data/
│   │   ├── datasets.py            # PyTorch Dataset classes (line-level, char-level)
│   │   ├── transforms.py          # Augmentation pipeline (affine, elastic, noise, blur)
│   │   ├── normalizer.py          # Bengali tokenizer + Unicode NFC normalization
│   │   └── grapheme_parser.py     # Grapheme cluster decomposition for BG-CER
│   ├── training/
│   │   ├── trainer_htr.py         # Stage 2: supervised hybrid HTR trainer
│   │   ├── trainer_char.py        # Stage 1: character pretraining
│   │   └── scst_trainer.py        # Stage 3: SCST RL fine-tuning
│   ├── evaluation/
│   │   └── metrics.py             # CER, WER, BG-CER, NED, Accuracy
│   └── postprocess/
│       ├── selector.py            # LM-aware ensemble selector (10-step scoring)
│       └── bangla_lm.py           # Zero-dependency Bangla language model
│
├── webapp/                        # Single-line Gradio demo
│   ├── app.py                     # Gradio interface
│   └── inference.py               # Full inference pipeline + confidence calibration
│
├── webapp_full_paragraph/         # Full paragraph Flask web application
│   ├── app.py                     # Flask server + REST API
│   ├── pipeline.py                # End-to-end paragraph pipeline
│   ├── segmenter.py               # OpenCV line segmentation (projection profile)
│   └── templates/index.html       # Web UI (segmentation visualization + analytics)
│
├── scripts/
│   ├── train_pipeline.py          # Full 3-stage training entry point
│   ├── verify_pipeline.py         # Hardware + pipeline sanity check
│   └── generate_manifests.py      # CSV manifest generation from dataset directories
│
├── configs/                       # YAML experiment configuration files
├── datasets/manifests/            # Lightweight CSV manifests (gitignored: large CSVs)
├── exports/                       # Production model weights + vocab.json
├── BANGHTR_X_Pipeline.ipynb       # Full pipeline demonstration notebook
└── Monitor_Training.ipynb         # Training monitoring and analysis notebook
```

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify setup

```bash
python scripts/verify_pipeline.py
```

### 3. Run the paragraph web application

```bash
cd webapp_full_paragraph
python app.py
# Open http://localhost:5000
```

### 4. Run the single-line demo

```bash
cd webapp
python app.py
```

---

## Hardware

| Setting | Value |
|---------|-------|
| GPU | NVIDIA GeForce RTX 4090 (24GB VRAM) |
| Framework | PyTorch 2.0+, CUDA 11.7 |
| Precision | FP16 mixed precision (AMP) |
| Parameters | ~22.6M |

---

## Datasets

| Dataset | Task | Size |
|---------|------|------|
| BN-HTRd | Line-level HTR (document-independent split) | 14,383 lines, 150 documents |
| Ekush | Isolated character classification (writer-independent split) | 367,018 images, 5,023 writers |

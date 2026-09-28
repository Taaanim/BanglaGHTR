# BanglaGHTR

**Bangla Grapheme-based Handwritten Text Recognition** — a research-grade deep learning system for Bengali handwritten text recognition.

---

## Overview

BanglaGHTR is an end-to-end neural architecture specifically designed for the Bengali script, incorporating three domain-specific modules:

- **Matra-Aware Attention** — structurally-grounded gating using the vertical ink profile of the ConvNeXt stem to detect the Bengali headline (matra)
- **Grapheme Mixture-of-Experts** — a 3-way expert router with contextual 3-frame convolution, specialized for diacritics, conjuncts (yuktakshar), and general consonants
- **Hybrid CTC + Attention Decoder** — jointly trained with Self-Critical Sequence Training (SCST) RL refinement for direct CER/WER optimization

The system handles both single text line recognition and full paragraph recognition via an OpenCV-based line segmentation pipeline.

---

## Architecture

```
Input [B, 1, 64, W]
        │
        ▼
ConvNeXt Visual Stem (4 stages, dual max+avg pooling)
        │
        ▼
Matra-Aware Attention (structurally-gated self-attention)
        │
        ▼
Grapheme MoE Fusion (diacritic + conjunct experts, 3-frame router)
        │
        ▼
Transformer Encoder (6L, d=384, heads=8, ffn=1536)
        │
   ┌────┴────┐
   ▼         ▼
CTC Head   Attention Decoder (4L causal + cross-attn)
   │              │
   └──── Best Prediction Selection ────┘
              │
        Bengali Text
```

**Total parameters:** ~22.6M

---

## Repository Structure

```
BanglaGHTR/
├── configs/                  # Experiment configuration files
│   ├── base.yaml             # Global parameters
│   ├── htr_v2.yaml           # Main HTR config (BanglaGHTR v3)
│   ├── htr_line.yaml         # Line-level HTR config
│   └── pretrain_char.yaml    # Character pretraining config
│
├── src/                      # Source code
│   ├── models/               # Model architectures
│   │   ├── banghtr_x.py      # BanglaGHTR unified model class
│   │   ├── vision/           # ConvNeXt visual stem
│   │   ├── grapheme/         # Matra attention + MoE fusion
│   │   ├── encoder/          # Transformer encoder
│   │   └── decoder/          # CTC + Attention decoders
│   ├── data/                 # Data pipeline
│   │   ├── datasets.py       # PyTorch Dataset classes
│   │   ├── transforms.py     # Augmentation pipeline
│   │   ├── normalizer.py     # Bengali tokenizer + Unicode normalization
│   │   └── grapheme_parser.py # Grapheme cluster decomposition
│   ├── training/             # Training harnesses
│   │   ├── trainer_htr.py    # Hybrid HTR trainer (Stage 2)
│   │   ├── trainer_char.py   # Character pretrain trainer (Stage 1)
│   │   └── scst_trainer.py   # SCST RL trainer (Stage 3)
│   └── evaluation/           # Metrics (CER, WER, BG-CER, NED)
│
├── scripts/                  # CLI tools
│   ├── train_pipeline.py     # Full 3-stage training script
│   ├── verify_pipeline.py    # Hardware + pipeline verification
│   └── generate_manifests.py # Dataset manifest generation
│
├── webapp/                   # Single-line Gradio demo
│   ├── app.py                # Gradio interface
│   └── inference.py          # Model inference + best prediction selection
│
├── webapp_full_paragraph/    # Paragraph recognition Flask app
│   ├── app.py                # Flask server
│   ├── pipeline.py           # End-to-end paragraph pipeline
│   └── segmenter.py          # OpenCV line segmentation
│
├── datasets/manifests/       # CSV manifests (lightweight)
├── exports/                  # Production model weights
├── doc/project_docs/         # Technical documentation (gitignored)
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **BN-HTRd** | Bengali handwritten line images | 14,383 lines, 150 documents |
| **Ekush** | Isolated character images, 122 classes | 367,018 images, 5,023 writers |

Both datasets use strict leakage-free splits (document-independent for BN-HTRd, writer-independent for Ekush).

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify environment

```bash
python scripts/verify_pipeline.py
```

### 3. Run single-line demo

```bash
cd webapp
python app.py
```

### 4. Run paragraph demo

```bash
cd webapp_full_paragraph
python app.py
```

---

## Training

### Stage 1: Character Pretraining

```bash
python scripts/train_pipeline.py --stage pretrain_char --config configs/pretrain_char.yaml
```

### Stage 2: Supervised Hybrid HTR

```bash
python scripts/train_pipeline.py --stage htr --config configs/htr_v2.yaml
```

### Stage 3: SCST RL Refinement

```bash
python scripts/train_pipeline.py --stage rl --config configs/htr_v2.yaml --checkpoint checkpoints/best_model.pt
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CER** | Character Error Rate (edit distance / reference length) |
| **WER** | Word Error Rate |
| **BG-CER** | Bangla Grapheme Character Error Rate — edit distance at grapheme cluster level |
| **NED** | Normalized Edit Distance [0,1], higher is better |
| **Accuracy** | Exact match fraction |

---

## Hardware

- **Target GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)
- **Framework:** PyTorch 2.0+, CUDA 11.7
- **Mixed precision:** FP16 AMP

---

## Documentation

Full technical documentation is in `doc/project_docs/` (gitignored):

- `RESEARCH_PAPER.md` — full architecture, training, and evaluation details
- `PRESENTATION_SCRIPT.md` — 10-minute video presentation script with slide-by-slide narration
- `POSTER.md` — academic conference poster layout and content

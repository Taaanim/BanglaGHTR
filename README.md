# BanglaGHTR: BANGHTR-X Research Framework

**BANGHTR-X (Bangla Hierarchical Adaptive Neural Grapheme Transformer)** is an advanced, research-grade handwritten text recognition (HTR) architecture for Bengali document-, line-, and character-level recognition.

---

## 1. System Architecture (BANGHTR-X)

```text
                 HANDWRITTEN LINE / PARAGRAPH
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │          CONVNEXT VISUAL STEM                │
        │   Depthwise-separable convolutions           │
        │   Multi-scale feature pyramid                │
        │   Vertical collapse to sequence [B, T, D]    │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │         MATRA-AWARE ATTENTION MODULE         │
        │   Models horizontal headline continuity      │
        │   Headline-guided feature gating             │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        GRAPHEME MIXTURE OF EXPERTS           │
        │   Main visual feature stream                 │
        │   Diacritic expert (Kars & Folas)            │
        │   Conjunct expert (Yuktakshar)               │
        │   Dynamic 3-way softmax routing              │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       BIDIRECTIONAL SEQUENCE ENCODER         │
        │   BiLSTM contextual representation           │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │                 CTC DECODER                  │
        │   Log-probability sequence emission          │
        │   Greedy & Beam Search Decoding              │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
                       FINAL BANGLA TEXT
```

---

## 2. Directory Structure

```text
BanglaGHTR/
├── .venv/                         # Isolated project Python virtual environment
├── configs/                       # Experiment and stage configuration files
│   ├── base.yaml                  # Global parameters, paths, seeds, hardware
│   ├── pretrain_char.yaml         # Stage 1: Isolated 122-class character pretraining
│   └── htr_line.yaml              # Stage 4: Line-level sequence HTR (BANGHTR-X)
│
├── datasets/                      # Manifests and data references
│   └── manifests/                 # Master verified CSV manifests (zero data leakage)
│       ├── char_classes_122.csv   # 122 discrete classes categorized by linguistic group
│       ├── dataset_1_lines.csv    # 14,383 lines (Document-independent split)
│       ├── dataset_1_words.csv    # 108,181 words (Document-independent split)
│       └── dataset_2_chars.csv    # 367,018 characters across 5,023 writers (Writer-independent split)
│
├── Dataset/Raw_dataset/           # Clean active datasets
│   ├── BN-HTRd A Benchmark.../    # BN-HTR Ground truth (1-150) + Auto annotation (151-237)
│   └── dataset_Char/              # 122 character folders (Ekush collection)
│
├── archive/                       # Archive of raw compressed zip archives
│   └── raw_zips/                  # Preserved .zip files moved safely out of working tree
│
├── src/                           # Modular research source code
│   ├── data/                      # Datasets, transforms, normalizers, grapheme parser
│   │   ├── normalizer.py          # Unicode NFC normalizer & BengaliTokenizer
│   │   ├── transforms.py          # Aspect-ratio preserving padding & contrast inversion
│   │   ├── grapheme_parser.py     # Root, vowel kar, and fola decomposition
│   │   └── datasets.py            # PyTorch Dataset classes & CTC collator
│   │
│   ├── models/                    # Model architectures
│   │   ├── vision/                # ConvNeXt visual stem & character backbones
│   │   ├── grapheme/              # Matra attention & Diacritic/Conjunct MoE
│   │   ├── decoder/               # CTC sequence decoder & greedy search
│   │   └── banghtr_x.py           # Unified BANGHTR-X Model class
│   │
│   ├── training/                  # Training harnesses
│   │   ├── trainer_char.py        # Stage 1 pretraining harness
│   │   └── trainer_htr.py         # Stage 4 line sequence HTR harness
│   │
│   ├── evaluation/                # Metrics
│   │   └── metrics.py             # CER, WER, and BG-CER (Bengali Grapheme CER)
│   │
│   └── utils/                     # Hardware and logging utilities
│       ├── device.py              # RTX 4090 GPU detection & deterministic seeds
│       └── logger.py              # Reproducible experiment logger & env recorder
│
├── scripts/                       # Executable CLI workflows
│   ├── generate_manifests.py      # Automated manifest generation
│   └── verify_pipeline.py         # End-to-end hardware & pipeline verification
│
├── tests/                         # Automated unit & integration tests
│   └── test_setup.py
│
├── checkpoints/                   # Saved model weights (best.pt, last.pt)
├── logs/                          # Training logs and execution traces
├── outputs/                       # Evaluation predictions and confusion matrices
├── requirements.txt               # Pinned dependencies
├── .gitignore
└── README.md
```

---

## 3. Dataset Preprocessing & Zero-Leakage Protocols

### `dataset_1` (BN-HTR Document HTR)
* **Total Lines:** 14,383 lines across 150 ground-truth documents.
* **Total Words:** 108,181 words.
* **Document-Independent Split:**
  * **Train:** 105 documents (9,894 lines / 74,260 words)
  * **Validation:** 20 documents (1,914 lines / 14,833 words)
  * **Test:** 25 documents (2,575 lines / 19,088 words)
* **Aspect-Ratio Preserving Resize:** Line images are scaled to a fixed height ($H=64$) preserving horizontal aspect ratio, padded with white pixels (`255`).

### `dataset_2` (Ekush 122 Isolated Characters)
* **Total Instances:** 367,018 images ($28 \times 28$ grayscale).
* **Taxonomy:** 10 Vowel Diacritics (0–9) + 11 Basic Vowels (10–20) + 39 Consonants (21–59) + 52 Compound Conjuncts (60–111) + 10 Numerals (112–121).
* **Writer-Independent Split (5,023 Unique Writers):**
  * **Train:** 4,018 writers (292,921 character images)
  * **Validation:** 502 writers (35,801 character images)
  * **Test:** 503 writers (38,296 character images)
* **Contrast Inversion:** Images are converted from black background (`0`) with white stroke (`255`) to natural white paper (`255`) with dark stroke (`0`) via `255 - img`.

---

## 4. Hardware & Environment

* **Target GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM, Ada Lovelace, Compute Capability 8.9).
* **CUDA / PyTorch:** PyTorch 2.0.1+cu117 with mixed precision (`torch.cuda.amp.autocast`).
* **Environment:** Completely isolated inside `.venv/`. Does not modify server global packages or use `sudo`.

---

## 5. Quickstart & Verification

### Activate Environment
```bash
source .venv/bin/activate
```

### Run Sanity Verification
Verifies GPU detection, data manifests, batch loading, forward/backward pass, and CTC decoding:
```bash
python scripts/verify_pipeline.py
```

### Run Unit Tests
```bash
python tests/test_setup.py
```

### Re-generate Manifests
```bash
python scripts/generate_manifests.py
```

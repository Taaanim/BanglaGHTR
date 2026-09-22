# Deep Analysis & Preprocessing Guide: Bangla Handwritten Text Recognition (BN-HTR) & Isolated Character Datasets

> **Target Workspace:** `/Users/muntasirabdullah/Desktop/Bangla_4090`  
> **Author / Maintainer:** Antigravity AI  
> **Date:** September 2026  
> **Scope:** Complete architectural analysis, directory layout, ground truth specifications, data leakage risks, pre-training recommendations, and preprocessing protocols for `dataset_1` and `dataset_2`.

---

## 1. Executive Summary & Big Picture

This repository contains two complementary datasets designed for comprehensive Bangla (Bengali) Handwritten Document Analysis, Segmentation, and Optical Character Recognition (OCR/HTR):

```
                        BANGLA HTR PIPELINE ARCHITECTURE
                        =================================
                                
   +------------------------------------------------------------------------+
   |             dataset_1: Hierarchical Document-Level HTR                 |
   |                                                                        |
   |   [Full Document Scans]  --->  [Page Scans / Photos]                   |
   |          (PDF / TXT)                  |                                |
   |                                       v  (Line Segmentation / YOLO)    |
   |                                [Cropped Lines]                         |
   |                                       |                                |
   |                                       v  (Word Segmentation / YOLO)    |
   |                                [Cropped Words]                         |
   |                                       |                                |
   |                                       v  (Sequence HTR / CRNN / TrOCR) |
   |                             [Transcribed Bengali Text]                 |
   +---------------------------------------+--------------------------------+
                                           ^
                                           | Pretraining Feature Extractor /
                                           | Character Classifier Weights
   +---------------------------------------+--------------------------------+
   |          dataset_2: 122-Class Isolated Character Recognition           |
   |                                                                        |
   |   10 Vowel Diacritics (Kars)  +  11 Independent Vowels                 |
   |   39 Consonants              +  52 Compound Characters (Yuktakshar)    |
   |   10 Bengali Numerals (0-9)                                            |
   |   Total: 367,018 full images (28x28 grayscale) | 30,500 balanced       |
   +------------------------------------------------------------------------+
```

### High-Level Comparison Table

| Dimension | `dataset_1` (BN-HTR & Automatic Annotation) | `dataset_2` (Ekush Isolated Characters) |
| :--- | :--- | :--- |
| **Primary Purpose** | End-to-end Document Layout Analysis, Line/Word Segmentation, and Sequence HTR | Isolated Single-Character / Glyphic Classification & Backbone Pretraining |
| **Hierarchical Granularity** | Document $\rightarrow$ Page $\rightarrow$ Line $\rightarrow$ Word $\rightarrow$ Character text | Individual isolated character glyphs (28 $\times$ 28 px) |
| **Total Disk Size** | ~8.62 GB (uncompressed + zip archives) | ~0.46 GB (uncompressed images) |
| **Total File Count** | 310,789 files | 397,523 files |
| **Components** | `BN-HTR_Dataset` (verified ground truth: 1-150)<br>`Automatic_Annotation` (pseudo-labeled: 151-237)<br>`Sample_Small` (mini-batch debugging: 1, 50, 100) | `dataset` (full collection: 367,018 images)<br>`dataset_filtered` (balanced subset: 30,500 images) |
| **Total Pages / Documents** | 150 manual docs (725 pages) + 87 auto docs (778 pages) | 122 discrete classes across >3,040 human writers |
| **Total Lines** | 14,384 (manual verified) + 14,836 (auto) | N/A (isolated characters only) |
| **Total Words** | 108,181 (manual cropped) + 105,958 (auto cropped) | N/A |
| **Transcribed Words** | 107,794 transcribed instances (14,563 unique vocabulary) | 367,018 labeled character instances |
| **Image Resolution & Format** | Full pages: ~2400 $\times$ 2200 to 2400 $\times$ 3300 px (RGB)<br>Lines: ~1800 $\times$ 150 px (RGB)<br>Words: ~150 $\times$ 70 px (RGB)<br>Formats: `.jpg`, `.JPG`, `.png`, `.PNG`, `.jpeg` | Fixed 28 $\times$ 28 pixels (Grayscale, 1-channel)<br>Format: `.jpg` |
| **Color Space & Contrast** | Standard document contrast: **Light/white background** (~240–255), **dark ink strokes** (~0–80) | MNIST contrast: **Inverted / Black background** (~0), **white character strokes** (~255) |
| **Annotation Formats** | Pascal VOC (`.xml`), YOLO normalized boxes (`.txt`), Excel alignments (`.xlsx`), Raw corpus (`.txt`, `.pdf`) | Folder name = Class ID (`0` to `121`), Filename = Demographics (`Gender_District_Age_Form_Writer.jpg`), `metaData_img.csv` |

---

## 2. Deep Structural Analysis of `dataset_1`

### 2.1 Overview & Structure (`Structure_of_Directory.pdf`)

`dataset_1` contains the complete handwritten document hierarchy, originally structured for academic research on Bangla Handwritten Text Recognition (BN-HTR). The directory contains:

```
dataset_1/
├── Structure_of_Directory.pdf      # Visual directory tree documentation
├── BN-HTR_Dataset.zip              # Archive of the human-verified ground truth (2.61 GB)
├── BN-HTR_Dataset/                 # Uncompressed human-verified ground truth (Docs 1-150)
├── Automatic_Annotation.zip        # Archive of machine-predicted detections (1.85 GB)
├── Automatic_Annotation/           # Uncompressed model predictions (Docs 151-237)
├── Sample_Small.zip                # Lightweight archive for debugging (76 MB)
└── Sample_Small/                   # Uncompressed subset: Docs 1, 50, 100
```

---

### 2.2 Sub-directory 1: `BN-HTR_Dataset` (Human-Verified Ground Truth)

This is the **gold standard ground truth** of the dataset. Every bounding box has `<annotation verified="yes">` and every word is human-transcribed.

```
BN-HTR_Dataset/
├── Recognition_Ground_Truth_Texts/
│   ├── 1/
│   │   ├── 1.txt                   # Complete raw source text of Document 1
│   │   ├── 1.pdf                   # Original printed/clean document source
│   │   └── 1.xlsx                  # Alignment table mapping Word IDs to Bengali text
│   ├── 2/ ...
│   └── 150/
└── Segmentation_Images/
    ├── Lines/
    │   ├── 1/
    │   │   ├── classes.txt         # Classes: [word, line]
    │   │   ├── 1_1.jpg             # Page 1 image of Document 1 (RGB scan/photo)
    │   │   ├── 1_1.xml             # Pascal VOC bounding boxes of lines on Page 1_1
    │   │   ├── 1_1.txt             # YOLO-format normalized bounding boxes of lines
    │   │   ├── 1_2.jpg, 1_2.xml, 1_2.txt
    │   │   ├── 1_1/                # Subdirectory containing cropped lines for Page 1_1
    │   │   │   ├── 1_1_1.jpg       # Cropped image of Line 1
    │   │   │   ├── 1_1_2.jpg       # Cropped image of Line 2
    │   │   │   └── ...
    │   │   └── 1_2/ ...
    │   ├── 2/ ...
    │   └── 150/
    └── Words/
        ├── 1/
        │   ├── 1_1/                # Corresponds to Page 1_1
        │   │   ├── classes.txt     # Classes: [word, line]
        │   │   ├── 1_1_1.jpg       # Image of Line 1
        │   │   ├── 1_1_1.xml       # Pascal VOC bounding boxes of words within Line 1
        │   │   ├── 1_1_1.txt       # YOLO-format normalized bounding boxes of words
        │   │   ├── 1_1_2.jpg, 1_1_2.xml, 1_1_2.txt
        │   │   ├── 1_1_1/          # Subdirectory containing cropped words for Line 1_1_1
        │   │   │   ├── 1_1_1_1.jpg # Cropped image of Word 1
        │   │   │   ├── 1_1_1_2.jpg # Cropped image of Word 2
        │   │   │   └── ...
        │   │   └── 1_1_2/ ...
        │   └── 1_2/ ...
        ├── 2/ ...
        └── 150/
```

#### Hierarchical ID Syntax

Every element in `BN-HTR_Dataset` adheres to a strict, dot-free naming convention:

$$\text{ID} = \mathbf{DocID\_PageID\_LineID\_WordID}$$

| Entity | ID Example | File Location | Meaning |
| :--- | :--- | :--- | :--- |
| **Document** | `1` | `Recognition_Ground_Truth_Texts/1/` | Document #1 |
| **Page** | `1_1` | `Segmentation_Images/Lines/1/1_1.jpg` | Document 1, Page 1 |
| **Line** | `1_1_1` | `Segmentation_Images/Lines/1/1_1/1_1_1.jpg` | Document 1, Page 1, Line 1 |
| **Word** | `1_1_1_1` | `Segmentation_Images/Words/1/1_1/1_1_1/1_1_1_1.jpg` | Document 1, Page 1, Line 1, Word 1 |

#### Ground Truth Alignment Table (`.xlsx`)

In `Recognition_Ground_Truth_Texts/{DocID}/{DocID}.xlsx`, rows map `Id` to `Word`:

```
+-----------+---------------+
|    Id     |     Word      |
+-----------+---------------+
| 1_1_1_1   | কথা           |
| 1_1_1_2   | প্রকাশ         |
| 1_1_2_1   | বৈচিত্র্যময়    |
| 1_1_2_2   | এই            |
| 1_1_2_3   | পৃথিবীর       |
+-----------+---------------+
```

- **Reconstructing Line Text:** Group rows by `_`.join(Id.split('_')[:3]) (e.g. `1_1_1`), sort by word index (`parts[3]`), and join with spaces:
  $$\text{Line 1\_1\_1} = \text{"কথা প্রকাশ"}$$
- **Reconstructing Page Text:** Group rows by `_`.join(Id.split('_')[:2]) (e.g. `1_1`).

#### Annotation Formats Provided

1. **Pascal VOC XML (`.xml`)**:
   - Contains: `<size><width>, <height>, <depth></size>`
   - Objects: `<name>line</name>` or `<name>word</name>`
   - Pixel coordinates: `<bndbox><xmin>, <ymin>, <xmax>, <ymax></bndbox>`
2. **YOLO TXT (`.txt`)**:
   - Coordinates: `<class_id> <x_center> <y_center> <width> <height>` (normalized $0.0 \dots 1.0$)
   - Class IDs: Defined in `classes.txt`: `0 = word`, `1 = line`.

---

### 2.3 Sub-directory 2: `Automatic_Annotation` (Machine-Generated Pseudo-Labels)

Documents **151 through 237** (87 documents total) contain automatically generated predictions created by running an object detector (e.g. YOLO) on scanned handwritten documents:

```
Automatic_Annotation/
├── line_resilts.txt        # Summary log: 805 doc images -> 14,836 lines detected
├── word_results.txt        # Summary log: 14,800 lines -> 105,958 words detected
├── 151/
│   ├── 151.txt             # Raw article text (e.g. BBC News Bangla article)
│   ├── 151.pdf             # Original document PDF
│   ├── 151_1.jpg           # Page 1 scan
│   ├── 151_2.jpg ...
│   ├── lines/
│   │   ├── images/         # Cropped line images organized in page folders
│   │   └── labels/         # YOLO-format detections with CONFIDENCE SCORES
│   └── words/
│       ├── images/         # Cropped word images organized in line folders
│       └── labels/         # YOLO-format detections with CONFIDENCE SCORES
└── 152/ ... 237/
```

> [!IMPORTANT]
> **Key Differences between `BN-HTR_Dataset` and `Automatic_Annotation`:**
> 1. **No Excel Alignment:** `Automatic_Annotation` has **zero** `.xlsx` files. The word images have NOT been aligned to text transcripts.
> 2. **Confidence Scores in Labels:** In `Automatic_Annotation`, label files contain 6 columns:
>    `<class> <x_center> <y_center> <width> <height> <confidence>` (e.g. `0 0.515 0.131 0.833 0.056 0.915242`).
> 3. **Role in Training:** Never use `Automatic_Annotation` as a test or validation benchmark! It is designed for **semi-supervised pretraining**, **pseudo-labeling / self-training**, or **contrastive pretraining**.

---

### 2.4 Critical Hidden Anomalies & Traps in `dataset_1`

Before running any script or data loader, you must account for these 5 findings:

> [!WARNING]
> 1. **Typo in Document 60 Excel Name:**
>    - In `BN-HTR_Dataset/Recognition_Ground_Truth_Texts/60/`, the file is named **`60.xl.xlsx`** instead of `60.xlsx`. A naive check for `f"{doc_id}.xlsx"` will fail or silently skip Document 60. Always search with glob: `[f for f in os.listdir(doc_path) if f.endswith('.xlsx')]`.
> 2. **Mixed Image File Extensions:**
>    - Image extensions are heterogeneous across folders: `{'.jpg', '.JPG', '.png', '.PNG', '.jpeg'}`.
>    - Documents 18, 95, and parts of 9 and 81 use `.png` and `.jpeg`! Any loader hardcoded to `.jpg` will drop multiple pages and hundreds of lines.
> 3. **Orphan XML Annotations:**
>    - Exactly 13 XML annotation files in `Segmentation_Images/Lines/` do not have a corresponding `.jpg` file in the same folder because the source image was saved as `.png` (e.g. `95_1.xml` matches `95_1.png`).
> 4. **Missing PDF Documents:**
>    - 23 documents in `BN-HTR_Dataset` lack a `.pdf` file (they have `.txt` and `.xlsx`). Pipelines should never mandate the presence of `.pdf`.
> 5. **Unicode Normalization Mismatch:**
>    - Python standard strings and raw Excel strings contain unnormalized Bengali diacritics and ligatures. Applying `unicodedata.normalize('NFC', text)` changes string length and prevents tokenization out-of-vocabulary (OOV) errors.

---

## 3. Deep Structural Analysis of `dataset_2`

### 3.1 Overview & Origin

`dataset_2` is an isolated handwritten Bangla character dataset derived from the **Ekush** dataset (developed by Shah et al.). It consists of:

```
dataset_2/
├── metaData_img.csv            # Mapping: Folder Name (0-121) -> Char Name
├── Untitled spreadsheet.xlsx   # Ekush cross-reference sheets & compound characters
├── dataset/                    # Full dataset: 122 classes, 367,018 images
│   ├── 0/                      # Class 0: ' া' (Vowel Kar Aa) -> 3,047 images
│   ├── 1/                      # Class 1: ' ি' (Vowel Kar I)  -> 3,086 images
│   ├── ...
│   └── 121/                    # Class 121: '৯' (Bengali Digit 9) -> 3,064 images
└── dataset_filtered/           # Perfectly balanced benchmark: 122 classes, 250 images each
    ├── 0/                      # Exactly 250 images
    ├── 1/                      # Exactly 250 images
    └── ... 121/                # Total: 30,500 images
```

---

### 3.2 The Complete 122-Class Taxonomy

The classes are numbered sequentially from `0` to `121` and categorized into 5 linguistic groups:

| Category | Folder IDs | Count | Examples / Description |
| :--- | :--- | :--- | :--- |
| **Vowel Diacritics (কার / Kars)** | `0` – `9` | 10 | া (Aa), ি (I), ী (Ee), ু (U), ূ (Oo), ৃ (Rri), ে (E), ৈ (Oi), ো (O), ৌ (Ou) |
| **Independent Vowels (স্বরবর্ণ)** | `10` – `20` | 11 | অ, আ, ই, ঈ, উ, ঊ, ঋ, এ, ঐ, ও, ঔ |
| **Consonants (ব্যঞ্জনবর্ণ)** | `21` – `59` | 39 | ক, খ, গ, ঘ, ঙ, চ, ছ, জ, ঝ, ঞ, ট, ঠ, ড, ঢ, ণ, ত, থ, দ, ধ, ন, প, ফ, ব, ভ, ম, য, র, ল, শ, ষ, স, হ, ড়, ঢ়, য়, ৎ, ং, ঃ, ঁ |
| **Compound Characters (যুক্তবর্ণ)** | `60` – `111` | 52 | শব্দ (ব্দ), অঙ্গ (ঙ্গ), স্কুল (স্ক), স্ফীতি (স্ফ), ইচ্ছা (চ্ছ), স্থান (স্থ), রক্ত (ক্ত), স্নান (স্ন), কৃষ্ণ (ষ্ণ), ক্ষ, ক্ত, জ্ঞ, ঙ্ক, জ্ব, ঞ্জ, দ্ধ, ন্ন, ঘ্ন, ক্ল, হ্ন, স্প, ল্ত, ইত্যাদি |
| **Bengali Numerals (সংখ্যা)** | `112` – `121` | 10 | ০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯ |

**Total:** $10 + 11 + 39 + 52 + 10 = \mathbf{122\text{ classes}}$.

---

### 3.3 Filename Metadata Grammar & Demographic Properties

Every image in `dataset_2` follows a standardized 5-part naming convention:

$$\mathbf{\{Gender\}\_\{District\}\_\{AgeGroup\}\_\{FormID\}\_\{WriterID\}.jpg}$$

Example: `1_DHA_12_1_712.jpg`

- **`Gender` (Token 0):** `0` = Male, `1` = Female.
  - Across Class 0: **1,508 Male** vs **1,533 Female** (near-perfect 50/50 balance).
- **`District` (Token 1):** District code where sample was collected.
  - 153 distinct district identifiers.
  - Top districts: `DHA` (Dhaka: 926), `BAR` (Barishal: 202), `COM` (Comilla: 163), `CHAD` (Chandpur: 111), `NOA` (Noakhali: 92), `TAN` (Tangail: 83), etc.
- **`AgeGroup` (Token 2):** Primary writer ages (peak between 11 and 16 years, ranging up to 24+).
- **`FormID` (Token 3):** Form template version used by the respondent.
- **`WriterID` (Token 4):** Unique human writer identifier (>3,040 unique writers).

> [!CAUTION]
> ### Major ML Trap: Data Leakage via Random Splitting!
> Because the same writer filled out a sheet containing all 122 characters, the filename `1_DHA_12_1_712.jpg` appears in folder `0`, folder `1`, folder `10`, folder `60`, etc.
> 
> If you perform a standard `train_test_split(shuffle=True)` on image files:
> - Samples from Writer `712` will appear in **both** training and test sets.
> - The model will overfit to individual handwriting styles rather than generalizable character morphologies.
> - **Mandatory Fix:** Always split the dataset by **`WriterID`** (Writer-Independent Split), ensuring that no writer present in the test set has any samples in the training set!

---

### 3.4 Image Format & Contrast (Inversion Alert)

- **Shape:** $28 \times 28$ pixels.
- **Color Mode:** Grayscale (`L` mode, uint8).
- **Background Pixel Values:** $0 \dots 5$ (Solid Black).
- **Stroke Pixel Values:** $180 \dots 255$ (Bright White).
- **Implication:** `dataset_2` is pre-inverted like MNIST. If using vision backbones pretrained on ImageNet (which expect natural contrast: dark ink on white paper), either:
  1. Invert the image: `inverted = 255 - img`
  2. Or adapt your normalization pipeline accordingly.

---

## 4. What To Do FIRST Before Training (Step-by-Step Preprocessing Protocol)

Follow this rigorous sequence of operations before training any model:

```
+-------------------------------------------------------------------------------+
|                        PRE-TRAINING CHECKLIST & ROADMAP                       |
+-------------------------------------------------------------------------------+
|  [Step 1] Data Cleansing & Master Manifest Generation                         |
|           • Build CSV manifests for pages, lines, words, and characters.      |
|           • Handle case-insensitivity (.jpg vs .png) & the 60.xl.xlsx typo.   |
|                                                                               |
|  [Step 2] Writer-Independent Splitting (Zero Data Leakage)                    |
|           • dataset_1: Split strictly by Document ID (1-150).                 |
|           • dataset_2: Split strictly by Writer ID.                           |
|                                                                               |
|  [Step 3] Text & Unicode Normalization                                        |
|           • Normalize all transcripts with unicodedata.normalize('NFC').      |
|           • Build character-level vocabulary with <BLANK>, <UNK>, <PAD>.      |
|                                                                               |
|  [Step 4] Image Preprocessing & Augmentation Strategy                         |
|           • dataset_1: Aspect-ratio preserving resize + padding + Sauvola.   |
|           • dataset_2: Contrast inversion check + padding/upscaling.          |
|                                                                               |
|  [Step 5] Model Selection & Transfer Learning Execution                       |
|           • Layout: YOLOv8/v11 on Lines & Words.                              |
|           • Backbone: Pretrain CNN on dataset_2 (122 classes).                |
|           • HTR: CRNN / TrOCR on cropped lines/words in dataset_1.            |
+-------------------------------------------------------------------------------+
```

---

### Step 1: Master Manifest Generation

Do not scan thousands of directories during every PyTorch epoch. Generate centralized CSV manifest files upfront:

1. **`page_manifest.csv`**: Contains `page_id, doc_id, image_path, xml_path, yolo_path, num_lines, width, height`.
2. **`line_manifest.csv`**: Contains `line_id, page_id, doc_id, image_path, text_transcription, num_words, width, height`.
3. **`word_manifest.csv`**: Contains `word_id, line_id, page_id, doc_id, image_path, text_transcription, width, height`.
4. **`char_manifest.csv`**: Contains `image_path, class_id, char_name, category, gender, district, age, writer_id, is_filtered`.

*(A complete, self-contained Python script to generate these manifests is provided in Section 6).*

---

### Step 2: Leakage-Free Dataset Splitting

#### For `dataset_1`: Document-Based Split
- Documents 1 to 150 represent different handwriting sessions / authors.
- **Recommended Split (150 documents):**
  - **Train:** 105 documents (~70%, ~75,000 words, ~10,000 lines)
  - **Validation:** 20 documents (~13%, ~15,000 words, ~2,000 lines)
  - **Test:** 25 documents (~17%, ~18,000 words, ~2,400 lines)
- **Automatic Annotation (Docs 151–237):** Keep strictly separate. Use only as unlabeled/pseudo-labeled data for semi-supervised training or self-training with Teacher-Student models.

#### For `dataset_2`: Writer-Independent Split
- Extract all unique `writer_id` values (from filename token 4).
- Partition the unique writers into:
  - **Train:** 80% of writers
  - **Validation:** 10% of writers
  - **Test:** 10% of writers
- All characters written by a specific person will only exist in one partition.

---

### Step 3: Unicode Normalization & Vocabulary Construction

Bengali text contains complex combining characters (hasant `্`, nukta `়`, vowel signs, and zero-width joiners `\u200c`, `\u200d`). 

1. **Mandatory NFC Normalization:**
   ```python
   import unicodedata
   clean_text = unicodedata.normalize('NFC', raw_text.strip())
   ```
2. **Special Tokens for CTC Loss / Sequence Modeling:**
   ```python
   SPECIAL_TOKENS = {
       '<PAD>': 0,
       '<BLANK>': 1,    # Mandatory for CTC blank symbol
       '<UNK>': 2,
       '<SOS>': 3,
       '<EOS>': 4,
   }
   ```
3. **Vocabulary Size:**
   - Word ground truth in `dataset_1` contains 74 distinct Bengali Unicode characters + punctuation + digits. Total character vocabulary size: **~110 tokens**.

---

### Step 4: Image Preprocessing & Augmentation Protocols

#### For `dataset_1` (Cropped Lines & Words)
- **Aspect Ratio Preservation:** Never stretch lines or words to a square!
  - **Fixed Height Resizing:** Resize height to 64 px (or 128 px for high-res), scale width proportionally:
    $$W_{\text{new}} = \min\left(\text{int}\left(W_{\text{orig}} \times \frac{H_{\text{target}}}{H_{\text{orig}}}\right), W_{\text{max}}\right)$$
  - Pad remaining width to $W_{\text{max}}$ with white pixels (`255`).
- **Illumination & Binarization:** Apply adaptive local thresholding (Sauvola binarization or Otsu thresholding) to eliminate shadow artifacts from smartphone cameras and book folds.
- **Augmentation Techniques:**
  - Random slight rotation ($\pm 3^\circ$) for deskewing resilience.
  - Affine shear along X-axis ($\pm 10^\circ$) to simulate varied handwriting slants.
  - Morphological dilation and erosion (simulating thick marker vs thin ballpoint pens).
  - Gaussian blur ($\sigma \in [0.5, 1.2]$) and brightness jitter.

#### For `dataset_2` (28x28 Isolated Characters)
- **Input Inversion / Normalization:**
  - If training a lightweight CNN from scratch: Keep the native 28 $\times$ 28 black background / white stroke format. Normalize pixel values to $[0.0, 1.0]$.
  - If fine-tuning a pretrained model (e.g. ResNet18, EfficientNet):
    1. Upsample bicubic to $64 \times 64$ or $112 \times 112$.
    2. Convert 1-channel to 3-channel (`torch.cat([x, x, x], dim=0)`).
    3. Invert if your backbone was trained on white paper documents.
- **Data Augmentation:**
  - Random affine transformations: Rotation ($\pm 10^\circ$), translation ($\pm 2$ px), scale ($0.9 \dots 1.1$).

---

## 5. Concrete Model Architectures & Training Recipes

| Task | Input | Output | Recommended Architecture | Loss Function | Dataset Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Page $\rightarrow$ Line Detection** | Full Page Scan ($2400 \times 2200$) | Bounding boxes for each line | **YOLOv8x / YOLOv11x Detection** or **DBNet++** | GIoU + Classification + DFL | `dataset_1/Segmentation_Images/Lines` |
| **Task 2: Line $\rightarrow$ Word Detection** | Cropped Line Image ($1800 \times 150$) | Bounding boxes for each word | **YOLOv8m Detection** or **CRAFT** | GIoU + Classification | `dataset_1/Segmentation_Images/Words` |
| **Task 3: Isolated Character Recognition** | $28 \times 28$ Grayscale Glyph | Class ID ($0 \dots 121$) | **ResNet18 / EfficientNet-B0 / MobileNetV3** | Cross-Entropy Loss | `dataset_2/dataset` & `dataset_filtered` |
| **Task 4: Word-Level Recognition** | Cropped Word Image ($150 \times 70$) | Bengali Word String | **CRNN (ResNet + BiLSTM + CTC)** or **TrOCR-Small** | CTC Loss / Cross-Entropy | `dataset_1/Segmentation_Images/Words` + `X.xlsx` |
| **Task 5: Line-Level Recognition** | Cropped Line Image ($1800 \times 150$) | Bengali Line String | **TrOCR-Base** or **PARSeq** or **SVTR** | CTC Loss / Sequence Cross-Entropy | `dataset_1/Segmentation_Images/Lines` + Reconstructed lines |
| **Task 6: Semi-Supervised Self-Training** | Unlabeled Line/Word Images | Pseudo-labels & Feature weights | **Teacher-Student Consistency (FixMatch)** | Consistency Loss + CTC | `dataset_1/Automatic_Annotation` (Docs 151-237) |

---

## 6. Ready-to-Run Production Helper Scripts

### Script 1: Master Manifest Generator (`build_manifests.py`)

Run this script once to produce indexed CSV files for all pages, lines, words, and character classes:

```python
"""
build_manifests.py
Generates unified metadata manifests for dataset_1 and dataset_2.
Handles 60.xl.xlsx, mixed extensions, and creates train/val/test splits.
"""

import os
import glob
import random
import unicodedata
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET

WORKSPACE_ROOT = "/Users/muntasirabdullah/Desktop/Bangla_4090"
D1_ROOT = os.path.join(WORKSPACE_ROOT, "dataset_1")
D2_ROOT = os.path.join(WORKSPACE_ROOT, "dataset_2")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "manifests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG')

def get_xlsx_words(xlsx_path):
    """Extracts Id and Word pairs from an .xlsx file using standard zipfile/xml."""
    entries = {}
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
                for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    t = si.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
                    if t is not None and t.text:
                        strings.append(t.text)
                    else:
                        strings.append(''.join([c.text for c in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if c.text]))
            
            tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
            rows = tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
            for r in rows:
                cells = r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c')
                row_vals = []
                for c in cells:
                    t_attr = c.attrib.get('t')
                    v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    if v is not None and v.text is not None:
                        val = v.text
                        if t_attr == 's' and val.isdigit() and int(val) < len(strings):
                            val = strings[int(val)]
                        row_vals.append(val)
                if len(row_vals) >= 2 and row_vals[0] != 'Id':
                    w_id = row_vals[0].strip()
                    word = unicodedata.normalize('NFC', row_vals[1].strip())
                    entries[w_id] = word
    except Exception as e:
        print(f"Warning reading {xlsx_path}: {e}")
    return entries

def build_dataset_1_manifests():
    print("Building manifests for dataset_1...")
    bn_root = os.path.join(D1_ROOT, "BN-HTR_Dataset")
    gt_dir = os.path.join(bn_root, "Recognition_Ground_Truth_Texts")
    lines_dir = os.path.join(bn_root, "Segmentation_Images", "Lines")
    words_dir = os.path.join(bn_root, "Segmentation_Images", "Words")
    
    # 1. Load all ground truth word texts
    doc_words = {}
    for d in range(1, 151):
        dp = os.path.join(gt_dir, str(d))
        if not os.path.exists(dp):
            continue
        xlsx_candidates = [f for f in os.listdir(dp) if f.endswith('.xlsx')]
        if xlsx_candidates:
            xlsx_path = os.path.join(dp, xlsx_candidates[0])
            doc_words[str(d)] = get_xlsx_words(xlsx_path)
    
    # 2. Build Document Split (70/13/17)
    random.seed(42)
    all_docs = [str(i) for i in range(1, 151)]
    random.shuffle(all_docs)
    train_docs = set(all_docs[:105])
    val_docs = set(all_docs[105:125])
    test_docs = set(all_docs[125:])
    
    # 3. Build Word Manifest
    word_records = []
    line_records = {}
    page_records = []
    
    for doc_id in range(1, 151):
        d_str = str(doc_id)
        split_name = "train" if d_str in train_docs else ("val" if d_str in val_docs else "test")
        words_doc_dir = os.path.join(words_dir, d_str)
        lines_doc_dir = os.path.join(lines_dir, d_str)
        
        if not os.path.exists(words_doc_dir):
            continue
            
        page_folders = [f for f in os.listdir(words_doc_dir) if os.path.isdir(os.path.join(words_doc_dir, f))]
        for page_id in page_folders:
            p_dir = os.path.join(words_doc_dir, page_id)
            line_folders = [f for f in os.listdir(p_dir) if os.path.isdir(os.path.join(p_dir, f))]
            for line_id in line_folders:
                l_dir = os.path.join(p_dir, line_id)
                word_files = [f for f in os.listdir(l_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
                
                line_words_list = []
                for wf in word_files:
                    wid = os.path.splitext(wf)[0]
                    text = doc_words.get(d_str, {}).get(wid, "")
                    w_path = os.path.join(l_dir, wf)
                    word_records.append({
                        "word_id": wid,
                        "line_id": line_id,
                        "page_id": page_id,
                        "doc_id": d_str,
                        "split": split_name,
                        "image_path": w_path,
                        "text": text
                    })
                    if text:
                        line_words_list.append((wid, text))
                
                # Line transcription
                line_img_candidates = [
                    os.path.join(lines_doc_dir, page_id, f"{line_id}{ext}")
                    for ext in ['.jpg', '.png', '.jpeg']
                ]
                line_img_path = next((p for p in line_img_candidates if os.path.exists(p)), "")
                
                # Sort words by last index
                sorted_words = [w[1] for w in sorted(line_words_list, key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else 0)]
                line_text = " ".join(sorted_words)
                
                line_records[line_id] = {
                    "line_id": line_id,
                    "page_id": page_id,
                    "doc_id": d_str,
                    "split": split_name,
                    "image_path": line_img_path,
                    "text": line_text,
                    "num_words": len(sorted_words)
                }

    pd.DataFrame(word_records).to_csv(os.path.join(OUTPUT_DIR, "dataset_1_words.csv"), index=False)
    pd.DataFrame(list(line_records.values())).to_csv(os.path.join(OUTPUT_DIR, "dataset_1_lines.csv"), index=False)
    print(f"Saved {len(word_records)} words to dataset_1_words.csv")
    print(f"Saved {len(line_records)} lines to dataset_1_lines.csv")

def build_dataset_2_manifests():
    print("Building manifests for dataset_2...")
    ds_path = os.path.join(D2_ROOT, "dataset")
    meta_df = pd.read_csv(os.path.join(D2_ROOT, "metaData_img.csv"))
    class_map = {str(row['Folder Name']): row['Char Name'].strip() for _, row in meta_df.iterrows()}
    
    # 1. Discover all images and extract unique writers
    records = []
    writers = set()
    
    for c_dir in sorted(os.listdir(ds_path), key=lambda x: int(x) if x.isdigit() else 999):
        cp = os.path.join(ds_path, c_dir)
        if not os.path.isdir(cp): continue
        for fn in os.listdir(cp):
            if fn.endswith('.jpg'):
                parts = fn[:-4].split('_')
                if len(parts) == 5:
                    gender, dist, age, form, writer_id = parts
                    w_uid = f"{gender}_{dist}_{writer_id}"
                    writers.add(w_uid)
                    records.append({
                        "filename": fn,
                        "class_id": int(c_dir),
                        "char_name": class_map.get(c_dir, ""),
                        "gender": int(gender),
                        "district": dist,
                        "age": int(age) if age.isdigit() else age,
                        "writer_id": w_uid,
                        "image_path": os.path.join(cp, fn)
                    })
    
    # 2. Writer-Independent Split (80/10/10)
    random.seed(42)
    writers_list = list(writers)
    random.shuffle(writers_list)
    n_w = len(writers_list)
    train_w = set(writers_list[:int(n_w * 0.8)])
    val_w = set(writers_list[int(n_w * 0.8):int(n_w * 0.9)])
    
    for r in records:
        w = r["writer_id"]
        r["split"] = "train" if w in train_w else ("val" if w in val_w else "test")
        
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "dataset_2_chars.csv"), index=False)
    print(f"Saved {len(df)} character records across {len(writers)} writers to dataset_2_chars.csv")

if __name__ == "__main__":
    build_dataset_1_manifests()
    build_dataset_2_manifests()
    print("Manifest generation complete!")
```

---

### Script 2: PyTorch PyTorch Dataset & Data Loader Template

```python
"""
datasets.py
High-performance PyTorch Datasets for BN-HTR lines and Ekush characters.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class BanglaCharacterDataset(Dataset):
    """PyTorch Dataset for dataset_2 (Ekush 122-class isolated characters)."""
    def __init__(self, manifest_csv, split="train", transform=None, invert=True):
        df = pd.read_csv(manifest_csv)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.invert = invert

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        
        if self.invert:
            # Inverts black background to white paper for standard CNNs
            arr = 255 - np.array(img, dtype=np.uint8)
            img = Image.fromarray(arr)
            
        if self.transform:
            img = self.transform(img)
            
        label = int(row["class_id"])
        return img, label

class BanglaLineHTRDataset(Dataset):
    """PyTorch Dataset for dataset_1 line-level handwritten text recognition."""
    def __init__(self, manifest_csv, char2idx, split="train", target_height=64, max_width=1024):
        df = pd.read_csv(manifest_csv)
        # Filter out rows with missing images or empty text
        self.data = df[(df["split"] == split) & (df["text"].str.strip() != "") & (df["image_path"].str.len() > 0)].reset_index(drop=True)
        self.char2idx = char2idx
        self.target_height = target_height
        self.max_width = max_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        
        # Aspect-ratio preserving resize
        w, h = img.size
        new_w = min(int(w * (self.target_height / h)), self.max_width)
        img = img.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
        
        # Pad width to max_width with white pixels (255)
        padded = Image.new("L", (self.max_width, self.target_height), 255)
        padded.paste(img, (0, 0))
        
        img_tensor = torch.tensor(np.array(padded), dtype=torch.float32).unsqueeze(0) / 255.0
        
        # Tokenize text
        text = str(row["text"])
        target_seq = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]
        
        return img_tensor, torch.tensor(target_seq, dtype=torch.long), len(target_seq), new_w
```

---

## 7. Guidelines for Future AI Agents

When working on this repository, any autonomous agent should adhere to this operational checklist:

- [ ] **Check Manifests First:** Check if `/Users/muntasirabdullah/Desktop/Bangla_4090/manifests/` exists. If not, generate them using `build_manifests.py`. Never run slow directory walks in inner loops.
- [ ] **Enforce Split Rules:** Verify that all data loaders respect Document ID splits for `dataset_1` and Writer ID splits for `dataset_2`. Flag any image-level random splits as data leakage.
- [ ] **Handle File Variations:** Always search for image extensions using case-insensitive checks (`.jpg`, `.JPG`, `.png`, `.PNG`, `.jpeg`).
- [ ] **Always Normalize Bengali Text:** Wrap all transcript strings with `unicodedata.normalize('NFC', text)`.
- [ ] **Distinguish Ground Truth from Predictions:** Treat `BN-HTR_Dataset` as human-verified ground truth; treat `Automatic_Annotation` strictly as machine pseudo-labels.
- [ ] **GPU Execution:** When training on the RTX 4090, set `pin_memory=True`, `num_workers=4`, and mixed precision (`torch.cuda.amp.autocast()`) for maximum throughput.

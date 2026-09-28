# BanglaGHTR — Full Paragraph Bengali Handwritten Text Recognition Web Application

A document-level offline handwritten Bengali text recognition (HTR) research platform that seamlessly bridges:
1. **OpenCV Heuristic Line Segmentation** (`segmenter.py`): Angular Hough deskewing, bilateral photometric filtering, adaptive Gaussian binarization, connected-component line height estimation, and smoothed horizontal projection profiling.
2. **BanglaGHTR Deep Recognition Model** (`webapp/inference.py`): Hybrid CNN-Conformer / Transformer with multi-head CTC Beam Search and autoregressive Attention decoding.

---

## 🚀 Quick Start

From the project root:

```bash
cd /home/suza/HandWritenDetection/BanglaGHTR
.venv/bin/python webapp_full_paragraph/app.py
```

Then open your browser at:
👉 **[http://localhost:7861](http://localhost:7861)**

*(Note: Port 7861 is used by default so it does not conflict with the existing single-line web app on port 7860 or Flask on 5050).*

To change the port or host:
```bash
PORT=8080 HOST=0.0.0.0 .venv/bin/python webapp_full_paragraph/app.py
```

---

## 🌟 Key Features

- **End-to-End Paragraph Processing:** Upload any handwritten document page or multi-line paragraph. The pipeline automatically extracts lines in natural reading order and transcribes them into coherent Bengali text.
- **Interactive Document Overlay:** Document viewer with SVG bounding boxes. Hovering over any bounding box highlights the corresponding line in the transcription list, and hovering over a line in the list highlights its bounding box on the original page.
- **Comparative Multi-Decoder Inspection:** Inspect per-line predictions from **CTC Beam Search**, **CTC Greedy**, and **Attention Autoregressive** decoders side-by-side.
- **Authentic Bengali Typography:** Rendered with native `Hind Siliguri` Google Web Fonts for accurate compound characters (*juktakkhor*), matra headlines, vowel signs (*kar*), and consonant modifiers (*fala*).
- **One-Click Export & Copy:**
  - One-click clipboard copy of the full paragraph or individual lines.
  - Export clean plain text (`.txt`).
  - Export structured research JSON (`.json`) containing bounding box coordinates, latency benchmarks, and individual decoder tokens.
- **Benchmark Sample Gallery:** Built-in gallery of 11 real handwritten benchmark paragraphs from the BN-HTRd dataset for immediate one-click testing.
- **Runtime Checkpoint Switching:** Switch between Stage 2 Supervised Best (`best_model.pt`), latest training epochs, or RL fine-tuned models directly from the UI without restarting the server.

---

## 🛠️ Architecture Overview

```
                                  [Scanned Document / Paragraph Image]
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │        OpenCV Line Segmentation Pipeline         │
                        │  1. Hough Transform Deskewing                    │
                        │  2. Bilateral Filter + Adaptive Binarization     │
                        │  3. Connected Component Height (h_line)          │
                        │  4. Smoothed Projection Profiling & Peak Merging │
                        │  5. Dynamic Midpoint Valley Boundary Splitting   │
                        └─────────────────────────┬────────────────────────┘
                                                  │
                                [Ordered Line Crops (Top -> Bottom)]
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │          BanglaGHTR Deep Recognition Model       │
                        │  1. Aspect-Ratio Preserving Normalization        │
                        │  2. Hybrid CNN-Conformer Feature Extractor       │
                        │  3. CTC Beam Search (W=1..10)                    │
                        │  4. Attention Autoregressive Decoder             │
                        └─────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                       [Interactive Full Paragraph Transcription & Analytics]
```

---

## 📡 REST API Reference

The web application exposes REST endpoints for automated evaluation scripts or headless pipelines:

### 1. `POST /api/predict_paragraph`
End-to-end line segmentation and text transcription.

- **Content-Type:** `multipart/form-data` or `application/json`
- **Parameters:**
  - `image`: Image file upload (multipart), OR
  - `sample`: Filename of built-in sample (e.g. `"100_1.jpg"`)
  - `beam_width`: Integer (1–10, default `5`)
  - `line_pad`: Padding around line crops in px (default `8`)
  - `deskew`: Boolean (`true` / `false`, default `true`)
  - `checkpoint_path`: Optional checkpoint path override
- **Response:**
  ```json
  {
    "ok": true,
    "result": {
      "full_text": "কথা প্রকাশ\nবৈচিত্র্যময় এই পৃথিবীর...",
      "line_count": 8,
      "total_chars": 245,
      "total_words": 42,
      "segmentation": { "line_count": 8, "time_ms": 32.4, "deskew_angle": 0.42 },
      "recognition": { "time_ms": 312.6, "device": "cuda:0", "active_checkpoint": "best_model.pt" },
      "lines": [
        {
          "index": 1,
          "box": { "top": 45, "bottom": 110, "left": 30, "right": 850, "width": 820, "height": 65 },
          "best_text": "কথা প্রকাশ",
          "ctc_beam_text": "কথা প্রকাশ",
          "ctc_greedy_text": "কথা প্রকাশ",
          "attn_text": "কথা প্রকাশ"
        }
      ]
    }
  }
  ```

### 2. `POST /api/segment_only`
Runs only OpenCV line segmentation without neural inference (fast preview).

### 3. `POST /api/switch_checkpoint`
Switches the active model checkpoint in memory:
```json
{ "checkpoint_path": "/home/suza/HandWritenDetection/BanglaGHTR/checkpoints/best_model.pt" }
```

### 4. `GET /api/status`
Returns active model info, device, parameter count, and sample list.

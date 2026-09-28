# BanglaGHTR Web App — Single Line Recognition

An interactive Gradio web application for evaluating the state-of-the-art BanglaGHTR Bengali Handwritten Text Recognition model on single text line images.

## Features
- **Out-of-the-Box Inference:** Pre-bundled with production model weights (`exports/banghtr_x_v2_production/banghtr_x_v2_weights.pt`) and vocabulary — runs immediately without downloading external checkpoints.
- **Dynamic Model Switcher:** Seamlessly switch between Stage 2 Supervised Best, Stage 3 RL fine-tuned models, or production export from the dropdown.
- **Multi-Decoder Comparison:** Real-time side-by-side display of CTC Beam Search (width=5), CTC Greedy, and Attention Autoregressive decoding.
- **Confidence Calibration:** Margin-based logit scoring with repetition detection.
- **Beam Width Control:** Adjust beam width slider (1–10) dynamically.

## 🚀 Quick Start

Run the following commands from the project root:

```bash
# 1. Set up and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch single-line demo
python webapp/app.py
```

Then open your browser at:
👉 **http://localhost:7860**

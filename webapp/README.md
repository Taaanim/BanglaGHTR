# BanglaGHTR Web App

An interactive Gradio web application for evaluating the state-of-the-art BanglaGHTR Bengali Handwritten Text Recognition model.

## Features
- **Dynamic Model Switcher:** Instantly compare Stage 2 Supervised Best (`best_model.pt`) vs Stage 3 RL Final (`banghtr_x_v2_rl_final.pt`) or production export.
- **Multi-Decoder Comparison:** Real-time side-by-side display of CTC Beam Search (width=5), CTC Greedy, and Attention Autoregressive decoding.
- **Beam Width Control:** Adjust beam width slider (1–10) dynamically.
- **Benchmark Presets:** One-click testing on authentic Bengali handwriting line scans.

## How to Run

```bash
cd /home/suza/HandWritenDetection/BanglaGHTR
.venv/bin/python webapp/app.py
```

Then open your browser at **http://localhost:7860** (or forward port 7860 in VS Code Remote SSH).


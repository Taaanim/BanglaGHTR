# BANGHTR-X v2 Web App

A minimal Gradio web app to test the trained Bengali Handwritten Text Recognition model.

## How to Run

```bash
# From the project root
cd /home/suza/HandWritenDetection/BanglaGHTR

# Start the app
.venv/bin/python webapp/app.py
```

Then open your browser at **http://localhost:7860**

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI — upload image, show predictions |
| `inference.py` | Model loading + prediction logic |

## What It Does

1. Upload any handwritten Bengali line image (JPG, PNG)
2. Runs two decoders:
   - **CTC Greedy** — faster, currently more accurate (~32% CER)
   - **Attention Greedy** — slower, needs more training epochs (~71% CER)
3. Shows the best prediction instantly

## Sample Test Images

```
Dataset/Raw_dataset/BN-HTRd .../BN-HTR_Dataset/Segmentation_Images/Lines/
```

Pick any `.jpg` file from there to test.

## Notes

- Model loaded from `exports/banghtr_x_v2_production/banghtr_x_v2_weights.pt`
- Runs on GPU if available, CPU otherwise
- Currently trained for only 10/50 epochs — accuracy will improve significantly after full training

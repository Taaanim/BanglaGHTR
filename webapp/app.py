"""
BANGHTR-X v2 — Minimal Gradio Web App
Run:  cd /home/suza/HandWritenDetection/BanglaGHTR
      .venv/bin/python webapp/app.py
Then open: http://localhost:7860
"""

import gradio as gr
from inference import predict

# ── UI ────────────────────────────────────────────────────────────────────────

def run_inference(image):
    """Gradio callback: receives a PIL Image, returns prediction strings."""
    if image is None:
        return "", "", "⚠️ Please upload an image first."

    result = predict(image)

    ctc  = result["ctc_text"]  or "(empty)"
    attn = result["attn_text"] or "(empty)"
    best = result["best_text"] or "(empty)"

    return ctc, attn, best


with gr.Blocks(title="BANGHTR-X v2 — Bangla HTR Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🇧🇩 BANGHTR-X v2 — Bangla Handwritten Text Recognition
        Upload a **handwritten Bengali line image** and the model will transcribe it.

        > ⚠️ **Note:** Model is currently at ~32% CER (trained 10 / 50 epochs).
        > CTC output is more reliable than Attention at this stage.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 Upload Handwritten Bengali Line Image",
                image_mode="RGB",
            )
            run_btn = gr.Button("🔍 Recognize", variant="primary")

        with gr.Column(scale=1):
            best_out  = gr.Textbox(label="✅ Best Prediction (CTC Greedy)", lines=2, interactive=False)
            ctc_out   = gr.Textbox(label="📊 CTC Greedy Output",           lines=2, interactive=False)
            attn_out  = gr.Textbox(label="🧠 Attention Greedy Output",     lines=2, interactive=False)

    run_btn.click(
        fn=run_inference,
        inputs=[image_input],
        outputs=[ctc_out, attn_out, best_out],
    )

    # Also trigger on image change
    image_input.change(
        fn=run_inference,
        inputs=[image_input],
        outputs=[ctc_out, attn_out, best_out],
    )

    gr.Markdown(
        """
        ---
        ### How to use
        1. Upload a **grayscale or color scan** of a single handwritten Bengali line.
        2. Click **Recognize** (or just upload — it auto-runs).
        3. Compare CTC vs Attention outputs.

        ### Sample images
        Sample validation images are at:
        `Dataset/Raw_dataset/BN-HTRd .../BN-HTR_Dataset/Segmentation_Images/Lines/`
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible from browser on your local network
        server_port=7860,
        share=False,              # set True to get a public gradio.live link
        show_error=True,
    )

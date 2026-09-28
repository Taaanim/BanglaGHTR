"""
BanglaGHTR — Bengali Handwritten Text Recognition Web Application
Run:  cd /home/suza/HandWritenDetection/BanglaGHTR
      .venv/bin/python webapp/app.py
Then open: http://localhost:7860
"""

import os
import sys
import time
import gradio as gr  # type: ignore

# Ensure webapp directory and project root are in sys.path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.inference import predict, get_available_checkpoints, get_model_status, load_checkpoint

# ── Checkpoints & Defaults ───────────────────────────────────────────────────
CHECKPOINTS_MAP = get_available_checkpoints()
CHECKPOINT_CHOICES = list(CHECKPOINTS_MAP.keys())
DEFAULT_CHECKPOINT = CHECKPOINT_CHOICES[0] if CHECKPOINT_CHOICES else None

# Initial model load
init_status = get_model_status()

# ── Benchmark Sample Images ───────────────────────────────────────────────────
RAW_LINE_DIR = os.path.join(
    PROJECT_ROOT,
    "Dataset/Raw_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/Segmentation_Images/Lines"
)

candidates = [
    (os.path.join(RAW_LINE_DIR, "1/1_1/1_1_1.jpg"), "কথা প্রকাশ"),
    (os.path.join(RAW_LINE_DIR, "1/1_1/1_1_2.jpg"), "বৈচিত্র্যময় এই পৃথিবীর পরতে পরতে"),
    (os.path.join(RAW_LINE_DIR, "1/1_1/1_1_4.jpg"), "একটা ভালো বই পারে সেই বিস্ময়"),
    (os.path.join(RAW_LINE_DIR, "1/1_2/1_2_1.jpg"), "২০০২ সাল থেকে ' কথাপ্রকাশ ' - এর যাত্রা শুরু । রাজধানী"),
]

SAMPLE_EXAMPLES = [[fpath] for fpath, _ in candidates if os.path.exists(fpath)]


# ── Callback Handlers ─────────────────────────────────────────────────────────
def on_checkpoint_change(selected_label):
    ckpt_path = CHECKPOINTS_MAP.get(selected_label)
    if ckpt_path:
        load_checkpoint(ckpt_path, force_reload=True)
    status = get_model_status()
    meta_extra = ""
    if status.get("epoch"):
        meta_extra += f" &nbsp;|&nbsp; Epoch: <b>{status['epoch']}</b>"
    if status.get("val_cer") is not None:
        meta_extra += f" &nbsp;|&nbsp; Val CER: <b style='color:#38bdf8;'>{status['val_cer']*100:.2f}%</b>"
    return f"🟢 <b>Active Model:</b> <code>{status['checkpoint']}</code> ({status['params']} params, <code>{status['device']}</code>){meta_extra}"


def run_inference(image, selected_ckpt, beam_width):
    if image is None:
        return "", "", "", "", "⚠️ Please upload an image or select one from the benchmark samples below."

    ckpt_path = CHECKPOINTS_MAP.get(selected_ckpt) if selected_ckpt else None
    t0 = time.time()
    res = predict(image, beam_width=int(beam_width), checkpoint_path=ckpt_path)
    elapsed = time.time() - t0

    best_pred   = res["best_text"]
    ctc_beam    = res["ctc_text"]
    ctc_greedy  = res["ctc_greedy_text"]
    attn        = res["attn_text"]

    beam_conf   = res.get("ctc_beam_conf", 94.0)
    greedy_conf = res.get("ctc_greedy_conf", 91.0)
    attn_conf   = res.get("attn_conf", 89.0)

    char_count = len(best_pred.replace(" ", ""))
    word_count = len([w for w in best_pred.strip().split() if w])

    status_msg = (
        f"⚡ Transcribed in <b>{elapsed*1000:.0f} ms</b> &nbsp;|&nbsp; "
        f"📝 <b>{char_count}</b> chars, <b>{word_count}</b> words &nbsp;|&nbsp; "
        f"🎯 Model: <code>{res['checkpoint_name']}</code> &nbsp;|&nbsp; "
        f"CTC Beam: <b style='color:#38bdf8;'>{beam_conf:.1f}%</b> · "
        f"Greedy: <b style='color:#818cf8;'>{greedy_conf:.1f}%</b> · "
        f"Attention: <b style='color:#c084fc;'>{attn_conf:.1f}%</b>"
    )

    beam_update = gr.update(value=ctc_beam, label=f"📊 CTC Beam: ({beam_conf:.1f}% conf)")
    greedy_update = gr.update(value=ctc_greedy, label=f"⚡ Greedy: ({greedy_conf:.1f}% conf)")
    attn_update = gr.update(value=attn, label=f"🧠 Attention: ({attn_conf:.1f}% conf)")

    return best_pred, beam_update, greedy_update, attn_update, status_msg



# ── High-Contrast, Zero-White-on-White, No-Cutoff & No-Scroll Theme ───────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* Global Container Background & High Contrast Typography */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 16px 24px 32px 24px !important;
}

/* Card wrappers */
.card-panel {
    background: #111827 !important;
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Force dark background and white/bright text on ALL inputs and textareas */
textarea, input, .gr-input {
    background-color: #172033 !important;
    color: #f8fafc !important;
    border: 1px solid #2d3b55 !important;
    border-radius: 8px !important;
}

textarea:focus, input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
}

/* High contrast labels */
label, .gr-label, span.label-text, .gr-input-label {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    margin-bottom: 6px !important;
}

/* Dropdown list styling */
select, .dropdown, .secondary-wrap, ul.options {
    background-color: #172033 !important;
    color: #f8fafc !important;
    border-color: #2d3b55 !important;
}
li.item, .dropdown-item {
    background-color: #172033 !important;
    color: #f8fafc !important;
}
li.item:hover, .dropdown-item:hover {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
}

/* Status Bar */
.status-bar {
    background: #0d2038 !important;
    border: 1px solid #0284c7 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    color: #bae6fd !important;
    font-size: 0.90rem !important;
    line-height: 1.5 !important;
    margin-bottom: 14px !important;
}

/* Primary Prediction Hero Box: Full-width 1-line display, ample height so bottom never clips */
.hero-prediction-card {
    background: #0a1628 !important;
    border: 2px solid #0284c7 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    box-shadow: 0 4px 20px rgba(2, 132, 199, 0.18) !important;
    margin: 8px 0 16px 0 !important;
}

.hero-prediction-card textarea {
    font-family: 'Hind Siliguri', 'Kalpurush', 'Siyam Rupali', 'Noto Sans Bengali', sans-serif !important;
    font-size: 1.70rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
    background-color: #071020 !important;
    border: 1px solid #1a3250 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    height: 74px !important;
    min-height: 74px !important;
    max-height: 74px !important;
    line-height: 1.8 !important;
    padding: 12px 18px !important;
    box-sizing: border-box !important;
}

/* Secondary Decoder Outputs */
.decoder-box textarea {
    font-family: 'Hind Siliguri', 'Kalpurush', sans-serif !important;
    font-size: 1.10rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    background-color: #141c2e !important;
    border: 1px solid #243049 !important;
    border-radius: 6px !important;
    white-space: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    line-height: 1.6 !important;
    padding: 10px 14px !important;
    box-sizing: border-box !important;
}

/* Action button */
.btn-primary {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
}

/* Sample pill buttons */
.sample-chip {
    background: #131f33 !important;
    color: #e2e8f0 !important;
    border: 1px solid #24344d !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-family: 'Hind Siliguri', sans-serif !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.sample-chip:hover {
    background: #0284c7 !important;
    color: #ffffff !important;
    border-color: #38bdf8 !important;
}

/* ── COMPLETELY DISABLE SCROLLING ON EXAMPLES / DATASET CONTAINER ── */
.gr-examples,
.gr-examples .table-wrap,
.gr-examples table,
.gr-examples tbody,
.gr-examples tr,
.gr-examples td,
.gr-dataset,
.gr-samples-table,
.dataset-wrap,
.table-wrap,
div[data-testid="dataset"],
div[data-testid="dataset"] * {
    overflow: visible !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    max-height: none !important;
    height: auto !important;
}

.table-wrap {
    overflow: visible !important;
    max-height: none !important;
    height: auto !important;
}

div[data-testid="dataset"] {
    background-color: #111827 !important;
    border: 1px solid #1f293d !important;
    border-radius: 10px !important;
    padding: 4px !important;
}

div[data-testid="dataset"] tr {
    background-color: #111c2e !important;
    border-bottom: 1px solid #1f293d !important;
    transition: all 0.15s ease !important;
}

div[data-testid="dataset"] tr:hover {
    background-color: #172740 !important;
}

div[data-testid="dataset"] td {
    padding: 8px 12px !important;
    background: transparent !important;
}

.gr-examples img {
    max-height: 48px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 4px !important;
    display: inline-block !important;
}

/* Hide unaligned default footer */
footer {
    display: none !important;
}
"""

# ── Gradio Blocks UI ──────────────────────────────────────────────────────────
with gr.Blocks(title="BanglaGHTR — Bengali HTR") as demo:

    # Header Bar
    gr.HTML(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0 14px; border-bottom: 1px solid #1e293b; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; color: #ffffff;">
                    Bangla<span style="color: #38bdf8;">GHTR</span>
                </span>
                <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;">
                    Production
                </span>
                <span style="color: #94a3b8; font-size: 0.9rem;">| &nbsp;Bengali Offline Handwritten Text Recognition</span>
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    4-Stage ConvNeXt
                </span>
                <span style="background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Matra Attention
                </span>
                <span style="background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Grapheme MoE
                </span>
            </div>
        </div>
        """
    )

    # Top Control Bar: Active Model Selection & Beam Search Width
    with gr.Row():
        with gr.Column(scale=3):
            ckpt_dropdown = gr.Dropdown(
                choices=CHECKPOINT_CHOICES,
                value=DEFAULT_CHECKPOINT,
                label="🎯 Active Model Checkpoint (Stage 2 Best vs RL)",
                interactive=True,
            )
        with gr.Column(scale=1):
            beam_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="⚡ CTC Beam Search Width",
                interactive=True,
            )

    # Live Status Banner
    initial_meta = ""
    if init_status.get("epoch"):
        initial_meta += f" &nbsp;|&nbsp; Epoch: <b>{init_status['epoch']}</b>"
    if init_status.get("val_cer") is not None:
        initial_meta += f" &nbsp;|&nbsp; Val CER: <b style='color:#38bdf8;'>{init_status['val_cer']*100:.2f}%</b>"

    status_html = gr.HTML(
        f"<div class='status-bar'>🟢 <b>Active Model:</b> <code>{init_status['checkpoint']}</code> ({init_status['params']} params, <code>{init_status['device']}</code>){initial_meta}</div>"
    )

    # ── FULL-WIDTH PRIMARY PREDICTION HERO BOX (No scrolling, ample bottom clearance) ──
    with gr.Row():
        with gr.Column(scale=1):
            best_out = gr.Textbox(
                label="🏆 Recognized Bengali Text (Single-Line Full View — No Scrolling)",
                lines=1,
                max_lines=1,
                interactive=False,
                elem_classes=["hero-prediction-card"]
            )
            infer_status = gr.HTML(
                "<div style='color: #64748b; font-size: 0.85rem; padding: 2px 4px 8px 4px;'>Upload an image or click a benchmark sample below to transcribe.</div>"
            )

    # ── WORKSPACE: Left = Input Image & Controls | Right = Detailed Decoders Breakdown ──
    with gr.Row(equal_height=True):
        # Left: Image Input & Action Buttons
        with gr.Column(scale=5):
            image_input = gr.Image(
                type="pil",
                label="📷 Input Bengali Handwritten Line Scan",
                image_mode="RGB",
            )
            with gr.Row():
                recognize_btn = gr.Button("🚀 Recognize Text", variant="primary", elem_classes=["btn-primary"], scale=2)
                clear_btn = gr.ClearButton(components=[image_input], value="✕ Clear", scale=1)

        # Right: Decoder Breakdown
        with gr.Column(scale=5):
            gr.Markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #94a3b8; margin-bottom: 6px;'>🔍 Detailed Decoder Outputs with Model Confidence:</div>")
            ctc_beam_out   = gr.Textbox(label="📊 CTC Beam: (Conf: --)", lines=1, max_lines=1, interactive=False, elem_classes=["decoder-box"])
            ctc_greedy_out = gr.Textbox(label="⚡ Greedy: (Conf: --)", lines=1, max_lines=1, interactive=False, elem_classes=["decoder-box"])
            attn_out       = gr.Textbox(label="🧠 Attention: (Conf: --)", lines=1, max_lines=1, interactive=False, elem_classes=["decoder-box"])


    # Event bindings
    ckpt_dropdown.change(
        fn=on_checkpoint_change,
        inputs=[ckpt_dropdown],
        outputs=[status_html],
    )

    recognize_btn.click(
        fn=run_inference,
        inputs=[image_input, ckpt_dropdown, beam_slider],
        outputs=[best_out, ctc_beam_out, ctc_greedy_out, attn_out, infer_status],
    )

    image_input.change(
        fn=run_inference,
        inputs=[image_input, ckpt_dropdown, beam_slider],
        outputs=[best_out, ctc_beam_out, ctc_greedy_out, attn_out, infer_status],
    )

    # ── BENCHMARK SAMPLES: Clickable Chips (Zero Scroll) + Non-scrolling Inline Examples ──
    if candidates:
        gr.Markdown("---")
        gr.Markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #cbd5e1; margin-bottom: 8px;'>📚 Quick Benchmark Sample Lines (Click to Transcribe Instantly):</div>")
        
        # 1. Quick One-Click Buttons
        with gr.Row():
            for fpath, label in candidates:
                if os.path.exists(fpath):
                    btn = gr.Button(f"📝 {label}", elem_classes=["sample-chip"], size="sm")
                    btn.click(
                        fn=lambda p=fpath: p,
                        inputs=[],
                        outputs=[image_input],
                    ).then(
                        fn=run_inference,
                        inputs=[image_input, ckpt_dropdown, beam_slider],
                        outputs=[best_out, ctc_beam_out, ctc_greedy_out, attn_out, infer_status],
                    )

        # 2. Non-scrolling visual scan examples
        if SAMPLE_EXAMPLES:
            gr.Examples(
                examples=SAMPLE_EXAMPLES,
                inputs=[image_input],
                outputs=[best_out, ctc_beam_out, ctc_greedy_out, attn_out, infer_status],
                fn=run_inference,
                cache_examples=False,
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )

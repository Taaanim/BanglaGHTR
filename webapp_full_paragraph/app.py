"""
BanglaGHTR Full Paragraph Recognition Web Application
Serves the merged OpenCV Line Segmenter + BANGHTR-X Deep Bengali HTR pipeline.

Usage:
    cd /home/suza/HandWritenDetection/BanglaGHTR
    .venv/bin/python webapp_full_paragraph/app.py

Default port: 7861 (to avoid conflict with 7860 Gradio and 5050 Flask)
"""
from __future__ import annotations

import os
import sys
import io
import time
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, render_template

# Ensure directories in sys.path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
WEBAPP_DIR = os.path.join(PROJECT_ROOT, "webapp")

for p in [CUR_DIR, PROJECT_ROOT, WEBAPP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import recognize_paragraph, encode_jpeg, draw_styled_boxes
from segmenter import segment_image, crop_lines
from inference import get_available_checkpoints, get_model_status, load_checkpoint

BASE_DIR = CUR_DIR
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")

app = Flask(
    __name__,
    template_folder=os.path.join(CUR_DIR, "templates"),
    static_folder=None,
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload


def _get_sample_list():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(valid_exts))
    return files


# ---------------------------------------------------------------------------
# Frontend Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    checkpoints = get_available_checkpoints()
    samples = _get_sample_list()
    status = get_model_status()
    return render_template(
        "index.html",
        samples=samples,
        checkpoints=checkpoints,
        model_status=status,
    )


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns current model status, device, checkpoints and sample files."""
    try:
        status = get_model_status()
        ckpts = get_available_checkpoints()
        samples = _get_sample_list()
        return jsonify({
            "ok": True,
            "model_status": status,
            "checkpoints": ckpts,
            "samples": samples,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/switch_checkpoint", methods=["POST"])
def api_switch_checkpoint():
    """Switches the active model checkpoint."""
    data = request.get_json(silent=True) or {}
    ckpt_path = data.get("checkpoint_path")
    if not ckpt_path:
        return jsonify({"ok": False, "error": "No checkpoint_path specified"}), 400
    try:
        load_checkpoint(ckpt_path, force_reload=True)
        status = get_model_status()
        return jsonify({"ok": True, "model_status": status})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predict_paragraph", methods=["POST"])
def api_predict_paragraph():
    """
    Main End-to-End Paragraph Transcription Endpoint:
    Accepts:
      - Multipart 'image' file upload OR
      - JSON { 'sample': 'filename.jpg' }
    Options:
      - 'beam_width': int (1-10, default 5)
      - 'checkpoint_path': optional string
      - 'line_pad': int (0-20, default 8)
      - 'deskew': bool (default True)
    """
    img = None
    beam_width = 5
    checkpoint_path = None
    line_pad = 8
    deskew = True

    # 1. Parse parameters and image
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        sample_name = payload.get("sample")
        beam_width = int(payload.get("beam_width", 5))
        checkpoint_path = payload.get("checkpoint_path") or None
        line_pad = int(payload.get("line_pad", 8))
        deskew = bool(payload.get("deskew", True))

        if not sample_name:
            return jsonify({"ok": False, "error": "No sample specified in JSON"}), 400

        sample_path = os.path.join(SAMPLES_DIR, sample_name)
        if not os.path.isfile(sample_path):
            return jsonify({"ok": False, "error": f"Sample file '{sample_name}' not found"}), 404

        img = cv2.imread(sample_path)
    else:
        # Form / Multipart upload
        beam_width = int(request.form.get("beam_width", 5))
        checkpoint_path = request.form.get("checkpoint_path") or None
        line_pad = int(request.form.get("line_pad", 8))
        deskew = request.form.get("deskew", "true").lower() in ("true", "1", "yes")

        sample_name = request.form.get("sample")
        if sample_name:
            sample_path = os.path.join(SAMPLES_DIR, sample_name)
            if not os.path.isfile(sample_path):
                return jsonify({"ok": False, "error": f"Sample '{sample_name}' not found"}), 404
            img = cv2.imread(sample_path)
        elif "image" in request.files:
            file_storage = request.files["image"]
            raw_bytes = file_storage.read()
            if not raw_bytes:
                return jsonify({"ok": False, "error": "Uploaded image file is empty"}), 400
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"ok": False, "error": "No 'image' file or 'sample' specified"}), 400

    if img is None:
        return jsonify({"ok": False, "error": "Failed to decode image"}), 400

    # 2. Run full pipeline
    try:
        result = recognize_paragraph(
            image_bgr=img,
            beam_width=beam_width,
            checkpoint_path=checkpoint_path,
            line_pad=line_pad,
            deskew=deskew,
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/segment_only", methods=["POST"])
def api_segment_only():
    """Runs only OpenCV line segmentation without neural recognition."""
    img = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        sample_name = payload.get("sample")
        if sample_name:
            path = os.path.join(SAMPLES_DIR, sample_name)
            if os.path.isfile(path):
                img = cv2.imread(path)
    elif "image" in request.files:
        f = request.files["image"]
        raw = f.read()
        if raw:
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    try:
        t0 = time.time()
        seg_res = segment_image(img)
        elapsed = time.time() - t0
        boxes = seg_res.get("boxes", [])

        annotated = draw_styled_boxes(img, boxes)
        annotated_uri = encode_jpeg(annotated)
        bw_uri = encode_jpeg(seg_res.get("bw"))

        crops = crop_lines(img, boxes, pad=8)
        thumbs = []
        for i, c in enumerate(crops, 1):
            h, w = c.shape[:2]
            thumbs.append({
                "index": i,
                "box": {"top": boxes[i-1][0], "bottom": boxes[i-1][1], "left": boxes[i-1][2], "right": boxes[i-1][3]},
                "width": int(w),
                "height": int(h),
                "uri": encode_jpeg(c, quality=85),
            })

        return jsonify({
            "ok": True,
            "result": {
                "line_count": len(boxes),
                "line_height": int(seg_res.get("line_height", 0)),
                "deskew_angle": round(float(seg_res.get("deskew_angle", 0.0)), 3),
                "time_ms": round(elapsed * 1000, 1),
                "annotated_uri": annotated_uri,
                "bw_uri": bw_uri,
                "lines": thumbs,
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket

    def is_port_in_use(port_num: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port_num)) == 0

    # Default to 7860 (matches IDE default port forwarding). Fallback to 7861 if 7860 is busy.
    target_port = int(os.environ.get("PORT", 7860))
    if is_port_in_use(target_port) and "PORT" not in os.environ:
        target_port = 7861

    host = os.environ.get("HOST", "0.0.0.0")
    print("\n" + "="*65)
    print(" 🚀 BANGHTR-X Full Paragraph Recognition Research App")
    print(f" 🌐 Running on: http://{host}:{target_port}  (Local: http://localhost:{target_port})")
    print("="*65 + "\n")
    app.run(host=host, port=target_port, debug=False, threaded=True)


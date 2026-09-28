"""
BanglaGHTR Inference Module
Loads the trained model and tokenizer, runs prediction on a handwritten Bengali line image.
"""
import os
import sys
import json
import unicodedata
import torch
import numpy as np
from PIL import Image
from typing import Optional, Any, Dict, List

# Add project root to path so src/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.banghtr_x import BANGHTR_X_V2
from src.data.transforms import AugmentedAspectRatioPadResize
# ── Post-processing selector (LM-aware meta-ensemble) ─────────────────────────
from src.postprocess import select_best_prediction as _select_best

# ── Paths (relative to project root) ─────────────────────────────────────────
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "exports", "banghtr_x_v2_production", "banghtr_x_v2_weights.pt")
VOCAB_PATH   = os.path.join(PROJECT_ROOT, "exports", "banghtr_x_v2_production", "vocab.json")

# Priority: best_model.pt (Stage 2 Supervised Best) > latest epoch ckpt > RL-final > RL-best > production export
CHECKPOINT_PRIORITY = [
    os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "checkpoint_epoch_21.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "banghtr_x_v2_rl_final.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "banghtr_x_v2_rl_best.pt"),
    WEIGHTS_PATH,
]

# ── Globals (loaded once or reloaded on checkpoint switch) ────────────────────
_model                = None
_char2idx             = None
_idx2char             = None
_device               = None
_transform            = None
_loaded_checkpoint    = None
_checkpoint_metadata  = {}


def get_available_checkpoints() -> dict:
    """
    Scans checkpoints/ and exports/ directories and returns a dictionary:
        {display_name: absolute_file_path}
    Sorted with the most recommended/accurate models on top.
    """
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    options = {}

    # 1. Best Supervised Stage 2 (Top Recommendation)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_path):
        size_mb = os.path.getsize(best_path) / (1024 * 1024)
        options[f"⭐ Stage 2 Best (best_model.pt - {size_mb:.0f}MB) [RECOMMENDED]"] = best_path

    # 2. Latest Epoch Checkpoints (e.g. checkpoint_epoch_21.pt)
    if os.path.isdir(ckpt_dir):
        epoch_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        def _get_epoch_num(fn):
            try:
                return int(fn.replace("checkpoint_epoch_", "").replace(".pt", ""))
            except Exception:
                return -1
        epoch_files.sort(key=_get_epoch_num, reverse=True)
        for ef in epoch_files[:3]:  # Top 3 latest
            p = os.path.join(ckpt_dir, ef)
            size_mb = os.path.getsize(p) / (1024 * 1024)
            options[f"🔄 Latest Epoch ({ef} - {size_mb:.0f}MB)"] = p

    # 3. RL Models
    rl_final = os.path.join(ckpt_dir, "banghtr_x_v2_rl_final.pt")
    if os.path.exists(rl_final):
        size_mb = os.path.getsize(rl_final) / (1024 * 1024)
        options[f"Stage 3 RL Final (banghtr_x_v2_rl_final.pt - {size_mb:.0f}MB)"] = rl_final

    rl_best = os.path.join(ckpt_dir, "banghtr_x_v2_rl_best.pt")
    if os.path.exists(rl_best):
        size_mb = os.path.getsize(rl_best) / (1024 * 1024)
        options[f"Stage 3 RL Best (banghtr_x_v2_rl_best.pt - {size_mb:.0f}MB)"] = rl_best

    # 4. Production Export
    if os.path.exists(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        options[f"Production Export (banghtr_x_v2_weights.pt - {size_mb:.0f}MB)"] = WEIGHTS_PATH

    return options


def load_checkpoint(target_path: Optional[str] = None, force_reload: bool = False):
    """
    Loads or switches the active model checkpoint.
    If target_path is None, chooses the highest-priority available checkpoint.
    """
    global _model, _char2idx, _idx2char, _device, _transform, _loaded_checkpoint, _checkpoint_metadata

    # Resolve target checkpoint path
    if target_path is None:
        for p in CHECKPOINT_PRIORITY:
            if os.path.exists(p):
                target_path = p
                break

    if target_path is None or not os.path.exists(target_path):
        raise FileNotFoundError("No valid model checkpoint found in checkpoints/ or exports/.")

    # Return if already loaded
    if _model is not None and _loaded_checkpoint == target_path and not force_reload:
        return _model

    # Load vocab if needed
    if _char2idx is None or _idx2char is None:
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab = json.load(f)
        _char2idx = vocab["char2idx"]
        _idx2char = {int(k): v for k, v in vocab["idx2char"].items()}

    num_classes = len(_char2idx)

    # Initialize device
    if _device is None:
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model if needed
    if _model is None:
        _model = BANGHTR_X_V2(
            num_classes=num_classes,
            in_channels=1,
            hidden_dim=384,
            encoder_layers=6,
            decoder_layers=4,
            num_heads=8,
            dim_feedforward=1536,
            use_matra_attn=True,
            use_grapheme_moe=True,
            moe_aux_weight=0.01,
            stem_feat_dim=384,
            decoder_type="hybrid",
        ).to(_device)

    # Load weights
    print(f"[BanglaGHTR] Loading weights from: {target_path}...")
    state = torch.load(target_path, map_location="cpu", weights_only=False)
    _checkpoint_metadata = {}

    if isinstance(state, dict) and "model_state_dict" in state:
        _checkpoint_metadata = {
            "epoch": state.get("epoch"),
            "val_cer": state.get("metrics", {}).get("val_cer"),
            "val_wer": state.get("metrics", {}).get("val_wer"),
            "val_loss": state.get("metrics", {}).get("val_loss"),
        }
        state = state["model_state_dict"]

    if isinstance(state, dict):
        state = {
            k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in state.items()
        }

    missing, unexpected = _model.load_state_dict(state, strict=False)

    _loaded_checkpoint = target_path
    _model.eval()

    if _transform is None:
        _transform = AugmentedAspectRatioPadResize(
            target_height=64, max_width=1024, augment=False
        )

    base = os.path.basename(target_path)
    meta_str = ""
    if _checkpoint_metadata.get("epoch"):
        meta_str = f" | Epoch: {_checkpoint_metadata['epoch']}"
    if _checkpoint_metadata.get("val_cer") is not None:
        meta_str += f" | Val CER: {_checkpoint_metadata['val_cer']*100:.2f}%"

    print(f"[BanglaGHTR] Active model: {base}{meta_str} on {_device}")
    return _model


def _load_model():
    """Backward-compatible loader."""
    load_checkpoint()


def get_model_status() -> dict:
    """Returns metadata about the currently active model."""
    load_checkpoint()
    base = os.path.basename(_loaded_checkpoint or "None")
    params = sum(p.numel() for p in _model.parameters())
    return {
        "checkpoint": base,
        "checkpoint_path": _loaded_checkpoint,
        "device": str(_device),
        "params": f"{params / 1e6:.1f}M",
        "epoch": _checkpoint_metadata.get("epoch"),
        "val_cer": _checkpoint_metadata.get("val_cer"),
        "val_wer": _checkpoint_metadata.get("val_wer"),
    }


# ── Decoding helpers ──────────────────────────────────────────────────────────

def _ctc_char_quality(probs_row):
    """
    Per-frame CTC character-quality metrics.
      q : max softmax probability for the emitted token (peak certainty).
      g : (top1 - top2) / top1  in [0, 1] - 1 = decisive, 0 = tied with #2.
      b : softmax probability of the BLANK token (silence/ambiguity at frame).
    """
    top2 = np.partition(probs_row, -2)[-2:]
    top1 = float(top2[1])
    top2v = float(top2[0])
    q = top1
    g = (top1 - top2v) / (top1 + 1e-9)
    b = float(probs_row[0])  # BLANK = index 0
    return q, max(g, 0.0), b


def _ctc_combine_conf(char_qmax, char_gmargin, char_pblank):
    """
    Aggregate per-character qualities into a single 0..1 confidence.
    Three factors (all in [0, 1] before combination):
        mean(q_k)              length-normalized per-token peak certainty
                               (log-mean, then exp). Stable across line lengths.
        geomean(g_k)           geometric mean of top-2 margins (rewards decisive
                               commitments).
        mean(1 - b_k)          frames that emit characters (not blank) carry
                               more information - down-weights noisy lines.
    Output is in [0, 1] x 100.
    """
    if not char_qmax:
        return 0.0
    eps = 1e-6
    n = len(char_qmax)
    log_q_norm = float(np.sum(np.log(np.clip(char_qmax, eps, 1.0)))) / n
    log_g = float(np.mean(np.log(np.clip(char_gmargin, eps, 1.0))))
    conf = float(np.exp(log_q_norm)) * float(np.exp(log_g)) * (1.0 - float(np.mean(char_pblank)))
    return float(np.clip(conf, 0.0, 1.0)) * 100.0


def _ctc_decode_greedy_with_conf(log_probs):
    """
    Greedy CTC: argmax -> collapse repeats -> remove blank(0).

    Confidence is calibrated on per-frame posteriors at the FRAME THAT EMITS
    EACH CHARACTER (first non-blank, non-repeat frame of a run). See
    _ctc_combine_conf. The OLD implementation was the mean of per-frame
    argmax probs, which always saturated at 85-99% regardless of whether
    the text was correct.
    """
    probs = log_probs.squeeze(1).exp().cpu().numpy()  # [T, C]
    T, C = probs.shape
    BLANK = 0
    tokens = probs.argmax(axis=-1)

    emitted_tokens = []
    char_qmax = []
    char_gmargin = []
    char_pblank = []

    prev = None
    for t in range(T):
        tok = int(tokens[t])
        if tok == BLANK:
            prev = None
            continue
        if tok == prev:
            continue
        q, g, b = _ctc_char_quality(probs[t])
        emitted_tokens.append(tok)
        char_qmax.append(q)
        char_gmargin.append(g)
        char_pblank.append(b)
        prev = tok

    conf = _ctc_combine_conf(char_qmax, char_gmargin, char_pblank)
    return emitted_tokens, conf


def _ctc_prefix_beam_search_with_conf(log_probs, beam_width=5):
    """
    Pure-Python CTC prefix beam search with a CALIBRATED confidence.

    The prefix-beam path score (Pb + Pnb) is an un-calibrated cumulative
    product over hundreds of frames and is NOT used for confidence.

    For confidence we recover per-frame posteriors via greedy alignment of
    the same log-probs, then aggregate with the same metric as greedy
    decoding (so beam/greedy/attention are comparable). The beam is only
    used to pick the SEQUENCE of tokens, not to score it.
    """
    probs = log_probs.squeeze(1).exp().cpu().numpy()  # [T, C]
    T, C = probs.shape
    BLANK = 0

    beam = {(): (1.0, 0.0)}

    for t in range(T):
        p = probs[t]
        new_beam = {}

        for prefix, (Pb, Pnb) in beam.items():
            new_Pb = (Pb + Pnb) * p[BLANK]
            _update_beam(new_beam, prefix, new_Pb, 0.0)

            for c in range(1, C):
                pc = p[c]
                if len(prefix) > 0 and prefix[-1] == c:
                    new_Pnb = Pb * pc
                else:
                    new_Pnb = (Pb + Pnb) * pc
                _update_beam(new_beam, prefix + (c,), 0.0, new_Pnb)

        beam = dict(
            sorted(new_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_width]
        )

    best_prefix = max(beam, key=lambda p: beam[p][0] + beam[p][1])
    tokens = list(best_prefix)
    if not tokens:
        return [], 0.0

    # Greedy alignment of probs to recover per-frame emissions.
    aligned = []
    prev = None
    for t in range(T):
        tok = int(probs[t].argmax())
        if tok == BLANK:
            prev = None
            continue
        if tok == prev:
            continue
        aligned.append((t, tok))
        prev = tok

    char_qmax, char_gmargin, char_pblank = [], [], []
    for tok in tokens:
        matched = [t for (t, tk) in aligned if tk == tok]
        if not matched:
            tok_col = probs[:, tok]
            matched = list(np.argsort(tok_col)[-3:][::-1])
        for t in matched:
            q, g, b = _ctc_char_quality(probs[t])
            char_qmax.append(q)
            char_gmargin.append(g)
            char_pblank.append(b)

    conf = _ctc_combine_conf(char_qmax, char_gmargin, char_pblank)
    return tokens, conf


def _ctc_decode_greedy(log_probs: torch.Tensor) -> list[int]:
    """Greedy CTC: argmax → collapse repeats → remove blank(0)."""
    toks, _ = _ctc_decode_greedy_with_conf(log_probs)
    return toks


def _ctc_prefix_beam_search(log_probs: torch.Tensor, beam_width: int = 5) -> list[int]:
    toks, _ = _ctc_prefix_beam_search_with_conf(log_probs, beam_width=beam_width)
    return toks


def _update_beam(beam, prefix, Pb, Pnb):
    if prefix not in beam:
        beam[prefix] = (0.0, 0.0)
    old_Pb, old_Pnb = beam[prefix]
    beam[prefix] = (old_Pb + Pb, old_Pnb + Pnb)


def _decode_tokens(token_ids: list[int]) -> str:
    """Convert token ids → Bengali string, NFC-normalized."""
    special = {0, 1, 2, 3, 4}  # BLANK, PAD, UNK, BOS, EOS
    raw = "".join(_idx2char.get(t, "") for t in token_ids if t not in special)
    return unicodedata.normalize("NFC", raw)


# ── Public API ────────────────────────────────────────────────────────────────

def predict(image: Image.Image, beam_width: int = 5, checkpoint_path: Optional[str] = None) -> dict:
    """
    Run BanglaGHTR inference on an image.

    Args:
        image: PIL Image, numpy array, or file path string.
        beam_width: Beam search width for CTC (1 = greedy, 5 = recommended).
        checkpoint_path: Optional path to a specific model checkpoint to use.

    Returns:
        dict with keys: 'best_text', 'ctc_text', 'ctc_greedy_text', 'attn_text',
                        'ctc_beam_conf', 'ctc_greedy_conf', 'attn_conf', 'checkpoint_name', etc.
    """
    # Load or switch checkpoint if requested
    if checkpoint_path is not None and checkpoint_path != _loaded_checkpoint:
        load_checkpoint(checkpoint_path)
    else:
        load_checkpoint()

    # Accept string path, numpy array, or PIL Image
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preprocess
    gray = image.convert("L")
    tensor, _ = _transform(gray)          # [1, H, W]
    tensor = tensor.unsqueeze(0).to(_device)  # [1, 1, H, W]

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = _model(tensor)
        ctc_lp  = outputs["ctc_log_probs"]   # [T, 1, C]

        # 1. CTC Greedy with confidence
        ctc_greedy_tok, ctc_greedy_conf = _ctc_decode_greedy_with_conf(ctc_lp)
        ctc_greedy_text = _decode_tokens(ctc_greedy_tok)

        # 2. CTC Beam Search with confidence
        if beam_width > 1:
            ctc_beam_tok, ctc_beam_conf = _ctc_prefix_beam_search_with_conf(ctc_lp, beam_width=beam_width)
            ctc_beam_text = _decode_tokens(ctc_beam_tok)
        else:
            ctc_beam_tok = ctc_greedy_tok
            ctc_beam_text = ctc_greedy_text
            ctc_beam_conf = ctc_greedy_conf

        # 3. Attention Greedy Autoregressive with confidence.
        # The original softmax-of-argmax confidence was unreliable because
        # it (a) ignored the margin between argmax and runner-up, and
        # (b) could not detect runaway repetition ("কককক"). We use a
        # margin-based, repetition-penalized scorer via the new
        # `greedy_decode_with_margin` method on the attention decoder.
        try:
            with torch.no_grad():
                encoder_out = _model.encode(tensor)
                attn_results, attn_scores = _model.attn_decoder.greedy_decode_with_margin(
                    encoder_out,
                    max_len=85,
                    repetition_penalty=0.40,
                )
            attn_tok = attn_results[0]
            attn_conf = float(attn_scores[0]) * 100.0 if attn_scores else 0.0
            attn_text = _decode_tokens(attn_tok)
        except Exception:
            try:
                attn_tok = _model.decode_attention(tensor, method="greedy", max_len=85)[0]
                attn_text = _decode_tokens(attn_tok)
                attn_conf = max(60.0, ctc_greedy_conf * 0.95)
            except Exception as e2:
                attn_text = f"(Attention decode error: {e2})"
                attn_conf = 0.0

    # 4. Best-text selection via LM-aware meta-ensemble ----------------------
    # Delegates to src/postprocess/selector.py which implements:
    #   Combined_Score = w_model * model_conf
    #                  + w_lm    * lm_score      (Bangla bigram LM / KenLM)
    #                  + w_agree * agreement_bonus
    #                  - repetition / length / hallucination penalties
    #
    # The selector is initialised once with this model's vocab so the LM
    # is aware of the in-distribution character set.
    selection = _select_best(
        candidates={
            "ctc_beam":   {"text": ctc_beam_text,   "confidence": ctc_beam_conf},
            "ctc_greedy": {"text": ctc_greedy_text, "confidence": ctc_greedy_conf},
            "attn":       {"text": attn_text,        "confidence": attn_conf},
        },
        vocab_json_path=VOCAB_PATH,
    )

    best_text   = selection["best_text"]
    best_name   = selection["best_source"]
    best_score  = selection["best_score"]
    combined    = selection["combined_scores"]

    ckpt_base = os.path.basename(_loaded_checkpoint or "unknown")
    meta_extra = ""
    if _checkpoint_metadata.get("val_cer") is not None:
        meta_extra = f" (Val CER: {_checkpoint_metadata['val_cer']*100:.2f}%)"

    return {
        "best_text":        best_text or "(empty)",
        "best_source":      best_name,
        "best_score":       best_score,
        "selection_reason": selection["selection_reason"],
        "ctc_text":         ctc_beam_text   or "(empty)",
        "ctc_greedy_text":  ctc_greedy_text or "(empty)",
        "attn_text":        attn_text       or "(empty)",
        # Combined scores shown in UI so the displayed number matches the
        # actual decision criterion (model_conf + LM + agreement).
        "ctc_beam_conf":    round(combined.get("ctc_beam",   ctc_beam_conf),   1),
        "ctc_greedy_conf":  round(combined.get("ctc_greedy", ctc_greedy_conf), 1),
        "attn_conf":        round(combined.get("attn",       attn_conf),       1),
        "checkpoint_name":  f"{ckpt_base}{meta_extra}",
        "device":           str(_device),
        "beam_width":       beam_width,
        # Extra diagnostics from the selector
        "lm_scores":        selection.get("lm_scores", {}),
        "is_low_conf":      selection.get("is_low_conf", False),
        "is_degenerate":    selection.get("is_degenerate", {}),
    }



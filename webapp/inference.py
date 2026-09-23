"""
BANGHTR-X v2 Inference Module
Loads the trained model and tokenizer, runs prediction on a handwritten Bengali line image.
"""

import os
import sys
import json
import unicodedata
import torch
import numpy as np
from PIL import Image

# Add project root to path so src/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.banghtr_x import BANGHTR_X_V2
from src.data.transforms import AugmentedAspectRatioPadResize

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
        options[f"🎮 Stage 3 RL Final (banghtr_x_v2_rl_final.pt - {size_mb:.0f}MB)"] = rl_final

    rl_best = os.path.join(ckpt_dir, "banghtr_x_v2_rl_best.pt")
    if os.path.exists(rl_best):
        size_mb = os.path.getsize(rl_best) / (1024 * 1024)
        options[f"🎮 Stage 3 RL Best (banghtr_x_v2_rl_best.pt - {size_mb:.0f}MB)"] = rl_best

    # 4. Production Export
    if os.path.exists(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        options[f"📦 Production Export (banghtr_x_v2_weights.pt - {size_mb:.0f}MB)"] = WEIGHTS_PATH

    return options


def load_checkpoint(target_path: str = None, force_reload: bool = False):
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
    print(f"[BANGHTR-X] Loading weights from: {target_path}...")
    state = torch.load(target_path, map_location="cpu")
    _checkpoint_metadata = {}

    if isinstance(state, dict) and "model_state_dict" in state:
        _checkpoint_metadata = {
            "epoch": state.get("epoch"),
            "val_cer": state.get("metrics", {}).get("val_cer"),
            "val_wer": state.get("metrics", {}).get("val_wer"),
            "val_loss": state.get("metrics", {}).get("val_loss"),
        }
        state = state["model_state_dict"]

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

    print(f"[BANGHTR-X] Active model: {base}{meta_str} on {_device}")
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

def _ctc_decode_greedy(log_probs: torch.Tensor) -> list[int]:
    """Greedy CTC: argmax → collapse repeats → remove blank(0)."""
    tokens = log_probs.argmax(dim=-1).squeeze(1).tolist()  # [T]
    result, prev = [], None
    for t in tokens:
        if t != prev:
            if t != 0:  # 0 = BLANK
                result.append(t)
        prev = t
    return result


def _ctc_prefix_beam_search(log_probs: torch.Tensor, beam_width: int = 5) -> list[int]:
    """
    Pure-Python CTC prefix beam search (no ctcdecode dependency).
    log_probs: [T, 1, C] — log probabilities from model.
    Returns: best decoded token id list.
    """
    probs = log_probs.squeeze(1).exp().cpu().numpy()  # [T, C]
    T, C = probs.shape
    BLANK = 0

    # beam: dict of prefix_tuple -> (prob_blank, prob_non_blank)
    beam = {(): (1.0, 0.0)}

    for t in range(T):
        p = probs[t]          # [C]
        new_beam: dict = {}

        for prefix, (Pb, Pnb) in beam.items():
            # Extend with blank
            new_Pb = (Pb + Pnb) * p[BLANK]
            _update_beam(new_beam, prefix, new_Pb, 0.0)

            # Extend with each character
            for c in range(1, C):
                pc = p[c]
                if len(prefix) > 0 and prefix[-1] == c:
                    # Repeated char: only prob_blank can extend without collapsing
                    new_Pnb = Pb * pc
                else:
                    new_Pnb = (Pb + Pnb) * pc
                _update_beam(new_beam, prefix + (c,), 0.0, new_Pnb)

        # Prune to top-k
        beam = dict(
            sorted(new_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_width]
        )

    best_prefix = max(beam, key=lambda p: beam[p][0] + beam[p][1])
    return list(best_prefix)


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

def predict(image: Image.Image, beam_width: int = 5, checkpoint_path: str = None) -> dict:
    """
    Run BANGHTR-X inference on an image.

    Args:
        image: PIL Image, numpy array, or file path string.
        beam_width: Beam search width for CTC (1 = greedy, 5 = recommended).
        checkpoint_path: Optional path to a specific model checkpoint to use.

    Returns:
        dict with keys: 'best_text', 'ctc_text', 'ctc_greedy_text', 'attn_text', 'checkpoint_name', 'model_status'
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

        # 1. CTC Greedy
        ctc_greedy_tok = _ctc_decode_greedy(ctc_lp)
        ctc_greedy_text = _decode_tokens(ctc_greedy_tok)

        # 2. CTC Beam Search
        if beam_width > 1:
            ctc_beam_tok = _ctc_prefix_beam_search(ctc_lp, beam_width=beam_width)
            ctc_beam_text = _decode_tokens(ctc_beam_tok)
        else:
            ctc_beam_text = ctc_greedy_text

        # 3. Attention Greedy Autoregressive
        try:
            attn_tok = _model.decode_attention(tensor, method="greedy", max_len=85)[0]
            attn_text = _decode_tokens(attn_tok)
        except Exception as e:
            attn_text = f"(Attention decode error: {e})"

    ckpt_base = os.path.basename(_loaded_checkpoint or "unknown")
    meta_extra = ""
    if _checkpoint_metadata.get("val_cer") is not None:
        meta_extra = f" (Val CER: {_checkpoint_metadata['val_cer']*100:.2f}%)"

    return {
        "best_text": ctc_beam_text or ctc_greedy_text or "(empty)",
        "ctc_text": ctc_beam_text or "(empty)",
        "ctc_greedy_text": ctc_greedy_text or "(empty)",
        "attn_text": attn_text or "(empty)",
        "checkpoint_name": f"{ckpt_base}{meta_extra}",
        "device": str(_device),
        "beam_width": beam_width,
    }


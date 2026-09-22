"""
BANGHTR-X v2 Inference Module
Loads the trained model and tokenizer, runs prediction on a handwritten Bengali line image.
"""

import os
import sys
import json
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

# ── Globals (loaded once) ─────────────────────────────────────────────────────
_model     = None
_char2idx  = None
_idx2char  = None
_device    = None
_transform = None


def _load_model():
    global _model, _char2idx, _idx2char, _device, _transform

    if _model is not None:
        return  # already loaded

    # Load vocabulary
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    _char2idx = vocab["char2idx"]
    _idx2char = {int(k): v for k, v in vocab["idx2char"].items()}

    num_classes = len(_char2idx)

    # Device
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build model (same arch as training)
    _model = BANGHTR_X_V2(
        num_classes=num_classes,
        in_channels=1,
        hidden_dim=256,
        encoder_layers=4,
        decoder_layers=4,
        num_heads=8,
        use_matra_attn=True,
        use_grapheme_moe=True,
        decoder_type="hybrid",
    ).to(_device)

    # Load weights
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    # Handle both raw state_dict and checkpoint dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    _model.load_state_dict(state)
    _model.eval()

    # Image transform (no augmentation for inference)
    _transform = AugmentedAspectRatioPadResize(
        target_height=64, max_width=1024, augment=False
    )

    print(f"[BANGHTR-X] Model loaded on {_device} | vocab size: {num_classes}")


def _ctc_decode(log_probs: torch.Tensor) -> list[int]:
    """Simple greedy CTC decode: argmax + collapse repeated + remove blank(0)."""
    tokens = log_probs.argmax(dim=-1).squeeze(1).tolist()  # [T]
    result, prev = [], None
    for t in tokens:
        if t != prev:
            if t != 0:  # 0 = BLANK
                result.append(t)
        prev = t
    return result


def _decode_tokens(token_ids: list[int]) -> str:
    """Convert token ids → Bengali string using vocab."""
    special = {0, 1, 2, 3, 4}  # BLANK, PAD, UNK, BOS, EOS
    return "".join(_idx2char.get(t, "") for t in token_ids if t not in special)


def predict(image: Image.Image) -> dict:
    """
    Run BANGHTR-X inference on a PIL image.

    Args:
        image: PIL Image (any mode, will be converted to grayscale)

    Returns:
        dict with keys: 'ctc_text', 'attn_text', 'best_text'
    """
    _load_model()

    # Preprocess
    gray = image.convert("L")
    tensor, _ = _transform(gray)          # [1, H, W]
    tensor = tensor.unsqueeze(0).to(_device)  # [1, 1, H, W]

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        # CTC greedy (faster, currently better quality)
        outputs = _model(tensor)
        ctc_lp  = outputs["ctc_log_probs"]           # [T, 1, C]
        ctc_tok = _ctc_decode(ctc_lp)
        ctc_text = _decode_tokens(ctc_tok)

        # Attention greedy (slower, decoder still training)
        attn_tok = _model.decode_attention(tensor, method="greedy", max_len=85)[0]
        attn_text = _decode_tokens(attn_tok)

    return {
        "ctc_text":  ctc_text,
        "attn_text": attn_text,
        # Use CTC as the primary output (better quality at current training stage)
        "best_text": ctc_text,
    }

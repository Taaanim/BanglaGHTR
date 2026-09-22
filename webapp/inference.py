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

# Priority: try RL-final > RL-best > Stage2-best > production export
CHECKPOINT_PRIORITY = [
    os.path.join(PROJECT_ROOT, "checkpoints", "banghtr_x_v2_rl_final.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "banghtr_x_v2_rl_best.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
    WEIGHTS_PATH,
]

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

    # Load best available weights (RL-best > Stage2-best > production export)
    weights_loaded = None
    for ckpt_path in CHECKPOINT_PRIORITY:
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            # Handle both raw state_dict and checkpoint dict
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                _model.load_state_dict(state, strict=True)
                weights_loaded = ckpt_path
                break
            except Exception as e:
                print(f"[BANGHTR-X] Warning: Could not load {ckpt_path}: {e}")
                continue

    if weights_loaded is None:
        raise FileNotFoundError(
            "No valid model checkpoint found. Train the model first by running the notebook."
        )

    _model.eval()

    # Image transform (no augmentation for inference)
    _transform = AugmentedAspectRatioPadResize(
        target_height=64, max_width=1024, augment=False
    )

    src = os.path.basename(weights_loaded)
    print(f"[BANGHTR-X] Model loaded on {_device} | vocab size: {num_classes} | weights: {src}")


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

def predict(image: Image.Image, beam_width: int = 5) -> dict:
    """
    Run BANGHTR-X inference on a PIL image.

    Args:
        image: PIL Image (any mode, will be converted to grayscale)
        beam_width: Beam search width for CTC (1 = greedy, 5 = recommended)

    Returns:
        dict with keys: 'ctc_text', 'attn_text', 'best_text'
    """
    _load_model()

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

        # CTC Beam Search (better than greedy)
        if beam_width > 1:
            ctc_tok = _ctc_prefix_beam_search(ctc_lp, beam_width=beam_width)
        else:
            ctc_tok = _ctc_decode_greedy(ctc_lp)
        ctc_text = _decode_tokens(ctc_tok)

        # Attention greedy (autoregressive — works better with more training)
        try:
            attn_tok = _model.decode_attention(tensor, method="greedy", max_len=85)[0]
            attn_text = _decode_tokens(attn_tok)
        except Exception as e:
            attn_text = f"(Attention decode error: {e})"

    return {
        "ctc_text":  ctc_text,
        "attn_text": attn_text,
        # Use CTC Beam Search as the primary output
        "best_text": ctc_text,
    }

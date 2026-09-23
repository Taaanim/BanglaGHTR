"""
BANGHTR-X v2: Bangla Hierarchical Adaptive Neural Grapheme Transformer.
Upgraded architecture with:
- Transformer Encoder (replacing BiLSTM)
- Dual CTC + Autoregressive Attention Decoder
- Self-Critical Sequence Training (SCST) support

Backward-compatible: original BANGHTR_X class is preserved.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

from .vision.backbones import ConvNeXtStem
from .grapheme.matra_attention import MatraAttentionModule
from .grapheme.diacritic_expert import GraphemeMoEFusion
from .decoder.ctc_decoder import CTCDecoder
from .decoder.attention_decoder import AttentionDecoder
from .encoder.transformer_encoder import BanglaTransformerEncoder


class BANGHTR_X(nn.Module):
    """
    Original BANGHTR-X Architecture (v1) — preserved for backward compatibility.
    Uses BiLSTM + CTC decoder only.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_dim: int = 256,
        use_matra_attn: bool = True,
        use_grapheme_moe: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_matra_attn = use_matra_attn
        self.use_grapheme_moe = use_grapheme_moe

        # 1. Visual Feature Extractor (ConvNeXt Stem)
        self.visual_stem = ConvNeXtStem(in_channels=in_channels, hidden_dim=hidden_dim)

        # 2. Matra-Aware Attention
        if self.use_matra_attn:
            self.matra_attention = MatraAttentionModule(hidden_dim=hidden_dim)

        # 3. Grapheme Mixture of Experts (Diacritic & Conjunct specialists)
        if self.use_grapheme_moe:
            self.grapheme_moe = GraphemeMoEFusion(hidden_dim=hidden_dim)

        # 4. Bidirectional Sequence Refinement (BiLSTM)
        self.seq_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # 5. CTC Sequence Decoder
        self.decoder = CTCDecoder(hidden_dim=hidden_dim, num_classes=num_classes, blank_idx=0)

    def forward(self, images: torch.Tensor, mode: str = "accurate") -> torch.Tensor:
        """
        Args:
            images: [B, 1, H, W] line/word image tensor.
            mode: "fast", "accurate", or "research".
        Returns:
            log_probs: [T, B, num_classes] for PyTorch CTCLoss.
        """
        # Step 1: Visual feature extraction -> [B, T, hidden_dim]
        feat = self.visual_stem(images)

        if mode != "fast":
            # Step 2: Matra attention
            if self.use_matra_attn:
                feat = self.matra_attention(feat)

            # Step 3: Grapheme MoE fusion
            if self.use_grapheme_moe:
                feat = self.grapheme_moe(feat)

        # Step 4: Sequence context refinement -> [B, T, hidden_dim]
        seq_out, _ = self.seq_encoder(feat)

        # Step 5: Log-probs for CTC loss -> [T, B, num_classes]
        log_probs = self.decoder(seq_out)
        return log_probs

    def decode(self, log_probs: torch.Tensor) -> List[List[int]]:
        """Greedy CTC sequence decoding."""
        return self.decoder.decode_greedy(log_probs)


class BANGHTR_X_V2(nn.Module):
    """
    BANGHTR-X v2: Research-Grade Bengali HTR Architecture.

    Pipeline:
        Image → ConvNeXt Stem → Matra Attention → MoE Fusion
              → Transformer Encoder → [CTC Head + Attention Decoder]

    Training modes:
        - "ctc": CTC loss only (fast convergence baseline)
        - "attention": Autoregressive cross-entropy only
        - "hybrid": λ_ctc * L_ctc + λ_attn * L_attn (recommended)
        - "rl": SCST reinforcement learning (post-convergence)

    Inference:
        - Greedy CTC decode (fast)
        - Greedy attention decode (good)
        - Beam search attention decode (best quality)
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_dim: int = 384,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1536,
        use_matra_attn: bool = True,
        use_grapheme_moe: bool = True,
        dropout: float = 0.1,
        label_smoothing: float = 0.05,
        decoder_type: str = "hybrid",  # "ctc", "attention", "hybrid"
        max_seq_len: int = 256,
        pad_idx: int = 1,
        bos_idx: int = 3,
        eos_idx: int = 4,
        moe_aux_weight: float = 0.01,  # weight for MoE load-balancing loss
        stem_feat_dim: int = 384,       # ConvNeXt stem's max-pool channel count
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type
        self.use_matra_attn = use_matra_attn
        self.use_grapheme_moe = use_grapheme_moe
        self.moe_aux_weight = moe_aux_weight
        self.stem_feat_dim = stem_feat_dim

        # ─── Vision Stem ───
        self.visual_stem = ConvNeXtStem(in_channels=in_channels, hidden_dim=hidden_dim)

        # ─── Bengali-Specific Domain Modules ───
        if use_matra_attn:
            self.matra_attention = MatraAttentionModule(
                hidden_dim=hidden_dim,
                num_heads=min(num_heads, 8),
                stem_feat_dim=stem_feat_dim
            )

        if use_grapheme_moe:
            self.grapheme_moe = GraphemeMoEFusion(hidden_dim=hidden_dim)

        # ─── Transformer Encoder (replaces BiLSTM) ───
        self.transformer_encoder = BanglaTransformerEncoder(
            d_model=hidden_dim,
            nhead=num_heads,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=512
        )

        # ─── CTC Decoder Head ───
        if decoder_type in ("ctc", "hybrid"):
            self.ctc_decoder = CTCDecoder(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                blank_idx=0
            )

        # ─── Autoregressive Attention Decoder ───
        if decoder_type in ("attention", "hybrid"):
            self.attn_decoder = AttentionDecoder(
                num_classes=num_classes,
                d_model=hidden_dim,
                nhead=num_heads,
                num_layers=decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_seq_len,
                label_smoothing=label_smoothing,
                pad_idx=pad_idx,
                bos_idx=bos_idx,
                eos_idx=eos_idx
            )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Runs visual stem + domain modules + Transformer encoder.

        Args:
            images: [B, 1, H, W]
        Returns:
            encoder_out: [B, T, hidden_dim]
        """
        # Visual features — also populates stem._max_feat for matra attention
        feat = self.visual_stem(images)  # [B, T, D]

        # Domain-specific modules
        if self.use_matra_attn:
            # Inject structural max-pool feature for structurally-grounded gating
            max_feat = self.visual_stem.get_max_feat()  # [B, T, stem_feat_dim]
            feat = self.matra_attention(feat, max_feat=max_feat)

        if self.use_grapheme_moe:
            feat = self.grapheme_moe(feat)

        # Transformer encoder for global context
        encoder_out = self.transformer_encoder(feat)  # [B, T, D]
        return encoder_out

    def forward(
        self,
        images: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass for training and inference.

        Args:
            images: [B, 1, H, W]
            target_tokens: [B, T_dec] target token indices (with BOS/EOS)
                          Required for attention decoder training.
            mode: "train" or "infer"

        Returns:
            Dict containing:
                - "ctc_log_probs": [T, B, C] (if ctc/hybrid)
                - "attn_logits": [B, T_dec, C] (if attention/hybrid, training only)
                - "attn_loss": scalar (if attention/hybrid, training only)
        """
        encoder_out = self.encode(images)
        results = {"encoder_out": encoder_out}

        # CTC head
        if self.decoder_type in ("ctc", "hybrid"):
            ctc_log_probs = self.ctc_decoder(encoder_out)  # [T, B, C]
            results["ctc_log_probs"] = ctc_log_probs

        # Attention decoder (training with teacher forcing)
        if self.decoder_type in ("attention", "hybrid") and target_tokens is not None:
            attn_logits, attn_loss = self.attn_decoder(encoder_out, target_tokens)
            results["attn_logits"] = attn_logits
            results["attn_loss"] = attn_loss

        # MoE load-balancing auxiliary loss (always computed during forward)
        if self.use_grapheme_moe and mode == "train":
            results["moe_aux_loss"] = self.grapheme_moe.aux_loss

        return results

    def decode_ctc(self, log_probs: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decoding."""
        return self.ctc_decoder.decode_greedy(log_probs)

    def decode_attention(
        self,
        images: torch.Tensor,
        method: str = "greedy",
        beam_width: int = 5,
        max_len: int = 150
    ) -> List[List[int]]:
        """
        Attention-based decoding.

        Args:
            images: [B, 1, H, W]
            method: "greedy" or "beam"
            beam_width: beam size (only for beam search)
        Returns:
            List of token index lists
        """
        encoder_out = self.encode(images)

        if method == "beam":
            return self.attn_decoder.beam_search(
                encoder_out, beam_width=beam_width, max_len=max_len
            )
        else:
            return self.attn_decoder.greedy_decode(
                encoder_out, max_len=max_len
            )

    def load_pretrained_stem(self, checkpoint_path: str, strict: bool = False):
        """
        Loads pretrained ConvNeXt stem weights from Stage 1 character pretraining.
        Transfers the visual backbone while allowing new architecture components.
        """
        from .vision.backbones import CharacterClassifierBackbone
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Load only the vision feature weights (not classifier head)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Map CharacterClassifierBackbone.features → ConvNeXtStem
        stem_keys = {k: v for k, v in state_dict.items() if k.startswith("features.")}
        if stem_keys:
            # The CharacterClassifier features map to ConvNeXtStem stages
            # We do a best-effort partial load
            current_stem = self.visual_stem.state_dict()
            loaded = 0
            for key, val in stem_keys.items():
                # Strip 'features.' prefix and try to match
                short_key = key.replace("features.", "")
                for stem_key in current_stem:
                    if short_key in stem_key and current_stem[stem_key].shape == val.shape:
                        current_stem[stem_key] = val
                        loaded += 1
            if loaded > 0:
                self.visual_stem.load_state_dict(current_stem, strict=False)
                print(f"[Pretrain] Loaded {loaded} parameters from character pretrain checkpoint")
            else:
                print("[Pretrain] Warning: No matching parameters found for stem transfer")
        else:
            # Try direct load
            self.visual_stem.load_state_dict(state_dict, strict=strict)

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts per component."""
        counts = {}
        counts["visual_stem"] = sum(p.numel() for p in self.visual_stem.parameters())
        if self.use_matra_attn:
            counts["matra_attention"] = sum(p.numel() for p in self.matra_attention.parameters())
        if self.use_grapheme_moe:
            counts["grapheme_moe"] = sum(p.numel() for p in self.grapheme_moe.parameters())
        counts["transformer_encoder"] = sum(p.numel() for p in self.transformer_encoder.parameters())
        if hasattr(self, "ctc_decoder"):
            counts["ctc_decoder"] = sum(p.numel() for p in self.ctc_decoder.parameters())
        if hasattr(self, "attn_decoder"):
            attn_count = sum(p.numel() for p in self.attn_decoder.parameters())
            counts["attn_decoder"] = attn_count
            counts["attention_decoder"] = attn_count
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts

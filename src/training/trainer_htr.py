"""
Training harness for BANGHTR-X v2: Hybrid CTC + Attention HTR.
Supports dual loss optimization, LR scheduling, early stopping,
checkpoint management, and mixed precision training.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from ..models.banghtr_x import BANGHTR_X, BANGHTR_X_V2
from ..data.normalizer import BengaliTokenizer
from ..evaluation.metrics import compute_cer, compute_wer, compute_bg_cer
from ..utils.logger import setup_logger


class HTRTrainer:
    """Original CTC-only trainer (preserved for backward compatibility)."""
    def __init__(self, config: Dict[str, Any], tokenizer: BengaliTokenizer, device: torch.device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.logger = setup_logger("HTRTrainer")

        num_classes = len(tokenizer)
        hidden_dim = config["model"].get("hidden_dim", 256)

        # Instantiate BANGHTR-X
        self.model = BANGHTR_X(
            num_classes=num_classes,
            in_channels=1,
            hidden_dim=hidden_dim,
            use_matra_attn=config["model"].get("matra_attention", True),
            use_grapheme_moe=config["model"].get("diacritic_expert", True)
        ).to(self.device)

        # PyTorch CTC Loss (blank index = 0, zero_infinity = True)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        lr = config["training"].get("lr", 0.0003)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(dataloader, desc="Train HTR Epoch", leave=False):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Forward pass: log_probs has shape [T, B, num_classes]
                log_probs = self.model(images, mode="accurate")
                T_seq, B_batch = log_probs.shape[0], log_probs.shape[1]
                input_lengths = torch.full((B_batch,), T_seq, dtype=torch.long, device=self.device)

                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * B_batch
            total_samples += B_batch

        return total_loss / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_refs = []

        for batch in tqdm(dataloader, desc="Val HTR Epoch", leave=False):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            refs = batch["texts"]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                log_probs = self.model(images, mode="accurate")
                T_seq, B_batch = log_probs.shape[0], log_probs.shape[1]
                input_lengths = torch.full((B_batch,), T_seq, dtype=torch.long, device=self.device)
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            total_loss += loss.item() * B_batch
            total_samples += B_batch

            # Decode sequences
            decoded_tokens_batch = self.model.decode(log_probs)
            for pred_tokens, ref_text in zip(decoded_tokens_batch, refs):
                pred_text = self.tokenizer.decode(pred_tokens)
                all_preds.append(pred_text)
                all_refs.append(ref_text)

        cer = compute_cer(all_preds, all_refs)
        wer = compute_wer(all_preds, all_refs)

        return {
            "val_loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "cer": cer,
            "wer": wer
        }


class HTRTrainerV2:
    """
    BANGHTR-X v2 Trainer with hybrid CTC + Attention loss,
    LR scheduling, early stopping, and checkpoint management.
    """
    def __init__(
        self,
        model: BANGHTR_X_V2,
        tokenizer: BengaliTokenizer,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.logger = setup_logger("HTRTrainerV2")

        os.makedirs(checkpoint_dir, exist_ok=True)

        training_cfg = config.get("training", {})
        self.decoder_type = config.get("model", {}).get("decoder_type", "hybrid")
        self.ctc_weight = training_cfg.get("ctc_weight", 0.3)
        self.attn_weight = training_cfg.get("attn_weight", 0.7)
        self.grad_clip = training_cfg.get("grad_clip", 5.0)
        self.epochs = training_cfg.get("epochs", 50)

        # CTC Loss
        self.ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # Optimizer
        lr = training_cfg.get("lr", 3e-4)
        weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # Scheduler (initialized per training run)
        self.scheduler = None

        # Early stopping
        self.best_cer = float("inf")
        self.patience = training_cfg.get("patience", 10)
        self.patience_counter = 0

        # Metric history
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_cer": [], "val_wer": [], "val_bg_cer": [],
            "lr": []
        }

    def init_scheduler(self, steps_per_epoch: int):
        """Initializes LR scheduler."""
        sched_type = self.config.get("training", {}).get("scheduler", "onecycle")
        if sched_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config["training"].get("lr", 3e-4),
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                pct_start=0.1,
                anneal_strategy="cos"
            )
        elif sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs * steps_per_epoch,
                eta_min=1e-7
            )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs one training epoch with hybrid loss."""
        self.model.train()
        total_ctc_loss = 0.0
        total_attn_loss = 0.0
        total_combined_loss = 0.0
        total_samples = 0

        for batch in tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            B = images.size(0)

            # Prepare attention decoder targets (BOS + tokens + EOS)
            attn_targets = None
            if self.decoder_type in ("attention", "hybrid"):
                attn_targets = self._prepare_attn_targets(batch).to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(images, target_tokens=attn_targets, mode="train")

                loss = torch.tensor(0.0, device=self.device)

                # CTC loss — use actual valid frame lengths to fix padding bug
                if "ctc_log_probs" in outputs:
                    log_probs = outputs["ctc_log_probs"]
                    T_seq = log_probs.shape[0]
                    if "valid_widths" in batch:
                        # valid_widths / 4 (stem total stride-w=4) = valid T frames
                        input_lengths = (
                            batch["valid_widths"].float() / 4
                        ).long().clamp(1, T_seq).to(self.device)
                    else:
                        input_lengths = torch.full((B,), T_seq, dtype=torch.long, device=self.device)
                    ctc_loss = self.ctc_criterion(log_probs, targets, input_lengths, target_lengths)
                    loss = loss + self.ctc_weight * ctc_loss
                    total_ctc_loss += ctc_loss.item() * B

                # Attention loss
                if "attn_loss" in outputs:
                    attn_loss = outputs["attn_loss"]
                    loss = loss + self.attn_weight * attn_loss
                    total_attn_loss += attn_loss.item() * B

                # MoE load-balancing auxiliary loss (Switch-Transformer style)
                if "moe_aux_loss" in outputs:
                    moe_aux_weight = getattr(self.model, "moe_aux_weight", 0.01)
                    loss = loss + moe_aux_weight * outputs["moe_aux_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()

            if self.scheduler is not None and scale_before <= scale_after:
                self.scheduler.step()

            total_combined_loss += loss.item() * B
            total_samples += B

        return {
            "train_loss": total_combined_loss / max(total_samples, 1),
            "ctc_loss": total_ctc_loss / max(total_samples, 1),
            "attn_loss": total_attn_loss / max(total_samples, 1),
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        decode_method: str = "greedy",
        use_ctc_for_eval: bool = False
    ) -> Dict[str, float]:
        """
        Evaluates model on validation/test set.

        Args:
            decode_method: "greedy" or "beam" (for attention decoder)
            use_ctc_for_eval: If True, uses CTC greedy decode instead of attention
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_refs = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["images"].to(self.device)
            refs = batch["texts"]
            B = images.size(0)

            if use_ctc_for_eval and hasattr(self.model, "ctc_decoder"):
                # CTC decode
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                    log_probs = outputs["ctc_log_probs"]
                decoded_tokens_batch = self.model.decode_ctc(log_probs)
            else:
                # Attention decode
                beam_width = self.config.get("model", {}).get("beam_width", 5)
                decoded_tokens_batch = self.model.decode_attention(
                    images, method=decode_method, beam_width=beam_width
                )

            for pred_tokens, ref_text in zip(decoded_tokens_batch, refs):
                pred_text = self.tokenizer.decode(pred_tokens)
                all_preds.append(pred_text)
                all_refs.append(ref_text)

            total_samples += B

        cer = compute_cer(all_preds, all_refs)
        wer = compute_wer(all_preds, all_refs)
        bg_cer = compute_bg_cer(all_preds, all_refs)

        return {
            "val_cer": cer,
            "val_wer": wer,
            "val_bg_cer": bg_cer,
            "num_samples": total_samples
        }

    def _prepare_attn_targets(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Prepares target token sequences for attention decoder.
        Adds BOS at the start and EOS at the end.
        Pads to max length in batch.

        Returns:
            target_tokens: [B, max_target_len + 2] with BOS and EOS
        """
        bos_idx = self.model.attn_decoder.bos_idx
        eos_idx = self.model.attn_decoder.eos_idx
        pad_idx = self.model.attn_decoder.pad_idx

        # Reconstruct per-sample token sequences from concatenated targets
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        sequences = []
        offset = 0
        for length in target_lengths:
            seq = targets[offset:offset + length].tolist()
            # Add BOS and EOS
            seq = [bos_idx] + seq + [eos_idx]
            sequences.append(seq)
            offset += length

        # Pad to max length
        max_len = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))

        return torch.tensor(padded, dtype=torch.long)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Saves training checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "history": self.history,
            "best_cer": self.best_cer
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(ckpt, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(ckpt, best_path)
            self.logger.info(f"Saved best model (CER={metrics.get('val_cer', 0):.4f})")

    def load_checkpoint(self, path: str):
        """Loads training checkpoint for resuming."""
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_cer = ckpt.get("best_cer", float("inf"))
        self.history = ckpt.get("history", self.history)
        return ckpt.get("epoch", 0)

    def check_early_stopping(self, val_cer: float) -> bool:
        """Returns True if training should stop."""
        if val_cer < self.best_cer:
            self.best_cer = val_cer
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True
            return False

    def update_history(self, train_metrics: Dict, val_metrics: Dict):
        """Updates training history for plotting."""
        self.history["train_loss"].append(train_metrics.get("train_loss", 0))
        self.history["val_cer"].append(val_metrics.get("val_cer", 0))
        self.history["val_wer"].append(val_metrics.get("val_wer", 0))
        self.history["val_bg_cer"].append(val_metrics.get("val_bg_cer", 0))
        self.history["lr"].append(train_metrics.get("lr", 0))

#!/usr/bin/env python3
"""
BANGHTR-X v2: End-to-End Production Training Pipeline
Executes:
  Stage 1: Character Visual Pretraining (Ekush 122 classes)
  Stage 2: Supervised Hybrid CTC + Attention Line HTR (BN-HTRd)
  Stage 3: SCST (RL) Sequence Policy Refinement
  Stage 4: Model Export for Web GUI & Testing

Continuously writes real-time progress to logs/training_status.json
and logs/train_progress.log so it can be monitored from Monitor_Training.ipynb.
"""

import os
import sys
import time
import json
import random
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import pandas as pd
from tqdm import tqdm

# Ensure workspace root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import BanglaCharacterDataset, BanglaLineHTRDataset, collate_line_fn
from src.data.normalizer import BengaliTokenizer
from src.data.transforms import CharacterTransform, AugmentedAspectRatioPadResize
from src.models.banghtr_x import BANGHTR_X_V2
from src.models.vision.backbones import CharacterClassifierBackbone
from src.training.trainer_htr import HTRTrainerV2
from src.training.scst_trainer import SCSTTrainer
from src.evaluation.metrics import compute_cer, compute_wer, compute_bg_cer

STATUS_FILE = os.path.join(PROJECT_ROOT, "logs", "training_status.json")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "train_progress.log")

def update_status(status_dict):
    """Safely writes training progress state to JSON for Monitor_Training.ipynb."""
    status_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    temp_file = STATUS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, STATUS_FILE)

def log_msg(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(f"Starting BANGHTR-X v2 Training Pipeline on device: {device}")
    if torch.cuda.is_available():
        log_msg(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, "configs", "htr_v2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Directories
    ckpt_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["checkpoint_dir"])
    export_dir = os.path.join(PROJECT_ROOT, "exports")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    status = {
        "status": "RUNNING",
        "current_stage": "Stage 1: Character Pretraining",
        "stage_num": 1,
        "total_stages": 4,
        "pid": os.getpid(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage1": {"epochs": 5, "current_epoch": 0, "train_loss": [], "val_acc": []},
        "stage2": {"epochs": 25, "current_epoch": 0, "train_loss": [], "val_cer": [], "val_wer": []},
        "stage3": {"epochs": 5, "current_epoch": 0, "rl_loss": [], "avg_reward": []},
        "completed": False,
        "best_supervised_cer": None,
        "best_rl_cer": None,
        "production_model_path": None
    }
    update_status(status)

    # =========================================================================
    # STAGE 1: Character-Level Pretraining (Ekush 122 classes)
    # =========================================================================
    char_manifest = os.path.join(PROJECT_ROOT, cfg["data"]["char_manifest_path"])
    stem_ckpt_path = os.path.join(ckpt_dir, "stage1_convnext_stem_best.pt")

    if os.path.exists(stem_ckpt_path):
        log_msg(f"Stage 1 checkpoint already exists at {stem_ckpt_path}. Skipping Stage 1.")
        status["stage1"]["status"] = "CACHED_OR_SKIPPED"
        update_status(status)
    elif os.path.exists(char_manifest):
        log_msg("=== STAGE 1: Starting Character-Level Pretraining (122 Classes) ===")
        # Note: CharacterTransform(invert=True) handles black background -> white paper
        train_char_ds = BanglaCharacterDataset(char_manifest, split="train", workspace_root=PROJECT_ROOT)
        val_char_ds = BanglaCharacterDataset(char_manifest, split="val", workspace_root=PROJECT_ROOT)
        log_msg(f"Char Samples: Train={len(train_char_ds):,}, Val={len(val_char_ds):,}")

        num_workers = cfg["data"].get("num_workers", 10)
        char_train_loader = DataLoader(
            train_char_ds, batch_size=256, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        char_val_loader = DataLoader(
            val_char_ds, batch_size=256, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        char_model = CharacterClassifierBackbone(in_channels=1, num_classes=122, hidden_dim=256).to(device)
        char_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        char_optimizer = torch.optim.AdamW(char_model.parameters(), lr=1e-3, weight_decay=1e-4)
        char_epochs = 5
        char_scheduler = CosineAnnealingLR(char_optimizer, T_max=char_epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_char_acc = 0.0
        for ep in range(1, char_epochs + 1):
            char_model.train()
            total_loss, total_samples = 0.0, 0
            for batch in char_train_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                char_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = char_model(imgs)
                    loss = char_criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(char_optimizer)
                scaler.update()

                b_sz = imgs.size(0)
                total_loss += loss.item() * b_sz
                total_samples += b_sz

            char_scheduler.step()
            train_l = total_loss / max(total_samples, 1)

            # Validation
            char_model.eval()
            correct, val_total = 0, 0
            with torch.no_grad():
                for batch in char_val_loader:
                    imgs = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        preds = char_model(imgs).argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = correct / max(val_total, 1)
            status["stage1"]["current_epoch"] = ep
            status["stage1"]["train_loss"].append(train_l)
            status["stage1"]["val_acc"].append(val_acc)
            update_status(status)

            is_best = val_acc > best_char_acc
            if is_best:
                best_char_acc = val_acc
                torch.save(char_model.stem.state_dict(), stem_ckpt_path)

            log_msg(f"Stage 1 [Epoch {ep}/{char_epochs}] Loss: {train_l:.4f} | Val Acc: {val_acc*100:.2f}% {'⭐ Best' if is_best else ''}")
        log_msg(f"Stage 1 Complete! Best Character Val Acc: {best_char_acc*100:.2f}%")
    else:
        log_msg("Warning: Character manifest not found, skipping Stage 1 pretraining.")

    # =========================================================================
    # STAGE 2: Supervised Hybrid HTR Training (BN-HTRd Lines)
    # =========================================================================
    status["current_stage"] = "Stage 2: Supervised Hybrid HTR"
    status["stage_num"] = 2
    update_status(status)

    line_manifest = os.path.join(PROJECT_ROOT, cfg["data"]["manifest_path"])
    tokenizer = BengaliTokenizer.from_manifest(line_manifest)
    log_msg(f"Bengali Tokenizer Vocab Size: {len(tokenizer)} characters")

    # Datasets with Augmentation (preserving white background, black ink)
    train_transform = AugmentedAspectRatioPadResize(target_height=64, max_width=1024, augment=True)
    val_transform = AugmentedAspectRatioPadResize(target_height=64, max_width=1024, augment=False)

    train_line_ds = BanglaLineHTRDataset(line_manifest, tokenizer, split="train", transform=train_transform)
    val_line_ds = BanglaLineHTRDataset(line_manifest, tokenizer, split="val", transform=val_transform)
    log_msg(f"Line HTR Datasets: Train={len(train_line_ds):,}, Val={len(val_line_ds):,}")

    num_workers = cfg["data"].get("num_workers", 10)
    line_train_loader = DataLoader(
        train_line_ds, batch_size=cfg["data"].get("batch_size", 32), shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_line_fn
    )
    line_val_loader = DataLoader(
        val_line_ds, batch_size=cfg["data"].get("val_batch_size", 32), shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_line_fn
    )

    # BANGHTR-X v2 Model Instantiation
    model_v2 = BANGHTR_X_V2(
        num_classes=len(tokenizer),
        in_channels=1,
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        encoder_layers=cfg["model"].get("encoder_layers", 4),
        decoder_layers=cfg["model"].get("decoder_layers", 4),
        num_heads=cfg["model"].get("num_heads", 8),
        use_matra_attn=cfg["model"].get("use_matra_attn", True),
        use_grapheme_moe=cfg["model"].get("use_grapheme_moe", True),
        decoder_type="hybrid"
    ).to(device)

    # Load Stage 1 pretrained stem if available
    if os.path.exists(stem_ckpt_path):
        log_msg("Transferring Stage 1 ConvNeXt visual stem weights...")
        try:
            model_v2.load_pretrained_stem(stem_ckpt_path)
            log_msg("Visual stem weights transferred successfully!")
        except Exception as e:
            log_msg(f"Could not load stem weights ({e}), training stem from scratch.")

    # Optimized Epochs: 25 epochs for fast high-accuracy convergence under 3 hours
    stage2_epochs = 25
    cfg["training"]["epochs"] = stage2_epochs
    status["stage2"]["epochs"] = stage2_epochs

    trainer_v2 = HTRTrainerV2(
        model=model_v2,
        tokenizer=tokenizer,
        config=cfg,
        device=device,
        checkpoint_dir=ckpt_dir
    )
    trainer_v2.init_scheduler(steps_per_epoch=len(line_train_loader))

    best_s2_cer = float("inf")
    best_s2_ckpt = os.path.join(ckpt_dir, "best_banghtr_x_v2.pt")

    log_msg(f"=== STAGE 2: Starting Supervised Hybrid HTR Training ({stage2_epochs} Epochs) ===")
    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        train_metrics = trainer_v2.train_epoch(line_train_loader, epoch)
        val_metrics = trainer_v2.evaluate(line_val_loader, decode_method="greedy")
        elapsed = time.time() - t0

        cer = val_metrics["cer"]
        wer = val_metrics["wer"]
        t_loss = train_metrics["train_loss"]

        status["stage2"]["current_epoch"] = epoch
        status["stage2"]["train_loss"].append(t_loss)
        status["stage2"]["val_cer"].append(cer)
        status["stage2"]["val_wer"].append(wer)

        is_best = cer < best_s2_cer
        if is_best:
            best_s2_cer = cer
            status["best_supervised_cer"] = best_s2_cer
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_v2.state_dict(),
                "optimizer_state_dict": trainer_v2.optimizer.state_dict(),
                "val_cer": cer,
                "val_wer": wer,
                "vocab": tokenizer.vocab,
                "config": cfg
            }, best_s2_ckpt)

        update_status(status)
        log_msg(f"Stage 2 [Epoch {epoch:2d}/{stage2_epochs}] Train Loss: {t_loss:.4f} | Val CER: {cer*100:.2f}% | WER: {wer*100:.2f}% | Time: {elapsed:.1f}s {'⭐ Best' if is_best else ''}")

    log_msg(f"Stage 2 Supervised Training Finished! Best Validation CER: {best_s2_cer*100:.2f}%")

    # =========================================================================
    # STAGE 3: SCST (RL) Sequence Policy Refinement (5 Epochs)
    # =========================================================================
    status["current_stage"] = "Stage 3: SCST Reinforcement Learning"
    status["stage_num"] = 3
    update_status(status)

    log_msg("=== STAGE 3: Starting Self-Critical Sequence Training (SCST RL Refinement) ===")
    # Load best supervised weights
    if os.path.exists(best_s2_ckpt):
        ckpt_data = torch.load(best_s2_ckpt, map_location=device)
        model_v2.load_state_dict(ckpt_data["model_state_dict"])
        log_msg(f"Loaded best supervised weights from {best_s2_ckpt} for RL refinement.")

    rl_epochs = 5
    status["stage3"]["epochs"] = rl_epochs
    scst_trainer = SCSTTrainer(
        model=model_v2,
        tokenizer=tokenizer,
        device=device,
        lr=1e-5,
        num_samples=2,
        temperature=1.2,
        baseline_ema=0.99
    )

    best_rl_ckpt = os.path.join(ckpt_dir, "best_banghtr_x_v2_rl.pt")
    best_rl_cer = best_s2_cer

    for rl_ep in range(1, rl_epochs + 1):
        t0 = time.time()
        rl_metrics = scst_trainer.train_epoch(line_train_loader, rl_ep)
        val_metrics = trainer_v2.evaluate(line_val_loader, decode_method="greedy")
        elapsed = time.time() - t0

        cer = val_metrics["cer"]
        wer = val_metrics["wer"]
        rl_loss = rl_metrics["rl_loss"]
        avg_rew = rl_metrics["avg_reward"]

        status["stage3"]["current_epoch"] = rl_ep
        status["stage3"]["rl_loss"].append(rl_loss)
        status["stage3"]["avg_reward"].append(avg_rew)

        is_best_rl = cer < best_rl_cer
        if is_best_rl:
            best_rl_cer = cer
            status["best_rl_cer"] = best_rl_cer
            torch.save({
                "epoch": rl_ep,
                "model_state_dict": model_v2.state_dict(),
                "val_cer": cer,
                "val_wer": wer,
                "vocab": tokenizer.vocab,
                "config": cfg
            }, best_rl_ckpt)

        update_status(status)
        log_msg(f"Stage 3 RL [Epoch {rl_ep}/{rl_epochs}] Loss: {rl_loss:.4f} | Avg Reward: {avg_rew:.4f} | Val CER: {cer*100:.2f}% | WER: {wer*100:.2f}% | Time: {elapsed:.1f}s {'⭐ Best' if is_best_rl else ''}")

    # =========================================================================
    # STAGE 4: Model Export for Production & Web GUI
    # =========================================================================
    status["current_stage"] = "Stage 4: Model Export"
    status["stage_num"] = 4
    update_status(status)

    production_bundle_path = os.path.join(export_dir, "banghtr_x_production.pt")
    # Select best model available (RL if improved, else Supervised)
    final_ckpt = best_rl_ckpt if os.path.exists(best_rl_ckpt) else best_s2_ckpt
    final_data = torch.load(final_ckpt, map_location="cpu")

    export_bundle = {
        "model_architecture": "BANGHTR_X_V2",
        "state_dict": final_data["model_state_dict"],
        "tokenizer_vocab": tokenizer.vocab,
        "config": cfg,
        "final_cer": final_data.get("val_cer", None),
        "final_wer": final_data.get("val_wer", None),
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_spec": {
            "channels": 1,
            "height": 64,
            "max_width": 1024,
            "background": "white (255)",
            "ink": "black (0)",
            "normalization": "[0, 1] range via / 255.0"
        }
    }
    torch.save(export_bundle, production_bundle_path)
    log_msg(f"✅ Self-contained production model exported to: {production_bundle_path}")

    status["status"] = "COMPLETED"
    status["completed"] = True
    status["production_model_path"] = production_bundle_path
    status["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    update_status(status)
    log_msg("🎉 BANGHTR-X v2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

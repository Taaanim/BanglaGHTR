"""
Training harness for Level 1: Isolated Character Classification.
Trains vision backbone on dataset_2 (Ekush 122 classes) with writer-independent splits.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

from ..models.vision.backbones import CharacterClassifierBackbone
from ..data.datasets import BanglaCharacterDataset
from ..utils.device import get_device, set_seed
from ..utils.logger import setup_logger

class CharacterTrainer:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = setup_logger("CharTrainer")

        self.num_classes = config["model"].get("num_classes", 122)
        self.hidden_dim = config["model"].get("hidden_dim", 256)

        # Instantiate Model
        self.model = CharacterClassifierBackbone(
            in_channels=1,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        lr = config["training"].get("lr", 0.001)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc="Train Char Epoch", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        return total_loss / total, acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc="Val Char Epoch", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        return {"val_loss": total_loss / total, "val_acc": acc}

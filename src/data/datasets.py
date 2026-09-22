"""
PyTorch Datasets and custom collators for Bangla HTR.
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List

from .transforms import AspectRatioPadResize, CharacterTransform
from .normalizer import BengaliTokenizer, normalize_bengali_text

class BanglaCharacterDataset(Dataset):
    """
    PyTorch Dataset for dataset_2 (Ekush 122 isolated character classes).
    Enforces writer-independent evaluation splits.
    """
    def __init__(
        self,
        manifest_csv: str,
        split: str = "train",
        workspace_root: str = ".",
        transform: Optional[CharacterTransform] = None
    ):
        self.workspace_root = workspace_root
        df = pd.read_csv(manifest_csv)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or CharacterTransform()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.workspace_root, str(row["image_path"]))

        with Image.open(img_path) as im:
            img = im.convert("L")

        tensor = self.transform(img)
        label = int(row["class_id"])

        return {
            "image": tensor,
            "label": label,
            "writer_id": str(row["writer_id"]),
            "gender": int(row["gender"]),
            "district": str(row["district"])
        }

class BanglaLineHTRDataset(Dataset):
    """
    PyTorch Dataset for dataset_1 (BN-HTR Line-level sequence recognition).
    Enforces document-independent evaluation splits and aspect-ratio preserving resizing.
    """
    def __init__(
        self,
        manifest_csv: str,
        tokenizer: BengaliTokenizer,
        split: str = "train",
        workspace_root: str = ".",
        target_height: int = 64,
        max_width: int = 1024,
        transform: Optional[Any] = None
    ):
        self.workspace_root = workspace_root
        self.tokenizer = tokenizer
        self.transform = transform or AspectRatioPadResize(target_height=target_height, max_width=max_width)

        df = pd.read_csv(manifest_csv)
        # Filter valid rows with non-empty transcription and existing image paths
        valid_mask = (
            (df["split"] == split) &
            (df["text"].fillna("").str.strip() != "") &
            (df["image_path"].fillna("").str.len() > 0)
        )
        self.data = df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.workspace_root, str(row["image_path"]))

        with Image.open(img_path) as im:
            img = im.convert("L")

        tensor, valid_width = self.transform(img)
        text = normalize_bengali_text(str(row["text"]))
        tokens = self.tokenizer.encode(text)

        return {
            "image": tensor,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "target_len": len(tokens),
            "valid_width": valid_width,
            "text": text,
            "line_id": str(row["line_id"]),
            "doc_id": str(row["doc_id"])
        }

class BanglaWordHTRDataset(Dataset):
    """
    PyTorch Dataset for dataset_1 (BN-HTR Word-level recognition).
    """
    def __init__(
        self,
        manifest_csv: str,
        tokenizer: BengaliTokenizer,
        split: str = "train",
        workspace_root: str = ".",
        target_height: int = 64,
        max_width: int = 256,
        transform: Optional[Any] = None
    ):
        self.workspace_root = workspace_root
        self.tokenizer = tokenizer
        self.transform = transform or AspectRatioPadResize(target_height=target_height, max_width=max_width)

        df = pd.read_csv(manifest_csv)
        valid_mask = (
            (df["split"] == split) &
            (df["text"].fillna("").str.strip() != "") &
            (df["image_path"].fillna("").str.len() > 0)
        )
        self.data = df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.workspace_root, str(row["image_path"]))

        with Image.open(img_path) as im:
            img = im.convert("L")

        tensor, valid_width = self.transform(img)
        text = normalize_bengali_text(str(row["text"]))
        tokens = self.tokenizer.encode(text)

        return {
            "image": tensor,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "target_len": len(tokens),
            "valid_width": valid_width,
            "text": text,
            "word_id": str(row["word_id"])
        }

def collate_line_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch variable-length target token sequences for CTC loss.
    """
    images = torch.stack([item["image"] for item in batch])
    target_lengths = torch.tensor([item["target_len"] for item in batch], dtype=torch.long)
    valid_widths = torch.tensor([item["valid_width"] for item in batch], dtype=torch.long)

    # Concatenate all targets into a 1D tensor for CTCLoss
    targets = torch.cat([item["tokens"] for item in batch])

    texts = [item["text"] for item in batch]
    ids = [item.get("line_id", item.get("word_id", "")) for item in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "valid_widths": valid_widths,
        "texts": texts,
        "ids": ids
    }

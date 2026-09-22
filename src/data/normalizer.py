"""
Bengali text and Unicode normalization utilities.
"""

import unicodedata
import pandas as pd
from typing import Dict, List, Set

def normalize_bengali_text(text: str) -> str:
    """
    Applies Unicode NFC normalization and strips irregular zero-width spaces/joiners.
    Prevents tokenization mismatches caused by varied diacritic combination orders.
    """
    if not isinstance(text, str):
        return ""
    # Unicode NFC normalization
    norm = unicodedata.normalize("NFC", text.strip())
    # Remove zero-width non-joiner / zero-width space if dangling
    norm = norm.replace("\u200b", "").replace("\ufeff", "")
    return norm

class BengaliTokenizer:
    """Character-level tokenizer for Bengali sequence HTR."""
    def __init__(self, vocab_list: List[str] = None):
        self.pad_token = "<PAD>"
        self.blank_token = "<BLANK>" # CTC Blank token (index 0)
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"     # Attention decoder BOS (index 3)
        self.eos_token = "<EOS>"     # Attention decoder EOS (index 4)
        self.space_token = " "

        self.special_tokens = [
            self.blank_token,
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        if vocab_list is not None:
            self.build_vocab(vocab_list)
        else:
            self.char2idx = {}
            self.idx2char = {}

    @property
    def pad_id(self) -> int:
        return self.char2idx.get(self.pad_token, 1)

    @property
    def blank_id(self) -> int:
        return self.char2idx.get(self.blank_token, 0)

    @property
    def unk_id(self) -> int:
        return self.char2idx.get(self.unk_token, 2)

    @property
    def bos_id(self) -> int:
        return self.char2idx.get(self.bos_token, 3)

    @property
    def eos_id(self) -> int:
        return self.char2idx.get(self.eos_token, 4)

    def build_vocab(self, chars: List[str]):
        """Builds char2idx and idx2char dictionaries."""
        all_tokens = list(self.special_tokens)
        for c in sorted(chars):
            if c not in all_tokens:
                all_tokens.append(c)

        self.char2idx = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.idx2char = {idx: tok for idx, tok in enumerate(all_tokens)}

    @classmethod
    def from_manifest(cls, manifest_csv: str, text_col: str = "text") -> "BengaliTokenizer":
        """Builds tokenizer vocabulary automatically from a text manifest."""
        df = pd.read_csv(manifest_csv)
        unique_chars: Set[str] = set()
        for text in df[text_col].dropna():
            norm_text = normalize_bengali_text(str(text))
            for char in norm_text:
                unique_chars.add(char)

        tokenizer = cls()
        tokenizer.build_vocab(list(unique_chars))
        return tokenizer

    def encode(self, text: str) -> List[int]:
        """Encodes Bengali text string to integer sequence."""
        norm_text = normalize_bengali_text(text)
        unk_idx = self.char2idx.get(self.unk_token, 2)
        return [self.char2idx.get(c, unk_idx) for c in norm_text]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decodes integer sequence back to Bengali text string."""
        chars = []
        for idx in indices:
            c = self.idx2char.get(idx, "")
            if remove_special and c in self.special_tokens:
                continue
            chars.append(c)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.char2idx)

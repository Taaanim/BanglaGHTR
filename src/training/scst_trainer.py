"""
Self-Critical Sequence Training (SCST) Trainer for BANGHTR-X v2.
Implements REINFORCE with self-critical baseline for direct metric optimization.

Reference: Rennie et al., "Self-Critical Sequence Training for Image Captioning" (CVPR 2017)
Adapted for HTR with CER/WER reward signals.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from ..models.banghtr_x import BANGHTR_X_V2
from ..data.normalizer import BengaliTokenizer
from ..evaluation.metrics import compute_cer, compute_wer, compute_bg_cer
from .reward import compute_rewards, compute_reward_stats
from ..utils.logger import setup_logger


class SCSTTrainer:
    """
    Self-Critical Sequence Training (SCST) for HTR.

    Algorithm:
    1. For each image, generate two sequences:
       - Greedy: argmax at each step (no gradient) → baseline
       - Sampled: multinomial sampling with temperature → policy
    2. Compute rewards for both: R(sampled), R(greedy)
    3. Advantage = R(sampled) - R(greedy)
    4. REINFORCE gradient: -advantage * log_prob(sampled)
    """

    def __init__(
        self,
        model: BANGHTR_X_V2,
        tokenizer: BengaliTokenizer,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.logger = setup_logger("SCSTTrainer")

        rl_config = config.get("training", {})
        self.lr = rl_config.get("rl_lr", 1e-5)
        self.temperature = rl_config.get("sample_temperature", 1.2)
        self.ema_decay = rl_config.get("baseline_ema", 0.99)
        self.grad_clip = rl_config.get("grad_clip", 5.0)

        reward_weights = rl_config.get("rl_reward_weights", {})
        self.cer_weight = reward_weights.get("cer", 0.6)
        self.wer_weight = reward_weights.get("wer", 0.3)
        self.bg_cer_weight = reward_weights.get("bg_cer", 0.1)

        # Optimizer for RL fine-tuning — typically much lower LR
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-6
        )

        # EMA baseline for reward normalization
        self.ema_reward = 0.0
        self.ema_initialized = False

    def _decode_tokens_to_text(self, token_lists: List[List[int]]) -> List[str]:
        """Converts lists of token indices back to text strings."""
        return [self.tokenizer.decode(tokens) for tokens in token_lists]

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Runs one epoch of SCST training.

        Returns:
            Dict with mean_reward, mean_advantage, mean_rl_loss, cer, wer
        """
        self.model.train()
        # Freeze encoder, only train decoder
        for param in self.model.visual_stem.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'matra_attention'):
            for param in self.model.matra_attention.parameters():
                param.requires_grad = False
        if hasattr(self.model, 'grapheme_moe'):
            for param in self.model.grapheme_moe.parameters():
                param.requires_grad = False
        if hasattr(self.model, 'transformer_encoder'):
            for param in self.model.transformer_encoder.parameters():
                param.requires_grad = False

        total_loss = 0.0
        total_samples = 0
        all_rewards = []
        all_advantages = []
        all_preds = []
        all_refs = []

        micro_batch_size = 8  # Safe chunk size for autoregressive policy gradient

        for batch in tqdm(dataloader, desc="SCST Train", leave=False):
            images = batch["images"].to(self.device)
            refs = batch["texts"]
            B = images.size(0)

            self.optimizer.zero_grad()
            batch_loss = 0.0

            # Process in micro-batches to prevent CUDA OOM during autoregressive sampling
            for i in range(0, B, micro_batch_size):
                sub_imgs = images[i:i + micro_batch_size]
                sub_refs = refs[i:i + micro_batch_size]
                sub_B = sub_imgs.size(0)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # Step 1: Encoder output (no grad)
                    with torch.no_grad():
                        encoder_out = self.model.encode(sub_imgs)
                        greedy_tokens = self.model.attn_decoder.greedy_decode(
                            encoder_out, max_len=80
                        )
                    greedy_texts = self._decode_tokens_to_text(greedy_tokens)

                    # Step 2: Sample decode (policy WITH gradients)
                    sampled_tokens, log_probs = self.model.attn_decoder.sample_decode(
                        encoder_out, temperature=self.temperature, max_len=80
                    )
                    sampled_texts = self._decode_tokens_to_text(sampled_tokens)

                    # Step 3: Compute rewards
                    greedy_rewards = compute_rewards(
                        greedy_texts, sub_refs,
                        self.cer_weight, self.wer_weight, self.bg_cer_weight
                    )
                    sampled_rewards = compute_rewards(
                        sampled_texts, sub_refs,
                        self.cer_weight, self.wer_weight, self.bg_cer_weight
                    )

                    rewards_tensor = torch.tensor(sampled_rewards, device=self.device)
                    baseline_tensor = torch.tensor(greedy_rewards, device=self.device)
                    advantages = rewards_tensor - baseline_tensor  # [sub_B]

                    # Scale loss by micro-batch fraction
                    rl_loss = -(advantages * log_probs).mean() * (sub_B / float(B))

                # Accumulate gradients
                rl_loss.backward()

                batch_loss += rl_loss.item() * B
                all_rewards.extend(sampled_rewards)
                all_advantages.extend(advantages.tolist())
                all_preds.extend(sampled_texts)
                all_refs.extend(sub_refs)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            total_loss += batch_loss
            total_samples += B


        # Compute epoch metrics
        cer = compute_cer(all_preds, all_refs)
        wer = compute_wer(all_preds, all_refs)
        reward_stats = compute_reward_stats(all_rewards)

        return {
            "rl_loss": total_loss / max(total_samples, 1),
            "mean_reward": reward_stats["mean"],
            "mean_advantage": sum(all_advantages) / max(len(all_advantages), 1),
            "cer": cer,
            "wer": wer,
            "ema_reward": self.ema_reward
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, use_beam_search: bool = True) -> Dict[str, float]:
        """
        Evaluates the model using greedy or beam search decoding.

        Returns:
            Dict with val_cer, val_wer, val_bg_cer, mean_reward
        """
        self.model.eval()
        all_preds = []
        all_refs = []

        for batch in tqdm(dataloader, desc="SCST Eval", leave=False):
            images = batch["images"].to(self.device)
            refs = batch["texts"]

            encoder_out = self.model.encode(images)

            if use_beam_search:
                beam_width = self.config.get("model", {}).get("beam_width", 5)
                decoded_tokens = self.model.attn_decoder.beam_search(
                    encoder_out, beam_width=beam_width, max_len=150
                )
            else:
                decoded_tokens = self.model.attn_decoder.greedy_decode(
                    encoder_out, max_len=150
                )

            pred_texts = self._decode_tokens_to_text(decoded_tokens)
            all_preds.extend(pred_texts)
            all_refs.extend(refs)

        cer = compute_cer(all_preds, all_refs)
        wer = compute_wer(all_preds, all_refs)
        bg_cer = compute_bg_cer(all_preds, all_refs)
        rewards = compute_rewards(
            all_preds, all_refs,
            self.cer_weight, self.wer_weight, self.bg_cer_weight
        )

        return {
            "val_cer": cer,
            "val_wer": wer,
            "val_bg_cer": bg_cer,
            "mean_reward": sum(rewards) / max(len(rewards), 1)
        }

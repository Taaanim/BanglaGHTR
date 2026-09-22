"""
Comprehensive End-to-End Verification Script for BANGHTR-X v2 Pipeline.
Tests all submodules, forward/backward passes, decoders, rewards, and datasets.
"""

import os
import sys
import time
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbedding
from src.models.encoder.transformer_encoder import TransformerEncoder
from src.models.decoder.attention_decoder import AttentionDecoder
from src.models.banghtr_x import BANGHTR_X_V2
from src.training.reward import compute_rewards, compute_reward_stats
from src.training.scst_trainer import SCSTTrainer
from src.training.trainer_htr import HTRTrainerV2
from src.evaluation.metrics import compute_all_metrics
from src.data.normalizer import BengaliTokenizer
from src.data.transforms import AugmentedAspectRatioPadResize
from src.data.datasets import BanglaLineHTRDataset, collate_line_fn


def run_verification():
    print("=" * 65)
    print("🚀 BANGHTR-X v2: END-TO-END PIPELINE VERIFICATION SUITE")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Hardware Check: Using {device.type.upper()}")
    if device.type == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 1. Tokenizer Check
    print("\n[2/6] Tokenizer & Manifest Check...")
    manifest_path = "datasets/manifests/dataset_1_lines.csv"
    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    tokenizer = BengaliTokenizer.from_manifest(manifest_path)
    print(f"      Vocab Size: {len(tokenizer)}")
    print(f"      Special IDs: BLANK={tokenizer.blank_id}, PAD={tokenizer.pad_id}, UNK={tokenizer.unk_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
    sample_text = "বাংলা বর্ণমালা"
    enc = tokenizer.encode(sample_text)
    dec = tokenizer.decode(enc)
    assert dec == sample_text, f"Mismatch: {dec} != {sample_text}"
    print(f"      Round-trip encoding verified: '{sample_text}' -> {enc} -> '{dec}'")

    # 2. Architecture & Forward/Backward Pass
    print("\n[3/6] BANGHTR-X v2 Forward & Dual Loss Backward Check...")
    with open("configs/htr_v2.yaml") as f:
        config = yaml.safe_load(f)

    model = BANGHTR_X_V2(
        num_classes=len(tokenizer),
        in_channels=1,
        hidden_dim=128,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=4,
        decoder_type="hybrid"
    ).to(device)

    counts = model.count_parameters()
    print(f"      Total Model Parameters (test config): {counts['total']:,}")

    dummy_img = torch.randn(2, 1, 64, 256, device=device)
    dummy_tgt = torch.tensor([[tokenizer.bos_id, 10, 11, tokenizer.eos_id],
                              [tokenizer.bos_id, 15, 20, tokenizer.eos_id]], device=device)
    
    out = model(dummy_img, target_tokens=dummy_tgt)
    assert "ctc_log_probs" in out, "Missing ctc_log_probs"
    assert "attn_logits" in out, "Missing attn_logits"
    assert "attn_loss" in out, "Missing attn_loss"
    
    loss = out["attn_loss"]
    loss.backward()
    print("      Dual forward pass and gradient backpropagation successful!")

    # 3. Decoders (Greedy, Sampling, Beam Search)
    print("\n[4/6] Decoding Mechanisms Verification...")
    with torch.no_grad():
        # CTC decode
        ctc_tokens = model.decode_ctc(out["ctc_log_probs"])
        assert len(ctc_tokens) == 2
        print(f"      CTC Greedy Decode: {len(ctc_tokens)} sequences generated")

        # Attention Greedy decode
        attn_greedy = model.decode_attention(dummy_img, method="greedy", max_len=15)
        assert len(attn_greedy) == 2
        print(f"      Attention Greedy Decode: {len(attn_greedy)} sequences generated")

        # Attention Beam Search decode (width=3)
        attn_beam = model.decode_attention(dummy_img, method="beam", beam_width=3, max_len=15)
        assert len(attn_beam) == 2
        print(f"      Attention Beam Search (width=3): {len(attn_beam)} sequences generated")

    # 4. SCST / RL Module Check
    print("\n[5/6] SCST Reinforcement Learning Check...")
    sampled_tokens, log_probs = model.attn_decoder.sample_decode(
        model.encode(dummy_img), temperature=1.2, max_len=15
    )
    assert len(sampled_tokens) == 2
    assert log_probs.shape == (2,)
    rl_loss = -log_probs.mean()
    rl_loss.backward()
    print("      Policy gradient computation and backward pass verified!")

    # Reward shaping check
    preds = ["আমার সোনার বাংলা", "আমি তোমায় ভালোবাসি"]
    refs = ["আমার সোনার বাংলা", "আমি তোমায় ভালোবাসি"]
    rewards = compute_rewards(preds, refs)
    assert len(rewards) == 2
    assert abs(rewards[0] - 1.0) < 1e-4
    print(f"      Reward shaping verified: max reward = {rewards[0]:.4f}")

    # 5. Dataset & Trainer Mini-Batch Integration
    print("\n[6/6] End-to-End Trainer Mini-Batch Check...")
    transform = AugmentedAspectRatioPadResize(target_height=64, max_width=512, augment=True)
    dataset = BanglaLineHTRDataset(manifest_path, tokenizer, split="val", transform=transform)
    subset = Subset(dataset, range(4))
    loader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=collate_line_fn)

    trainer = HTRTrainerV2(model, tokenizer, config, device, checkpoint_dir="checkpoints/verify_chk")
    trainer.init_scheduler(steps_per_epoch=len(loader))
    
    train_metrics = trainer.train_epoch(loader, epoch=1)
    eval_metrics = trainer.evaluate(loader, decode_method="greedy")
    print(f"      Train Loss: {train_metrics['train_loss']:.4f}")
    print(f"      Val CER: {eval_metrics['val_cer']*100:.2f}% | WER: {eval_metrics['val_wer']*100:.2f}%")

    print("\n" + "=" * 65)
    print("🎉 ALL 6 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()

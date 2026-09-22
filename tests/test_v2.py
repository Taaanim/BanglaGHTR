import os
import sys
import unittest
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbedding
from src.models.encoder.transformer_encoder import TransformerEncoder
from src.models.decoder.attention_decoder import AttentionDecoder
from src.models.banghtr_x import BANGHTR_X_V2
from src.training.reward import compute_rewards, compute_reward_stats
from src.evaluation.metrics import (
    compute_cer, compute_wer, compute_bg_cer,
    compute_accuracy, compute_ned, compute_all_metrics
)
from src.data.transforms import HTRAugmentation, AugmentedAspectRatioPadResize
from src.data.normalizer import BengaliTokenizer


class TestV2Components(unittest.TestCase):

    def test_positional_encoding(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 50, 64)
        out = pe(x)
        self.assertEqual(out.shape, (2, 50, 64))

        le = LearnedPositionalEmbedding(max_len=100, d_model=64)
        out_le = le(x)
        self.assertEqual(out_le.shape, (2, 50, 64))

    def test_transformer_encoder(self):
        enc = TransformerEncoder(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(2, 30, 64)
        out = enc(x)
        self.assertEqual(out.shape, (2, 30, 64))

    def test_attention_decoder(self):
        dec = AttentionDecoder(
            num_classes=50, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=128, max_seq_len=60, pad_idx=1, bos_idx=3, eos_idx=4
        )
        memory = torch.randn(2, 30, 64)
        targets = torch.tensor([[3, 10, 11, 4, 1], [3, 15, 4, 1, 1]], dtype=torch.long)
        logits, loss = dec(memory, targets)
        # Sequence length is T_dec - 1 = 4
        self.assertEqual(logits.shape, (2, 4, 50))
        self.assertGreater(loss.item(), 0.0)

        # Greedy inference
        decoded = dec.greedy_decode(memory, max_len=10)
        self.assertEqual(len(decoded), 2)

        # Sampling inference with gradient flow
        sampled_tokens, log_probs = dec.sample_decode(memory, max_len=10)
        self.assertEqual(len(sampled_tokens), 2)
        self.assertEqual(log_probs.shape, (2,))
        loss = -log_probs.mean()
        loss.backward()

    def test_banghtr_x_v2_forward(self):
        model = BANGHTR_X_V2(
            num_classes=50,
            in_channels=1,
            hidden_dim=64,
            encoder_layers=2,
            decoder_layers=2,
            num_heads=4,
            decoder_type="hybrid"
        )
        x = torch.randn(2, 1, 64, 128)
        targets = torch.tensor([[3, 10, 11, 4], [3, 15, 20, 4]], dtype=torch.long)

        out = model(x, target_tokens=targets)
        self.assertIn("ctc_log_probs", out)
        self.assertIn("attn_logits", out)
        self.assertIn("attn_loss", out)
        self.assertEqual(out["ctc_log_probs"].shape[1], 2) # Batch dim is 1 in [T, B, C]

        # Test CTC decode
        ctc_res = model.decode_ctc(out["ctc_log_probs"])
        self.assertEqual(len(ctc_res), 2)

        # Test Attention decode
        attn_res = model.decode_attention(x, method="greedy", max_len=15)
        self.assertEqual(len(attn_res), 2)

    def test_reward_module(self):
        preds = ["আমার সোনার বাংলা", "আমি তোমায় ভালোবাসি"]
        refs = ["আমার সোনার বাংলা", "আমি তোমায় ভালোবাসি"]
        rewards = compute_rewards(preds, refs, cer_weight=0.6, wer_weight=0.3, bg_cer_weight=0.1)
        self.assertEqual(len(rewards), 2)
        self.assertAlmostEqual(rewards[0], 1.0, places=3)
        self.assertAlmostEqual(rewards[1], 1.0, places=3)

        stats = compute_reward_stats(rewards)
        self.assertIn("mean", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)

    def test_enhanced_metrics(self):
        preds = ["বাংলা", "ভারত", "দেশ"]
        refs = ["বাংলা", "ভারত", "বিদেশ"]
        acc = compute_accuracy(preds, refs)
        self.assertAlmostEqual(acc, 2.0 / 3.0, places=3)

        ned = compute_ned(preds, refs)
        self.assertGreater(ned, 0.0)
        self.assertLessEqual(ned, 1.0)

        all_m = compute_all_metrics(preds, refs)
        for key in ["cer", "wer", "bg_cer", "accuracy", "ned"]:
            self.assertIn(key, all_m)

    def test_augmentations(self):
        img = Image.new("L", (100, 40), color=200)
        aug = HTRAugmentation(p=1.0)
        aug_img = aug(img)
        self.assertIsInstance(aug_img, Image.Image)

        transform = AugmentedAspectRatioPadResize(target_height=64, max_width=256, is_training=True)
        tensor, valid_w = transform(img)
        self.assertEqual(tensor.shape, (1, 64, 256))
        self.assertGreater(valid_w, 0)


if __name__ == "__main__":
    unittest.main()

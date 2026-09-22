import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.data.normalizer import BengaliTokenizer, normalize_bengali_text
from src.data.grapheme_parser import split_grapheme_clusters, analyze_cluster
from src.data.transforms import AspectRatioPadResize, CharacterTransform
from src.models.vision.backbones import ConvNeXtStem, CharacterClassifierBackbone
from src.models.banghtr_x import BANGHTR_X
from src.evaluation.metrics import compute_cer, compute_wer, compute_bg_cer

class TestBanglaSetup(unittest.TestCase):

    def test_unicode_normalization(self):
        text = "বৈচিত্র্যময় এই পৃথিবীর পরতে পরতে লুকিয়ে আছে বিস্ময়।"
        norm = normalize_bengali_text(text)
        self.assertIsInstance(norm, str)
        self.assertGreater(len(norm), 0)

    def test_grapheme_parser(self):
        word = "বাংলাদেশ"
        clusters = split_grapheme_clusters(word)
        self.assertIn("বা", clusters)
        self.assertIn("লা", clusters)
        self.assertIn("দে", clusters)
        self.assertIn("শ", clusters)

        analysis = analyze_cluster("ক্যা")
        self.assertEqual(analysis["root"], "ক")
        self.assertEqual(analysis["vowel"], "া")
        self.assertEqual(analysis["fola"], "ya_fola")

    def test_transforms_shape(self):
        from PIL import Image
        dummy_img = Image.new("L", (200, 50), color=128)
        transform = AspectRatioPadResize(target_height=64, max_width=512)
        tensor, valid_w = transform(dummy_img)
        self.assertEqual(tensor.shape, (1, 64, 512))
        self.assertGreater(valid_w, 0)

    def test_character_model_forward(self):
        model = CharacterClassifierBackbone(in_channels=1, num_classes=122, hidden_dim=64)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, 122))

    def test_banghtr_x_forward(self):
        model = BANGHTR_X(num_classes=50, in_channels=1, hidden_dim=64)
        x = torch.randn(2, 1, 64, 256)
        log_probs = model(x, mode="accurate") # [T, B, C]
        self.assertEqual(log_probs.shape[1], 2)
        self.assertEqual(log_probs.shape[2], 50)

    def test_metrics(self):
        preds = ["বাংলা", "দেশ"]
        refs = ["বাংলা", "দেশ"]
        cer = compute_cer(preds, refs)
        wer = compute_wer(preds, refs)
        bg_cer = compute_bg_cer(preds, refs)
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer, 0.0)
        self.assertEqual(bg_cer, 0.0)

if __name__ == "__main__":
    unittest.main()

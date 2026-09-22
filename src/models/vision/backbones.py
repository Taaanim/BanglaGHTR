"""
Vision Backbones and Feature Extractors for Bengali Handwriting.
Implements ConvNeXt-inspired convolutional stem with multi-scale feature downsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block for efficient stroke feature learning."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3, stride=stride, groups=in_ch)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        x = self.act(x)
        return x

class ConvNeXtStem(nn.Module):
    """
    Hierarchical ConvNeXt stem for line images.
    Reduces height from H=64 to H=1 while preserving fine horizontal resolution (W/4 or W/8).
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        # Stage 1: [B, 1, 64, W] -> [B, 64, 32, W/2]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            DepthwiseSeparableConv(64, 64)
        )

        # Stage 2: [B, 64, 32, W/2] -> [B, 128, 16, W/4]
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            DepthwiseSeparableConv(128, 128)
        )

        # Stage 3: [B, 128, 16, W/4] -> [B, 256, 4, W/4] (stride (4, 1))
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(4, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            DepthwiseSeparableConv(256, hidden_dim)
        )

        # Final vertical collapse to 1D sequence along horizontal width
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 1, H, W]
        Output: Sequence [B, T, hidden_dim] where T = W / 4
        """
        feat = self.stage1(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)

        # Collapse vertical height: [B, C, 1, T]
        feat = self.vertical_pool(feat)
        feat = feat.squeeze(2) # [B, C, T]
        feat = feat.permute(0, 2, 1) # [B, T, C]
        feat = self.out_proj(feat)
        return feat

class CharacterClassifierBackbone(nn.Module):
    """
    Visual backbone for Level 1: 122-class isolated character pretraining.
    Input: [B, 1, 28, 28] -> Output: logits [B, num_classes]
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 122, hidden_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            DepthwiseSeparableConv(64, 128, stride=2), # 14x14
            DepthwiseSeparableConv(128, 256, stride=2), # 7x7
            DepthwiseSeparableConv(256, hidden_dim, stride=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

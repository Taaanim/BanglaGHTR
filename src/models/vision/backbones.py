"""
Vision Backbones and Feature Extractors for Bengali Handwriting.
BANGHTR-X v3: Upgraded 4-stage ConvNeXt stem with:
  - Residual blocks with LayerScale for stable gradient flow
  - Gradual height compression — no aggressive stride-4 jumps
  - Dual max+avg pooling for vertical collapse:
      MaxPool preserves matra top-zone signal
      AvgPool captures global energy
  - Learnable fusion of dual-pool features
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


class ResidualConvNeXtBlock(nn.Module):
    """
    ConvNeXt-V2 style residual block with inverted bottleneck and LayerScale.
    Channels are preserved. Use DownsampleBlock for stride changes.
    """
    def __init__(self, channels: int, expand_ratio: int = 4, kernel_size: int = 7):
        super().__init__()
        mid = channels * expand_ratio
        pad = kernel_size // 2
        # Depthwise (channel-wise spatial mixing)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                 padding=pad, groups=channels)
        # Channel-last LayerNorm (ConvNeXt style)
        self.norm = nn.LayerNorm(channels)
        # Inverted-bottleneck pointwise expansion
        self.pw1 = nn.Linear(channels, mid)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(mid, channels)
        # LayerScale for stable deep residual training
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)      # [B, H, W, C] for LayerNorm
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x * self.gamma              # LayerScale
        x = x.permute(0, 3, 1, 2)      # [B, C, H, W]
        return residual + x


class DownsampleBlock(nn.Module):
    """Strided patch-merge downsampling with GroupNorm (ConvNeXt patchify style)."""
    def __init__(self, in_ch: int, out_ch: int, stride_h: int = 2, stride_w: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(stride_h, stride_w),
                              stride=(stride_h, stride_w), padding=0)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class ConvNeXtStem(nn.Module):
    """
    BANGHTR-X v3 — 4-Stage Hierarchical ConvNeXt Stem.

    Spatial progression for H=64 input:
      Stage 1: [B,  1, 64, W]   → [B,  96, 32, W/2]  (stride_h=2, stride_w=2)
      Stage 2: [B, 96, 32, W/2] → [B, 192, 16, W/2]  (stride_h=2, stride_w=1)
      Stage 3: [B,192, 16, W/2] → [B, 384,  8, W/2]  (stride_h=2, stride_w=1)
      Stage 4: [B,384,  8, W/2] → [B, 384,  4, W/4]  (stride_h=2, stride_w=2)
      DualPool:                 → [B, 384,  1, W/4] × 2 (max + avg)
      OutProj:  768 → hidden_dim

    Vertical info is preserved separately:
      MaxPool → matra top-zone strength per column  (stored as self._max_feat)
      AvgPool → overall ink energy per column
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 384):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Stage 1: patch embed + 1 residual block
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=(2, 2), padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualConvNeXtBlock(96, expand_ratio=4),
        )

        # Stage 2: downsample + 2 residual blocks
        self.down1 = DownsampleBlock(96, 192, stride_h=2, stride_w=1)
        self.stage2 = nn.Sequential(
            ResidualConvNeXtBlock(192, expand_ratio=4),
            ResidualConvNeXtBlock(192, expand_ratio=4),
        )

        # Stage 3: downsample + 3 residual blocks (main representation stage)
        self.down2 = DownsampleBlock(192, 384, stride_h=2, stride_w=1)
        self.stage3 = nn.Sequential(
            ResidualConvNeXtBlock(384, expand_ratio=4),
            ResidualConvNeXtBlock(384, expand_ratio=4),
            ResidualConvNeXtBlock(384, expand_ratio=4),
        )

        # Stage 4: final spatial compression + 1 residual block
        self.down3 = DownsampleBlock(384, 384, stride_h=2, stride_w=2)
        self.stage4 = nn.Sequential(
            ResidualConvNeXtBlock(384, expand_ratio=4),
        )

        # Dual vertical collapse: both max and avg pooling
        self.max_pool_v = nn.AdaptiveMaxPool2d((1, None))
        self.avg_pool_v = nn.AdaptiveAvgPool2d((1, None))

        # Project concat(avg, max) 768 → hidden_dim
        self.out_proj = nn.Sequential(
            nn.Linear(384 * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Store max feat for matra attention
        self._max_feat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, 1, H, W]
        Output: [B, T, hidden_dim]  where T ≈ W/4
        Side-effect: stores self._max_feat = [B, T, 384] for MatraAttentionModule
        """
        x = self.stage1(x)           # [B,  96, 32, W/2]
        x = self.down1(x)
        x = self.stage2(x)           # [B, 192, 16, W/2]
        x = self.down2(x)
        x = self.stage3(x)           # [B, 384,  8, W/2]
        x = self.down3(x)
        x = self.stage4(x)           # [B, 384,  4, W/4]

        # Dual vertical collapse
        feat_max = self.max_pool_v(x).squeeze(2)   # [B, 384, T]
        feat_avg = self.avg_pool_v(x).squeeze(2)   # [B, 384, T]

        # Cache max-pool features for matra attention (upper-zone signal)
        self._max_feat = feat_max.permute(0, 2, 1).contiguous()  # [B, T, 384]

        # Fuse and project
        feat = torch.cat([feat_avg, feat_max], dim=1)  # [B, 768, T]
        feat = feat.permute(0, 2, 1)                   # [B, T, 768]
        feat = self.out_proj(feat)                      # [B, T, hidden_dim]
        return feat

    def get_max_feat(self) -> torch.Tensor:
        """Returns last computed max-pool spatial feature [B, T, 384] for matra attention."""
        return self._max_feat


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
            DepthwiseSeparableConv(64, 128, stride=2),   # 14x14
            DepthwiseSeparableConv(128, 256, stride=2),  # 7x7
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

    @property
    def stem(self):
        return self.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

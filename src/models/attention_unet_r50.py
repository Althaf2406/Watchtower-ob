"""SSL4EO Attention U-Net with ResNet-50 Encoder, 1x1 Lateral Projections, and MC Dropout2d."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from src.models.weight_surgery import (
    SSLSourceType,
    adapt_ssl4eo_first_layer_weights,
    load_ssl4eo_pretrained_weights,
)


class AdditiveAttentionGate(nn.Module):
    """Oktay et al. (2018) Additive Soft-Attention Gate for suppressing irrelevant background/smoke clutter.

    Computes spatial attention coefficients `alpha in [0, 1]` from a coarse gating signal `g`
    and a higher-resolution skip connection `x`.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class DecoderBlockWithDropout(nn.Module):
    """Attention-gated U-Net decoder stage with bilinear upsampling, double 3x3 convs, and MC Dropout2d."""

    def __init__(
        self,
        in_ch_g: int,
        skip_ch: int,
        out_ch: int,
        dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        inter_ch = max(16, skip_ch // 2)
        self.attention_gate = AdditiveAttentionGate(F_g=in_ch_g, F_l=skip_ch, F_int=inter_ch)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch_g + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, g: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip_att = self.attention_gate(g=g_up, x=skip)
        concat = torch.cat([g_up, skip_att], dim=1)
        return self.conv_block(concat)


class SSL4EOAttentionUNetResNet50(nn.Module):
    """Attention U-Net with SSL4EO ResNet-50 Backbone (~26.24M parameters).

    Architecture highlights:
      - ResNet-50 encoder with 17-channel `conv1` initialized via `adapt_ssl4eo_first_layer_weights`.
      - Lateral 1x1 bottleneck projections reducing encoder feature widths
        `[2048, 1024, 512, 256, 64] -> [256, 192, 128, 64, 32]` to fit comfortably within <16 GB VRAM
        (targeting ~5.45 GB peak VRAM at batch_size=16, 256x256, FP16).
      - `nn.Dropout2d(p=0.15)` in the bottleneck and all 4 decoder stages for isolated MC Dropout UQ.
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_classes: int = 3,
        ssl_source: SSLSourceType = "ssl4eo_s12",
        dropout_p: float = 0.15,
        lateral_channels: Sequence[int] = (256, 192, 128, 64, 32),
        download_ssl_weights: bool = False,
    ) -> None:
        super().__init__()
        if len(lateral_channels) != 5:
            raise ValueError(f"Expected 5 lateral channel dimensions, got {lateral_channels}")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.ssl_source = ssl_source
        self.dropout_p = float(dropout_p)
        self.lateral_channels: Tuple[int, ...] = tuple(int(c) for c in lateral_channels)

        backbone = resnet50(weights=None)
        ssl_state = load_ssl4eo_pretrained_weights("resnet50", ssl_source=ssl_source, download=download_ssl_weights)

        if ssl_state is not None and "conv1.weight" in ssl_state:
            pre_conv1_weight = ssl_state.pop("conv1.weight")
            backbone.load_state_dict(ssl_state, strict=False)
        else:
            # Synthesize source-sensor kernel shape (7-ch for ssl4eo_l, 13-ch for ssl4eo_s12) for weight surgery
            src_c = 7 if ssl_source == "ssl4eo_l" else 13
            base_w = backbone.conv1.weight.detach()  # (64, 3, 7, 7)
            repeats = (src_c + 2) // 3
            pre_conv1_weight = base_w.repeat(1, repeats, 1, 1)[:, :src_c, :, :].contiguous()

        adapted_conv1 = adapt_ssl4eo_first_layer_weights(
            pretrained_weight=pre_conv1_weight,
            source_sensor=ssl_source,
            target_in_chans=in_channels,
        )

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(adapted_conv1)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 256 ch, H/4
        self.layer2 = backbone.layer2  # 512 ch, H/8
        self.layer3 = backbone.layer3  # 1024 ch, H/16
        self.layer4 = backbone.layer4  # 2048 ch, H/32

        c4, c3, c2, c1, c0 = self.lateral_channels  # (256, 192, 128, 64, 32)

        # Lateral 1x1 channel reduction projections: [2048, 1024, 512, 256, 64] -> [256, 192, 128, 64, 32]
        self.lat4 = nn.Sequential(
            nn.Conv2d(2048, c4, kernel_size=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.lat3 = nn.Sequential(
            nn.Conv2d(1024, c3, kernel_size=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.lat2 = nn.Sequential(
            nn.Conv2d(512, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.lat1 = nn.Sequential(
            nn.Conv2d(256, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.lat0 = nn.Sequential(
            nn.Conv2d(64, c0, kernel_size=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        # Attention-gated decoder blocks with Dropout2d(p=0.15)
        self.dec4 = DecoderBlockWithDropout(in_ch_g=c4, skip_ch=c3, out_ch=c3, dropout_p=dropout_p)
        self.dec3 = DecoderBlockWithDropout(in_ch_g=c3, skip_ch=c2, out_ch=c2, dropout_p=dropout_p)
        self.dec2 = DecoderBlockWithDropout(in_ch_g=c2, skip_ch=c1, out_ch=c1, dropout_p=dropout_p)
        self.dec1 = DecoderBlockWithDropout(in_ch_g=c1, skip_ch=c0, out_ch=c0, dropout_p=dropout_p)

        # Final 2x upsampling refinement and segmentation head (H/2 -> H)
        self.final_head = nn.Sequential(
            nn.Conv2d(c0, c0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(c0, num_classes, kernel_size=1, bias=True),
        )

        # Expose target layer for Captum LayerGradCam attribution
        self.target_cam_layer: nn.Module = self.dec1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]

        # Encoder forward pass
        s0 = self.relu(self.bn1(self.conv1(x)))   # (B, 64, H/2, W/2)
        s1 = self.layer1(self.maxpool(s0))        # (B, 256, H/4, W/4)
        s2 = self.layer2(s1)                      # (B, 512, H/8, W/8)
        s3 = self.layer3(s2)                      # (B, 1024, H/16, W/16)
        s4 = self.layer4(s3)                      # (B, 2048, H/32, W/32)

        # 1x1 Lateral bottleneck projections
        b4 = self.lat4(s4)                        # (B, 256, H/32, W/32)
        b3 = self.lat3(s3)                        # (B, 192, H/16, W/16)
        b2 = self.lat2(s2)                        # (B, 128, H/8, W/8)
        b1 = self.lat1(s1)                        # (B, 64, H/4, W/4)
        b0 = self.lat0(s0)                        # (B, 32, H/2, W/2)

        # Attention U-Net decoder
        d3 = self.dec4(b4, b3)                    # (B, 192, H/16, W/16)
        d2 = self.dec3(d3, b2)                    # (B, 128, H/8, W/8)
        d1 = self.dec2(d2, b1)                    # (B, 64, H/4, W/4)
        d0 = self.dec1(d1, b0)                    # (B, 32, H/2, W/2)

        up_full = F.interpolate(d0, size=input_hw, mode="bilinear", align_corners=False)
        logits = self.final_head(up_full)         # (B, num_classes, H, W)
        return logits


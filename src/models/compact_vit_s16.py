"""SSL4EO Compact ViT-S/16 Segmentation Model with 2D Bicubic Pos-Embed Interpolation & 4-Stage ConvTranspose2d Head."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.weight_surgery import (
    SSLSourceType,
    adapt_ssl4eo_first_layer_weights,
    load_ssl4eo_pretrained_weights,
)


def interpolate_pos_embed_2d(
    pos_embed: torch.Tensor,
    target_grid_size: Tuple[int, int] = (16, 16),
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    """Perform 2D bicubic interpolation on ViT `pos_embed` from e.g. `14x14` (224x224) to `16x16` (256x256).

    Args:
        pos_embed: Positional embedding tensor of shape `(1, num_prefix_tokens + H_src * W_src, D)`.
        target_grid_size: Target patch grid `(H_dst, W_dst)`, e.g., `(16, 16)` for 256x256 with patch_size=16.
        num_prefix_tokens: Number of prefix tokens (1 for CLS token).

    Returns:
        Interpolated positional embedding of shape `(1, num_prefix_tokens + H_dst * W_dst, D)`.
    """
    prefix_tokens = pos_embed[:, :num_prefix_tokens, :]
    patch_tokens = pos_embed[:, num_prefix_tokens:, :]

    b, num_patches, embed_dim = patch_tokens.shape
    src_h = int(math.isqrt(num_patches))
    src_w = num_patches // src_h
    dst_h, dst_w = int(target_grid_size[0]), int(target_grid_size[1])

    if src_h == dst_h and src_w == dst_w:
        return pos_embed

    patch_2d = patch_tokens.reshape(b, src_h, src_w, embed_dim).permute(0, 3, 1, 2).float()
    patch_interp = F.interpolate(
        patch_2d,
        size=(dst_h, dst_w),
        mode="bicubic",
        align_corners=False,
    )
    patch_flat = patch_interp.permute(0, 2, 3, 1).reshape(b, dst_h * dst_w, embed_dim)
    patch_flat = patch_flat.to(dtype=pos_embed.dtype)

    return torch.cat([prefix_tokens, patch_flat], dim=1)


class ProgressiveTransposeStage(nn.Module):
    """Single 2x spatial upsampling stage using `ConvTranspose2d` + 3x3 refinement + `Dropout2d`."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.15) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SSL4EOCompactViTS16Seg(nn.Module):
    """SSL4EO Compact ViT-Small/16 Semantic Segmentation Architecture (~22.81M parameters).

    Key design features:
      - `vit_small_patch16_224` backbone (`embed_dim=384`, `depth=12`, `num_heads=6`) with first-layer
        patch projection (`Conv2d(17, 384, kernel_size=16, stride=16)`) adapted via `adapt_ssl4eo_first_layer_weights`.
      - 2D bicubic interpolation of `pos_embed` from `14x14` (224x224) to `16x16` (256x256).
      - Multi-scale intermediate token extraction at transformer blocks `[2, 5, 8, 11]`.
      - 4-stage `ConvTranspose2d` progressive upsampling decoder (`16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256`)
        with `nn.Dropout2d(p=0.15)` for stochastic MC Dropout uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_classes: int = 3,
        img_size: int = 256,
        patch_size: int = 16,
        ssl_source: SSLSourceType = "ssl4eo_s12",
        extract_blocks: Sequence[int] = (2, 5, 8, 11),
        dropout_p: float = 0.15,
        download_ssl_weights: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.ssl_source = ssl_source
        self.extract_blocks: Tuple[int, ...] = tuple(int(b) for b in extract_blocks)
        self.dropout_p = float(dropout_p)

        vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_rate=dropout_p,
            proj_drop_rate=dropout_p,
        )
        embed_dim = int(vit.embed_dim)  # 384 for ViT-S/16
        self.embed_dim = embed_dim

        ssl_state = load_ssl4eo_pretrained_weights(
            "vit_small_patch16_224",
            ssl_source=ssl_source,
            download=download_ssl_weights,
        )
        if ssl_state is not None and "patch_embed.proj.weight" in ssl_state:
            pre_patch_weight = ssl_state.pop("patch_embed.proj.weight")
            vit.load_state_dict(ssl_state, strict=False)
        else:
            src_c = 7 if ssl_source == "ssl4eo_l" else 13
            base_w = vit.patch_embed.proj.weight.detach()  # (384, 3, 16, 16)
            repeats = (src_c + 2) // 3
            pre_patch_weight = base_w.repeat(1, repeats, 1, 1)[:, :src_c, :, :].contiguous()

        adapted_patch_weight = adapt_ssl4eo_first_layer_weights(
            pretrained_weight=pre_patch_weight,
            source_sensor=ssl_source,
            target_in_chans=in_channels,
        )

        # Replace patch embedding projection with 17-channel Conv2d(17, 384, 16, 16)
        new_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=vit.patch_embed.proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.copy_(adapted_patch_weight)
            if new_proj.bias is not None and vit.patch_embed.proj.bias is not None:
                new_proj.bias.copy_(vit.patch_embed.proj.bias)

        vit.patch_embed.proj = new_proj
        vit.patch_embed.img_size = (img_size, img_size)
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        vit.patch_embed.grid_size = (grid_h, grid_w)
        vit.patch_embed.num_patches = grid_h * grid_w

        # Perform 2D bicubic interpolation on pos_embed from 14x14 (224x224) to 16x16 (256x256)
        num_prefix = getattr(vit, "num_prefix_tokens", 1)
        self.num_prefix_tokens = int(num_prefix)
        with torch.no_grad():
            interp_pos = interpolate_pos_embed_2d(
                vit.pos_embed.detach(),
                target_grid_size=(grid_h, grid_w),
                num_prefix_tokens=self.num_prefix_tokens,
            )
        vit.pos_embed = nn.Parameter(interp_pos)

        self.vit = vit

        # Multi-scale pyramid fusion across extracted blocks [2, 5, 8, 11]
        num_extracted = len(self.extract_blocks)
        self.multi_level_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * num_extracted, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
        )

        # 4-stage ConvTranspose2d progressive upsampling head (16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256)
        self.up_stage1 = ProgressiveTransposeStage(256, 192, dropout_p=dropout_p)
        self.up_stage2 = ProgressiveTransposeStage(192, 128, dropout_p=dropout_p)
        self.up_stage3 = ProgressiveTransposeStage(128, 64, dropout_p=dropout_p)
        self.up_stage4 = ProgressiveTransposeStage(64, 32, dropout_p=dropout_p)

        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

        # Expose target spatial layer for Captum LayerGradCam attribution
        self.target_cam_layer: nn.Module = self.up_stage3

    def _forward_tokens(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """Run patch embedding, 2D pos_embed addition, and collect spatial feature maps at `extract_blocks`."""
        b, _, h, w = x.shape
        gh = h // self.patch_size
        gw = w // self.patch_size

        # Project directly with Conv2d to allow dynamic spatial resolutions seamlessly
        feat_proj = self.vit.patch_embed.proj(x)  # (B, D, gh, gw)
        tokens = feat_proj.flatten(2).transpose(1, 2)  # (B, gh * gw, D)

        if getattr(self.vit, "cls_token", None) is not None:
            cls_tokens = self.vit.cls_token.expand(b, -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        pos_embed = self.vit.pos_embed
        if pos_embed.shape[1] != tokens.shape[1]:
            pos_embed = interpolate_pos_embed_2d(
                pos_embed,
                target_grid_size=(gh, gw),
                num_prefix_tokens=self.num_prefix_tokens,
            )

        tokens = self.vit.pos_drop(tokens + pos_embed)

        extracted_maps: List[torch.Tensor] = []
        extract_set = set(self.extract_blocks)

        for idx, blk in enumerate(self.vit.blocks):
            tokens = blk(tokens)
            if idx in extract_set:
                normed = self.vit.norm(tokens) if idx == len(self.vit.blocks) - 1 else tokens
                patch_tok = normed[:, self.num_prefix_tokens :, :]  # (B, gh * gw, D)
                spatial_map = patch_tok.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
                extracted_maps.append(spatial_map)

        return extracted_maps, (gh, gw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        extracted_maps, _ = self._forward_tokens(x)

        fused = self.multi_level_fusion(torch.cat(extracted_maps, dim=1))  # (B, 256, 16, 16)
        u1 = self.up_stage1(fused)  # (B, 192, 32, 32)
        u2 = self.up_stage2(u1)     # (B, 128, 64, 64)
        u3 = self.up_stage3(u2)     # (B, 64, 128, 128)
        u4 = self.up_stage4(u3)     # (B, 32, 256, 256)

        if u4.shape[-2:] != input_hw:
            u4 = F.interpolate(u4, size=input_hw, mode="bilinear", align_corners=False)

        logits = self.seg_head(u4)  # (B, num_classes, H, W)
        return logits


"""Captum Feature Attribution (IntegratedGradients, Saliency, LayerGradCam) over Spatial ROI Masks."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam, Saliency

from src.data.sen2fire_dataset import SEN2FIRE_BAND_NAMES

AggregationMode = Literal["class_logit", "entropy"]


class SegmentationAttributionWrapper(nn.Module):
    """Scalar-output wrapper reducing 2D segmentation maps `(B, C, H, W)` over a spatial ROI mask for Captum.

    Supports two aggregation modes:
      - `"class_logit"`: Mean target-class logit over the spatial ROI mask.
      - `"entropy"`: Mean predictive Shannon entropy (in FP32) over the spatial ROI mask.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: AggregationMode = "class_logit",
        target_class: int = 2,
        roi_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if mode not in ("class_logit", "entropy"):
            raise ValueError(f"Unsupported aggregation mode {mode!r}. Expected 'class_logit' or 'entropy'.")
        self.model = model
        self.mode = mode
        self.target_class = int(target_class)
        self.roi_mask = roi_mask
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x).float()  # (B, C, H, W) in FP32
        b, c, h, w = logits.shape

        if self.roi_mask is not None:
            mask = self.roi_mask.to(device=logits.device, dtype=torch.float32)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and b > 1:
                mask = mask.expand(b, -1, -1)
        else:
            mask = torch.ones((b, h, w), device=logits.device, dtype=torch.float32)

        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)

        if self.mode == "class_logit":
            cls_idx = min(max(0, self.target_class), c - 1)
            target_map = logits[:, cls_idx, :, :]  # (B, H, W)
        else:
            probs = F.softmax(logits, dim=1)
            probs_safe = probs.clamp(min=self.eps, max=1.0)
            target_map = -torch.sum(probs * torch.log(probs_safe), dim=1)  # (B, H, W)

        scalar_per_sample = (target_map * mask).sum(dim=(1, 2)) / denom  # (B,)
        return scalar_per_sample


def _resolve_cam_layer(model: nn.Module, target_layer: Optional[nn.Module] = None) -> nn.Module:
    """Resolve a 2D convolutional layer for `LayerGradCam`."""
    if target_layer is not None:
        return target_layer
    if hasattr(model, "target_cam_layer") and isinstance(getattr(model, "target_cam_layer"), nn.Module):
        return getattr(model, "target_cam_layer")
    # Fallback: find last Conv2d layer in model
    last_conv: Optional[nn.Module] = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("Could not automatically locate a Conv2d layer for LayerGradCam.")
    return last_conv


def compute_spectral_feature_attributions(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 2,
    roi_mask: Optional[torch.Tensor] = None,
    mode: AggregationMode = "class_logit",
    ig_steps: int = 16,
    target_layer: Optional[nn.Module] = None,
    channel_names: Sequence[str] = SEN2FIRE_BAND_NAMES,
) -> Dict[str, Any]:
    """Compute IntegratedGradients, Saliency, and LayerGradCam attributions and normalized 17-channel importance shares.

    Args:
        model: Segmentation model (`SSL4EOAttentionUNetResNet50` or `SSL4EOCompactViTS16Seg`).
        x: Input tensor of shape `(B, 17, H, W)`.
        target_class: Semantic class index (`1` for Burn Scar, `2` for Active Flame Front).
        roi_mask: Optional binary mask `(B, H, W)` or `(H, W)` restricting spatial aggregation.
        mode: `"class_logit"` or `"entropy"`.
        ig_steps: Number of Riemann integration steps for `IntegratedGradients`.
        target_layer: Optional layer for `LayerGradCam`.
        channel_names: Names for the 17 input channels.

    Returns:
        Dictionary containing:
          - `integrated_gradients`: `(B, C, H, W)` tensor
          - `saliency`: `(B, C, H, W)` tensor
          - `layer_gradcam`: `(B, 1, H, W)` upsampled spatial attribution map
          - `channel_importance_shares`: `(C,)` float32 tensor summing strictly to `1.0`
          - `channel_ranking`: Dictionary mapping channel names to normalized importance shares
    """
    model.eval()
    x_input = x.detach().float().clone().requires_grad_(True)
    baseline = torch.zeros_like(x_input)

    wrapper = SegmentationAttributionWrapper(
        model=model,
        mode=mode,
        target_class=target_class,
        roi_mask=roi_mask,
    )

    # 1. Integrated Gradients
    ig = IntegratedGradients(wrapper)
    ig_attr = ig.attribute(
        x_input,
        baselines=baseline,
        n_steps=max(2, int(ig_steps)),
        internal_batch_size=x_input.shape[0],
    ).detach().float()

    # 2. First-order Saliency
    saliency = Saliency(wrapper)
    sal_attr = saliency.attribute(x_input, abs=True).detach().float()

    # 3. LayerGradCam
    cam_layer = _resolve_cam_layer(model, target_layer)
    lgc = LayerGradCam(wrapper, cam_layer)
    cam_raw = lgc.attribute(x_input, relu_attributions=True).detach().float()
    if cam_raw.shape[-2:] != x_input.shape[-2:]:
        cam_upsampled = F.interpolate(
            cam_raw,
            size=x_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        cam_upsampled = cam_raw

    # Compute normalized per-channel importance shares from Integrated Gradients magnitude
    per_channel_mag = ig_attr.abs().mean(dim=(0, 2, 3))  # (C,)
    total_mag = per_channel_mag.sum()
    if float(total_mag.item()) <= 1e-12:
        # Fallback to uniform shares if gradients vanish on synthetic zero-response inputs
        channel_shares = torch.full_like(per_channel_mag, 1.0 / float(per_channel_mag.numel()))
    else:
        channel_shares = per_channel_mag / total_mag

    # Guarantee exact sum to 1.0 in float32
    channel_shares = channel_shares / channel_shares.sum()

    ranking: Dict[str, float] = {}
    for idx, share in enumerate(channel_shares.cpu().tolist()):
        name = channel_names[idx] if idx < len(channel_names) else f"Channel_{idx}"
        ranking[name] = float(share)

    return {
        "integrated_gradients": ig_attr,
        "saliency": sal_attr,
        "layer_gradcam": cam_upsampled,
        "channel_importance_shares": channel_shares,
        "channel_ranking": ranking,
    }


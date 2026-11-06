"""Unit tests for Phase 3: Captum Spectral & Spatial Attribution Explainer."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.attribution.captum_explainer import (
    SegmentationAttributionWrapper,
    compute_spectral_feature_attributions,
)


class TinySegModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(17, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 3, kernel_size=1)
        self.target_cam_layer = self.conv1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


def test_segmentation_attribution_wrapper() -> None:
    model = TinySegModel()
    x = torch.randn(2, 17, 16, 16)
    roi = torch.zeros((2, 16, 16), dtype=torch.float32)
    roi[:, 4:12, 4:12] = 1.0

    wrap_logit = SegmentationAttributionWrapper(model, mode="class_logit", target_class=2, roi_mask=roi)
    scalar_logit = wrap_logit(x)
    assert scalar_logit.shape == (2,)
    assert torch.isfinite(scalar_logit).all()

    wrap_ent = SegmentationAttributionWrapper(model, mode="entropy", roi_mask=roi)
    scalar_ent = wrap_ent(x)
    assert scalar_ent.shape == (2,)
    assert torch.isfinite(scalar_ent).all()


def test_spectral_attributions_sum_to_one() -> None:
    model = TinySegModel()
    x = torch.randn(1, 17, 16, 16)

    attr = compute_spectral_feature_attributions(
        model=model,
        x=x,
        target_class=2,
        mode="class_logit",
        ig_steps=4,
    )

    assert "integrated_gradients" in attr
    assert "saliency" in attr
    assert "layer_gradcam" in attr
    assert "channel_importance_shares" in attr

    shares = attr["channel_importance_shares"]
    assert shares.shape == (17,)
    shares_sum = float(shares.sum().item())
    assert math.isclose(shares_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert len(attr["channel_ranking"]) == 17


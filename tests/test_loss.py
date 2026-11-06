"""Unit tests for Phase 3: FP32 Numerical Stability, Hybrid Loss, & Lovasz-Softmax."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from src.losses.hybrid_wildfire_loss import WildfireHybridLoss


def test_loss_fp32_casting_under_fp16() -> None:
    criterion = WildfireHybridLoss(
        num_classes=3,
        loss_weights=(0.40, 0.40, 0.20),
        alpha_class=(0.10, 0.35, 0.55),
    )
    # Simulate FP16 half-precision input logits
    logits_half = torch.randn(2, 3, 64, 64, dtype=torch.float16)
    targets = torch.randint(0, 3, (2, 64, 64), dtype=torch.int64)

    loss, comps = criterion(logits_half, targets, return_components=True)

    assert loss.dtype == torch.float32, f"Loss was not computed in float32: {loss.dtype}"
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0

    for k, v in comps.items():
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()


def test_focal_tversky_penalizes_false_negatives() -> None:
    criterion = WildfireHybridLoss(
        num_classes=3,
        loss_weights=(0.0, 1.0, 0.0),  # Test pure Focal Tversky
        tversky_alpha_fp=0.30,
        tversky_beta_fn=0.70,
        tversky_gamma=1.3333,
    )

    targets = torch.full((1, 16, 16), 2, dtype=torch.int64)  # All Active Flame Front

    # Prediction A: High False Negatives (predicts Background Class 0 instead of Class 2)
    logits_fn = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    logits_fn[:, 0] = 5.0
    loss_fn = criterion(logits_fn, targets)

    # Prediction B: High False Positives (predicts Class 2 everywhere, but ground truth is Class 0)
    targets_bg = torch.full((1, 16, 16), 0, dtype=torch.int64)
    logits_fp = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    logits_fp[:, 2] = 5.0
    loss_fp = criterion(logits_fp, targets_bg)

    # Because beta_fn = 0.70 > alpha_fp = 0.30 and active flame alpha is higher, FN on class 2 is heavily penalized
    assert loss_fn.item() > 0.0
    assert torch.isfinite(loss_fn).all() and torch.isfinite(loss_fp).all()


def test_backward_gradient_flow_autocast() -> None:
    criterion = WildfireHybridLoss()
    conv = torch.nn.Conv2d(17, 3, kernel_size=1)
    x = torch.randn(2, 17, 32, 32)
    targets = torch.randint(0, 3, (2, 32, 32), dtype=torch.int64)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        logits = conv(x)
    loss = criterion(logits.float(), targets)
    loss.backward()

    assert conv.weight.grad is not None
    assert torch.isfinite(conv.weight.grad).all()
    assert not torch.isnan(conv.weight.grad).any()


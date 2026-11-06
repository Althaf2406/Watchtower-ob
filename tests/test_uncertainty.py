"""Unit tests for Phase 3: Isolated MC Dropout & Epistemic/Aleatoric BALD Uncertainty Engine."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.uncertainty.mc_ensemble_uq import enable_mc_dropout_only, predict_with_uncertainty


class DummyNetWithBNAndDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(17, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(p=0.50)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        return self.conv2(x)


def test_mc_dropout_isolation_preserves_batchnorm() -> None:
    model = DummyNetWithBNAndDropout()
    # Initialize running mean/var with non-zero values
    with torch.no_grad():
        model.bn1.running_mean.fill_(0.5)
        model.bn1.running_var.fill_(1.2)

    mean_initial = model.bn1.running_mean.clone()
    var_initial = model.bn1.running_var.clone()

    enable_mc_dropout_only(model)

    assert not model.bn1.training, "BatchNorm2d was incorrectly set to training mode!"
    assert model.drop.training, "Dropout2d was not enabled for stochastic inference!"

    # Run forward passes with different batches
    for _ in range(5):
        x = torch.randn(4, 17, 32, 32) * 5.0 + 3.0
        _ = model(x)

    # Invariant: running stats must remain identical
    assert torch.equal(model.bn1.running_mean, mean_initial)
    assert torch.equal(model.bn1.running_var, var_initial)


def test_predict_with_uncertainty_modes_and_mi_nonnegativity() -> None:
    m1 = DummyNetWithBNAndDropout()
    m2 = DummyNetWithBNAndDropout()
    x = torch.randn(2, 17, 32, 32)

    # 1. mc_dropout
    out_mc = predict_with_uncertainty(m1, x, mode="mc_dropout", num_mc_passes=6, use_amp=False)
    assert out_mc["mean_probs"].shape == (2, 3, 32, 32)
    assert out_mc["pred_mask"].shape == (2, 32, 32)
    assert out_mc["predictive_entropy"].shape == (2, 32, 32)
    assert out_mc["aleatoric_entropy"].shape == (2, 32, 32)
    assert out_mc["epistemic_mi"].shape == (2, 32, 32)
    assert (out_mc["epistemic_mi"] >= 0.0).all(), "Epistemic MI has negative values!"

    # 2. deep_ensemble
    out_ens = predict_with_uncertainty([m1, m2], x, mode="deep_ensemble", use_amp=False)
    assert out_ens["mean_probs"].shape == (2, 3, 32, 32)
    assert (out_ens["epistemic_mi"] >= 0.0).all()

    # 3. hybrid_ensemble_mc
    out_hyb = predict_with_uncertainty([m1, m2], x, mode="hybrid_ensemble_mc", num_mc_passes=3, use_amp=False)
    assert out_hyb["mean_probs"].shape == (2, 3, 32, 32)
    assert (out_hyb["epistemic_mi"] >= 0.0).all()


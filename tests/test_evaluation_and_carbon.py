"""Unit tests for Phase 4: Calibration Evaluation, Smoke-FCER Mask, & GFW Carbon Accounting."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from src.carbon.gfw_carbon_accounting import (
    estimate_burn_scar_carbon_loss,
    track_pipeline_carbon_footprint,
)
from src.evaluation.calibration_metrics import (
    SmokeObscuredSegmentationAndCalibrationEvaluator,
)


def test_smoke_fcer_mask_construction() -> None:
    evaluator = SmokeObscuredSegmentationAndCalibrationEvaluator(
        num_classes=3,
        smoke_aerosol_threshold=1.0,
        boundary_kernel_size=5,  # Smaller kernel for test
    )
    targets = torch.zeros((1, 32, 32), dtype=torch.int64)
    targets[0, 16, 16] = 2  # Active flame front at center

    raw_aerosol = torch.zeros((1, 32, 32), dtype=torch.float32)
    raw_aerosol[0, 2:6, 2:6] = 1.8  # Smoke cluster at top-left

    fcer_mask = evaluator.build_smoke_fcer_mask(targets, raw_aerosol)
    assert fcer_mask.shape == (1, 32, 32)
    assert fcer_mask[0, 3, 3].item() is True, "Smoke area not included in FCER mask"
    assert fcer_mask[0, 16, 16].item() is True, "Fire center not included in FCER mask"
    assert fcer_mask[0, 17, 17].item() is True, "Dilation buffer not included in FCER mask"


def test_evaluator_metrics_computation() -> None:
    evaluator = SmokeObscuredSegmentationAndCalibrationEvaluator(num_classes=3)
    probs = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    probs[:, 1] = 0.8  # Predict class 1
    probs[:, 0] = 0.1
    probs[:, 2] = 0.1

    targets = torch.ones((2, 32, 32), dtype=torch.int64)  # Ground truth is class 1

    evaluator.update(probs, targets)
    metrics = evaluator.compute()

    assert "mIoU" in metrics
    assert "macro_F1" in metrics
    assert "NLL" in metrics
    assert "Brier_Score" in metrics
    assert "ECE" in metrics
    assert "AUSE" in metrics

    assert metrics["per_class"]["Burn_Scar"]["IoU"] == 1.0
    assert metrics["Brier_Score"] < 0.10


def test_gfw_carbon_accounting_three_regimes() -> None:
    probs = np.zeros((3, 64, 64), dtype=np.float32)
    # Background
    probs[0] = 0.1
    # Burn Scar region (quadrant)
    probs[1, 16:48, 16:48] = 0.8
    # Active Flame Front border
    probs[2, 16:24, 16:48] = 0.85
    # Normalize probabilities
    probs = probs / probs.sum(axis=0, keepdims=True)

    variance = np.full((3, 64, 64), 0.04, dtype=np.float32)

    carbon_report = estimate_burn_scar_carbon_loss(
        mean_probs=probs,
        pred_variance=variance,
        pixel_resolution_m=10.0,
    )

    for regime in ("conservative_lower_bound", "nominal_mean", "upper_bound_95ci"):
        assert regime in carbon_report
        res = carbon_report[regime]
        assert res["burned_area_ha"] >= 0.0
        assert res["total_carbon_loss_tc"] >= 0.0
        assert res["co2_equivalent_tonnes"] >= 0.0

    # Upper bound should report >= burned area than conservative lower bound
    assert (
        carbon_report["upper_bound_95ci"]["burned_area_ha"]
        >= carbon_report["conservative_lower_bound"]["burned_area_ha"]
    )


def test_track_pipeline_carbon_footprint() -> None:
    with track_pipeline_carbon_footprint(project_suffix="UnitTest", enabled=False) as footprint:
        assert "emissions_kg_co2eq" in footprint
        footprint["emissions_kg_co2eq"] = 0.0042

    assert footprint["emissions_kg_co2eq"] == 0.0042


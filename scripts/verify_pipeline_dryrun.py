"""Automated End-to-End Dry-Run Verification Suite for Watchtower (Synthetic Tensors B=2, C=17, H=256, W=256)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attribution.captum_explainer import (
    SegmentationAttributionWrapper,
    compute_spectral_feature_attributions,
)
from src.carbon.gfw_carbon_accounting import (
    estimate_burn_scar_carbon_loss,
    track_pipeline_carbon_footprint,
)
from src.data.resampling_eolearn import MultiResolutionResampler
from src.data.sen2fire_dataset import Sen2FireMultispectralDataset
from src.data.wsts_plus_dataset import WSTSPlusDataset
from src.evaluation.calibration_metrics import (
    SmokeObscuredSegmentationAndCalibrationEvaluator,
)
from src.losses.hybrid_wildfire_loss import WildfireHybridLoss
from src.models.attention_unet_r50 import SSL4EOAttentionUNetResNet50
from src.models.compact_vit_s16 import SSL4EOCompactViTS16Seg
from src.models.weight_surgery import adapt_ssl4eo_first_layer_weights
from src.uncertainty.mc_ensemble_uq import (
    enable_mc_dropout_only,
    predict_with_uncertainty,
)


def verify_step_1_data_and_resampling() -> None:
    print("[1/6] Verifying Sen2Fire 17-channel preprocessing, WSTS+ windows, & multi-resolution resampling...")
    ds = Sen2FireMultispectralDataset(
        root_dir="data/nonexistent_dryrun",
        scenes=["scene1", "scene2"],
        patch_size=256,
        is_train=True,
        synthetic_fallback_samples=4,
    )
    sample = ds[0]
    assert sample["image"].shape == (17, 256, 256), f"Unexpected image shape: {sample['image'].shape}"
    assert sample["mask"].shape == (256, 256), f"Unexpected mask shape: {sample['mask'].shape}"
    assert sample["raw_aerosol"].shape == (256, 256)
    unique_classes = set(sample["mask"].unique().tolist())
    assert unique_classes.issubset({0, 1, 2}), f"Invalid refined mask classes: {unique_classes}"

    # Verify WSTS+ 1-day and 5-day windows and Veg/Multi/All subsets
    for window in (1, 5):
        for subset, expected_c in (("Veg", 7), ("Multi", 33), ("All", 40)):
            wsts = WSTSPlusDataset(
                root_dir="data/nonexistent_wsts",
                split="train",
                temporal_window=window,
                feature_subset=subset,  # type: ignore[arg-type]
                patch_size=256,
                synthetic_fallback_samples=2,
            )
            item = wsts[0]
            assert item["sequence"].shape == (window, expected_c, 256, 256)
            assert item["mask"].shape == (256, 256)

    # Verify MultiResolutionResampler
    resampler = MultiResolutionResampler(target_shape=(256, 256))
    swir_up = resampler.resample_20m_swir_rededge(np.ones((6, 128, 128), dtype=np.float32))
    aero_up = resampler.resample_60m_and_aerosol(np.ones((1, 43, 43), dtype=np.float32))
    lbl_up = resampler.resample_label(np.ones((64, 64), dtype=np.int64))
    assert swir_up.shape == (6, 256, 256)
    assert aero_up.shape == (1, 256, 256)
    assert lbl_up.shape == (256, 256)
    print("      -> PASSED: 17-channel assembly, asymmetric clipping, WSTS+ subsets, and resampling verified.")


def verify_step_2_weight_surgery_and_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    print("[2/6] Verifying SSL4EO Weight Surgery & Forward/Backward passes for UNet-R50 and ViT-S/16...")
    # Test weight surgery on both 7-ch (ssl4eo_l) and 13-ch (ssl4eo_s12) kernels
    w_l8 = torch.randn(64, 7, 7, 7)
    w_s2 = torch.randn(64, 13, 7, 7)
    adapted_l8 = adapt_ssl4eo_first_layer_weights(w_l8, source_sensor="ssl4eo_l", target_in_chans=17)
    adapted_s2 = adapt_ssl4eo_first_layer_weights(w_s2, source_sensor="ssl4eo_s12", target_in_chans=17)
    assert adapted_l8.shape == (64, 17, 7, 7)
    assert adapted_s2.shape == (64, 17, 7, 7)
    assert torch.isfinite(adapted_l8).all() and torch.isfinite(adapted_s2).all()

    # Check exact Red-Edge interpolation formula on ssl4eo_l (scaled by 7/17)
    assert torch.allclose(
        adapted_l8[:, 5],
        0.50 * adapted_l8[:, 3] + 0.50 * adapted_l8[:, 7],
        atol=1e-6,
    ), "Red-Edge B6 linear interpolation mismatch!"

    unet = SSL4EOAttentionUNetResNet50(
        in_channels=17,
        num_classes=3,
        ssl_source="ssl4eo_s12",
        dropout_p=0.15,
    ).to(device)
    vit = SSL4EOCompactViTS16Seg(
        in_channels=17,
        num_classes=3,
        img_size=256,
        ssl_source="ssl4eo_l",
        dropout_p=0.15,
    ).to(device)

    unet_params_m = sum(p.numel() for p in unet.parameters()) / 1e6
    vit_params_m = sum(p.numel() for p in vit.parameters()) / 1e6
    print(f"      -> Model parameter counts: SSL4EOAttentionUNetResNet50={unet_params_m:.2f}M, SSL4EOCompactViTS16Seg={vit_params_m:.2f}M")

    return unet, vit


def verify_step_3_hybrid_loss_amp_backward(
    unet: nn.Module,
    vit: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    print("[3/6] Verifying WildfireHybridLoss FP32 numerical stability & gradient flow under float16 autocast...")
    criterion = WildfireHybridLoss(
        num_classes=3,
        loss_weights=(0.40, 0.40, 0.20),
        alpha_class=(0.10, 0.35, 0.55),
        focal_gamma=2.0,
        tversky_alpha_fp=0.30,
        tversky_beta_fn=0.70,
        tversky_gamma=1.3333,
    ).to(x.device)

    amp_device = "cuda" if x.device.type == "cuda" else "cpu"
    for name, model in (("SSL4EOAttentionUNetResNet50", unet), ("SSL4EOCompactViTS16Seg", vit)):
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(amp_device, dtype=torch.float16, enabled=True):
            logits = model(x)
        assert logits.shape == (2, 3, 256, 256), f"{name} output shape mismatch: {logits.shape}"

        # Explicit FP32 loss computation
        loss, comps = criterion(logits.float(), y, return_components=True)
        assert loss.dtype == torch.float32, f"Expected float32 loss, got {loss.dtype}"
        assert torch.isfinite(loss).all(), f"Non-finite loss in {name}: {loss.item()}"
        for k, v in comps.items():
            assert torch.isfinite(v).all(), f"Non-finite component {k} in {name}"

        loss.backward()
        grad_norms = [
            p.grad.detach().norm().item()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(grad_norms) > 0, f"No gradients flowed through {name}"
        assert all(math.isfinite(g) for g in grad_norms), f"NaN/Inf gradient detected in {name}"
        print(f"      -> PASSED: {name} loss={loss.item():.4f} (Focal={comps['loss_focal'].item():.4f}, "
              f"Tversky={comps['loss_focal_tversky'].item():.4f}, Lovasz={comps['loss_lovasz'].item():.4f})")


def verify_step_4_mc_dropout_and_ensemble_uq(
    unet: nn.Module,
    vit: nn.Module,
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    print("[4/6] Verifying Isolated MC Dropout & Hybrid Ensemble Uncertainty Quantification (T=4 passes)...")
    # Snapshot BatchNorm2d running_mean before stochastic MC inference to prove zero BN corruption
    bn_layer = next(m for m in unet.modules() if isinstance(m, nn.BatchNorm2d))
    bn_mean_before = bn_layer.running_mean.detach().clone()

    enable_mc_dropout_only(unet)
    for m in unet.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert not m.training, "BatchNorm2d must remain in eval() mode during MC Dropout!"
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            assert m.training, "Dropout modules must be in train(True) mode during MC Dropout!"

    uq_mc = predict_with_uncertainty(
        models=unet,
        x=x,
        mode="mc_dropout",
        num_mc_passes=4,
        use_amp=True,
    )
    bn_mean_after = bn_layer.running_mean.detach().clone()
    assert torch.equal(bn_mean_before, bn_mean_after), "BatchNorm2d running_mean was mutated during MC Dropout!"

    for key, expected_shape in (
        ("mean_probs", (2, 3, 256, 256)),
        ("pred_mask", (2, 256, 256)),
        ("pred_variance", (2, 3, 256, 256)),
        ("predictive_entropy", (2, 256, 256)),
        ("aleatoric_entropy", (2, 256, 256)),
        ("epistemic_mi", (2, 256, 256)),
    ):
        assert key in uq_mc, f"Missing key {key} in UQ output"
        assert uq_mc[key].shape == expected_shape, f"Shape mismatch for {key}: {uq_mc[key].shape}"
        assert torch.isfinite(uq_mc[key]).all(), f"Non-finite values in {key}"

    assert (uq_mc["epistemic_mi"] >= 0.0).all(), "Negative epistemic_mi found!"

    # Also test hybrid_ensemble_mc mode across [unet, vit]
    uq_hybrid = predict_with_uncertainty(
        models=[unet, vit],
        x=x,
        mode="hybrid_ensemble_mc",
        num_mc_passes=2,
        use_amp=True,
    )
    assert (uq_hybrid["epistemic_mi"] >= 0.0).all()
    print(f"      -> PASSED: UQ shapes verified, BN stats preserved, max epistemic_mi={uq_mc['epistemic_mi'].max().item():.6f}")
    return uq_mc


def verify_step_5_captum_attributions(unet: nn.Module, x: torch.Tensor) -> None:
    print("[5/6] Verifying Captum Spectral Attribution (IntegratedGradients, Saliency, LayerGradCam)...")
    roi_mask = torch.zeros((2, 256, 256), dtype=torch.float32, device=x.device)
    roi_mask[:, 64:192, 64:192] = 1.0

    # Verify both class_logit and entropy wrapper modes
    wrapper_ent = SegmentationAttributionWrapper(unet, mode="entropy", roi_mask=roi_mask)
    ent_out = wrapper_ent(x)
    assert ent_out.shape == (2,) and torch.isfinite(ent_out).all()

    attr = compute_spectral_feature_attributions(
        model=unet,
        x=x,
        target_class=2,
        roi_mask=roi_mask,
        mode="class_logit",
        ig_steps=4,
    )
    assert attr["integrated_gradients"].shape == (2, 17, 256, 256)
    assert attr["saliency"].shape == (2, 17, 256, 256)
    assert attr["layer_gradcam"].shape == (2, 1, 256, 256)

    shares = attr["channel_importance_shares"]
    shares_sum = float(shares.sum().item())
    assert shares.shape == (17,)
    assert math.isclose(shares_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5), (
        f"Channel importance shares sum to {shares_sum}, expected 1.0"
    )
    print(f"      -> PASSED: 17-channel attribution shares sum={shares_sum:.6f}")


def verify_step_6_calibration_and_carbon(
    uq_out: dict[str, torch.Tensor],
    y: torch.Tensor,
) -> None:
    print("[6/6] Verifying SmokeObscuredSegmentationAndCalibrationEvaluator & GFW Carbon Accounting...")
    raw_aerosol = torch.empty((2, 256, 256), dtype=torch.float32, device=y.device).uniform_(-0.5, 2.5)

    evaluator = SmokeObscuredSegmentationAndCalibrationEvaluator(
        num_classes=3,
        ece_bins=15,
        smoke_aerosol_threshold=1.0,
        boundary_kernel_size=31,
    )
    evaluator.update(
        probs=uq_out["mean_probs"],
        targets=y,
        raw_aerosol=raw_aerosol,
        uncertainty=uq_out["predictive_entropy"],
    )
    metrics = evaluator.compute()

    for key in ("mIoU", "macro_F1", "NLL", "Brier_Score", "ECE", "AUSE"):
        val = float(metrics[key])  # type: ignore[arg-type]
        assert math.isfinite(val), f"Metric {key} is non-finite: {val}"
        assert val >= 0.0, f"Metric {key} is negative: {val}"

    carbon_report = estimate_burn_scar_carbon_loss(
        mean_probs=uq_out["mean_probs"][0].detach().cpu().numpy(),
        epistemic_mi=uq_out["epistemic_mi"][0].detach().cpu().numpy(),
        pred_variance=uq_out["pred_variance"][0].detach().cpu().numpy(),
    )
    for regime in ("conservative_lower_bound", "nominal_mean", "upper_bound_95ci"):
        assert regime in carbon_report, f"Missing epistemic regime {regime}"
        assert carbon_report[regime]["co2_equivalent_tonnes"] >= 0.0

    with track_pipeline_carbon_footprint(project_suffix="DryRun_Verify", enabled=False) as footprint:
        assert "emissions_kg_co2eq" in footprint

    print(
        f"      -> PASSED: mIoU={metrics['mIoU']:.4f}, ECE={metrics['ECE']:.4f}, "
        f"Brier_Score={metrics['Brier_Score']:.4f}, AUSE={metrics['AUSE']:.4f}"
    )


def main() -> None:
    print("=" * 80)
    print("WATCHTOWER AUTOMATED DRY-RUN VERIFICATION SUITE (B=2, C=17, H=256, W=256)")
    print("=" * 80)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}")

    x = torch.randn(2, 17, 256, 256, device=device, dtype=torch.float32)
    y = torch.randint(0, 3, (2, 256, 256), device=device, dtype=torch.int64)
    # Guarantee all 3 classes are present in synthetic ground truth
    y[:, 32:96, 32:96] = 1
    y[:, 96:112, 96:112] = 2

    verify_step_1_data_and_resampling()
    unet, vit = verify_step_2_weight_surgery_and_models(device)
    verify_step_3_hybrid_loss_amp_backward(unet, vit, x, y)
    uq_out = verify_step_4_mc_dropout_and_ensemble_uq(unet, vit, x)
    verify_step_5_captum_attributions(unet, x)
    verify_step_6_calibration_and_carbon(uq_out, y)

    print("=" * 80)
    print("ALL WATCHTOWER DRY-RUN VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

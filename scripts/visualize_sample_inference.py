"""Visualization Demo: End-to-End Inference, Uncertainty, Attribution & Carbon Accounting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.attribution.captum_explainer import compute_spectral_feature_attributions
from src.carbon.gfw_carbon_accounting import estimate_burn_scar_carbon_loss
from src.data.sen2fire_dataset import SEN2FIRE_BAND_NAMES, Sen2FireMultispectralDataset
from src.models.attention_unet_r50 import SSL4EOAttentionUNetResNet50
from src.uncertainty.mc_ensemble_uq import predict_with_uncertainty


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / "inference_visualization_demo.png"

    print("[Watchtower Demo] 1. Generating synthetic multispectral sample...")
    ds = Sen2FireMultispectralDataset(
        root_dir="data/sen2fire",
        synthetic_fallback_samples=1,
        patch_size=256,
        is_train=False,
    )
    sample = ds[0]
    x_tensor = sample["image"].unsqueeze(0)  # (1, 17, 256, 256)
    y_true = sample["mask"].numpy()          # (256, 256)

    print("[Watchtower Demo] 2. Loading SSL4EOAttentionUNetResNet50...")
    model = SSL4EOAttentionUNetResNet50(in_channels=17, num_classes=3, ssl_source="ssl4eo_s12")

    print("[Watchtower Demo] 3. Running Isolated MC Dropout Inference (T=6 passes)...")
    uq_out = predict_with_uncertainty(model, x_tensor, mode="mc_dropout", num_mc_passes=6, use_amp=False)
    pred_mask = uq_out["pred_mask"][0].numpy()
    pred_entropy = uq_out["predictive_entropy"][0].numpy()
    epistemic_mi = uq_out["epistemic_mi"][0].numpy()
    mean_probs = uq_out["mean_probs"][0].numpy()

    print("[Watchtower Demo] 4. Computing Captum Spectral Attributions (Integrated Gradients & LayerGradCam)...")
    attr_out = compute_spectral_feature_attributions(model, x_tensor, target_class=2, ig_steps=4)
    gradcam = attr_out["layer_gradcam"][0, 0].numpy()
    channel_shares = attr_out["channel_importance_shares"].cpu().numpy()

    print("[Watchtower Demo] 5. Quantifying GFW Carbon & CO2 Loss Across 3 Epistemic Regimes...")
    carbon_report = estimate_burn_scar_carbon_loss(
        mean_probs=mean_probs,
        epistemic_mi=epistemic_mi,
        pixel_resolution_m=10.0,
    )

    print("[Watchtower Demo] 6. Rendering 8-panel diagnostic figure to:", out_img_path)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Watchtower: Uncertainty-Aware Wildfire Segmentation & Carbon Accounting", fontsize=16, fontweight="bold")

    # Panel 1: False Color Composite (SWIR2 [B12], NIR [B8], Red [B4])
    # Denormalize roughly for visual display
    swir = np.clip(sample["image"][11].numpy() * 0.165 + 0.155, 0, 1)
    nir = np.clip(sample["image"][7].numpy() * 0.122 + 0.258, 0, 1)
    red = np.clip(sample["image"][3].numpy() * 0.088 + 0.112, 0, 1)
    false_color = np.stack([swir, nir, red], axis=-1)

    axes[0, 0].imshow(false_color)
    axes[0, 0].set_title("False Color (SWIR-NIR-Red)")
    axes[0, 0].axis("off")

    # Panel 2: Ground Truth Refined Mask
    cmap_mask = matplotlib.colors.ListedColormap(["#2b2b2b", "#d95f02", "#e41a1c"])
    axes[0, 1].imshow(y_true, cmap=cmap_mask, vmin=0, vmax=2)
    axes[0, 1].set_title("Ground Truth Mask (Refined)")
    axes[0, 1].axis("off")

    # Panel 3: Predicted Mask
    axes[0, 2].imshow(pred_mask, cmap=cmap_mask, vmin=0, vmax=2)
    axes[0, 2].set_title("Predicted Mask (MAP)")
    axes[0, 2].axis("off")

    # Panel 4: Predictive Total Entropy
    im_ent = axes[0, 3].imshow(pred_entropy, cmap="inferno")
    axes[0, 3].set_title("Total Predictive Entropy H[E[p]]")
    axes[0, 3].axis("off")
    plt.colorbar(im_ent, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Panel 5: Epistemic Mutual Information (BALD)
    im_mi = axes[1, 0].imshow(epistemic_mi, cmap="magma")
    axes[1, 0].set_title("Epistemic Uncertainty (BALD MI)")
    axes[1, 0].axis("off")
    plt.colorbar(im_mi, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 6: LayerGradCam Spatial Attribution
    im_cam = axes[1, 1].imshow(gradcam, cmap="plasma")
    axes[1, 1].set_title("LayerGradCam (Active Flame Front)")
    axes[1, 1].axis("off")
    plt.colorbar(im_cam, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Panel 7: 17-Channel Spectral Attribution Shares
    y_pos = np.arange(len(SEN2FIRE_BAND_NAMES))
    axes[1, 2].barh(y_pos, channel_shares, color="#1f77b4", align="center")
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(SEN2FIRE_BAND_NAMES, fontsize=7)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("Normalized Importance Share")
    axes[1, 2].set_title("Captum Spectral Attribution (Sum=1.0)")

    # Panel 8: Carbon Accounting Summary Table
    axes[1, 3].axis("off")
    table_data = [
        ["Regime", "Burned (ha)", "Carbon (tC)", "CO2e (tonnes)"],
        [
            "Conservative",
            f"{carbon_report['conservative_lower_bound']['burned_area_ha']:.1f}",
            f"{carbon_report['conservative_lower_bound']['total_carbon_loss_tc']:.1f}",
            f"{carbon_report['conservative_lower_bound']['co2_equivalent_tonnes']:.1f}",
        ],
        [
            "Nominal Mean",
            f"{carbon_report['nominal_mean']['burned_area_ha']:.1f}",
            f"{carbon_report['nominal_mean']['total_carbon_loss_tc']:.1f}",
            f"{carbon_report['nominal_mean']['co2_equivalent_tonnes']:.1f}",
        ],
        [
            "Upper 95% CI",
            f"{carbon_report['upper_bound_95ci']['burned_area_ha']:.1f}",
            f"{carbon_report['upper_bound_95ci']['total_carbon_loss_tc']:.1f}",
            f"{carbon_report['upper_bound_95ci']['co2_equivalent_tonnes']:.1f}",
        ],
    ]
    table = axes[1, 3].table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.22, 0.24, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    axes[1, 3].set_title("GFW Carbon Accounting", pad=20)

    plt.tight_layout()
    plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("[Watchtower Demo] Completed! Diagnostic figure successfully generated at:", out_img_path)
    print("\n--- CARBON ACCOUNTING METRICS REPORT ---")
    print(json.dumps(carbon_report, indent=2))


if __name__ == "__main__":
    main()

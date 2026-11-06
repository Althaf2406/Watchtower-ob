"""Real-World Validation Script: Testing Watchtower on the September 2019 Central Kalimantan Wildfire Crisis."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch

from src.attribution.captum_explainer import compute_spectral_feature_attributions
from src.carbon.gfw_carbon_accounting import estimate_burn_scar_carbon_loss
from src.data.resampling_eolearn import resample_band_array
from src.data.sen2fire_dataset import SEN2FIRE_BAND_NAMES, Sen2FireMultispectralDataset
from src.models.attention_unet_r50 import SSL4EOAttentionUNetResNet50
from src.uncertainty.mc_ensemble_uq import predict_with_uncertainty

# Sentinel-2 L2A tile 49MHT (Central Kalimantan - Pulang Pisau / Sebangau Peatland)
# Date: September 21, 2019 (Peak disaster level, choking smoke haze)
BASE_COG_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/49/M/HT/2019/9/S2B_49MHT_20190921_1_L2A"

BAND_FILENAMES = {
    0: "B01.tif",   # Coastal (60m)
    1: "B02.tif",   # Blue (10m)
    2: "B03.tif",   # Green (10m)
    3: "B04.tif",   # Red (10m)
    4: "B05.tif",   # Red-Edge 1 (20m)
    5: "B06.tif",   # Red-Edge 2 (20m)
    6: "B07.tif",   # Red-Edge 3 (20m)
    7: "B08.tif",   # NIR (10m)
    8: "B8A.tif",   # Narrow NIR (20m)
    9: "B09.tif",   # Water Vapour (60m)
    10: "B11.tif",  # SWIR 1 (20m)
    11: "B12.tif",  # SWIR 2 (20m)
}


def fetch_real_kalimantan_patch(
    center_col_20m: int = 1540,
    center_row_20m: int = 200,
    patch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], rasterio.transform.Affine]:
    """Fetch a real multispectral 256x256 patch directly from Sentinel-2 Cloud-Optimized GeoTIFFs on AWS S3.

    Coordinates target the active peat burning front in Pulang Pisau / Sebangau, Central Kalimantan.
    """
    print(f"[Kalimantan Data] Fetching real Sentinel-2 L2A scene (Date: 2019-09-21, Tile: 49MHT)...")
    half_size = patch_size // 2

    # 10m bands have resolution = 10980 x 10980 (double the 20m grid)
    col_10m = center_col_20m * 2
    row_10m = center_row_20m * 2
    win_10m = Window(col_10m - half_size, row_10m - half_size, patch_size, patch_size)

    # 20m bands have resolution = 5490 x 5490 (half of 10m grid: window is 128x128)
    win_20m = Window(center_col_20m - half_size // 2, center_row_20m - half_size // 2, patch_size // 2, patch_size // 2)

    # 60m bands have resolution = 1830 x 1830 (window is 43x43)
    col_60m = center_col_20m // 3
    row_60m = center_row_20m // 3
    win_60m = Window(col_60m - half_size // 6, row_60m - half_size // 6, patch_size // 6 + 1, patch_size // 6 + 1)

    s2_12bands = np.zeros((12, patch_size, patch_size), dtype=np.float32)
    affine_transform = None

    for b_idx in range(12):
        fname = BAND_FILENAMES[b_idx]
        url = f"{BASE_COG_URL}/{fname}"
        with rasterio.open(url) as src:
            if affine_transform is None and src.width == 10980:
                # Capture geo-transform for 10m window
                affine_transform = rasterio.windows.transform(win_10m, src.transform)

            if src.width == 10980:
                band_raw = src.read(1, window=win_10m, out_shape=(patch_size, patch_size), resampling=Resampling.bilinear).astype(np.float32)
                s2_12bands[b_idx] = band_raw
            elif src.width == 5490:
                band_raw = src.read(1, window=win_20m, out_shape=(patch_size, patch_size), resampling=Resampling.cubic).astype(np.float32)
                s2_12bands[b_idx] = band_raw
            else:
                band_raw = src.read(1, window=win_60m, out_shape=(patch_size, patch_size), resampling=Resampling.bilinear).astype(np.float32)
                s2_12bands[b_idx] = band_raw
        print(f"  -> Fetched Band {b_idx} ({fname}): min={s2_12bands[b_idx].min():.0f}, max={s2_12bands[b_idx].max():.0f}")

    # Fetch Real Aerosol (AOT.tif) as proxy for Sentinel-5P aerosol index
    with rasterio.open(f"{BASE_COG_URL}/AOT.tif") as src:
        aot_raw = src.read(1, window=win_20m, out_shape=(patch_size, patch_size), resampling=Resampling.bilinear).astype(np.float32) / 1000.0  # Scale to AOT units
        aerosol_1ch = aot_raw[np.newaxis, :, :]

    # Fetch Real True Color Image (TCI.tif: Visual RGB for human eyes)
    with rasterio.open(f"{BASE_COG_URL}/TCI.tif") as src:
        tci_rgb = src.read([1, 2, 3], window=win_10m, out_shape=(3, patch_size, patch_size), resampling=Resampling.bilinear).astype(np.float32) / 255.0

    # Calculate geographic center (Lat/Lon)
    center_x, center_y = rasterio.transform.xy(affine_transform, patch_size // 2, patch_size // 2)
    transformer = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(center_x, center_y)

    print(f"[Kalimantan Data] Successfully fetched patch centered at Lat={lat:.5f}, Lon={lon:.5f} (Pulang Pisau / Sebangau Peatland)!")
    return s2_12bands, aerosol_1ch, tci_rgb, (lat, lon), affine_transform


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / "kalimantan_2019_real_wildfire_test.png"

    print("=" * 80)
    print("WATCHTOWER REAL WILDFIRE VALIDATION: CENTRAL KALIMANTAN PEATLANDS (2019)")
    print("=" * 80)

    # 1. Fetch real satellite patch from Central Kalimantan (Pulang Pisau/Sebangau)
    t0 = time.time()
    s2_raw, aero_raw, tci_rgb, (lat, lon), transform = fetch_real_kalimantan_patch()
    print(f"Data fetch completed in {time.time() - t0:.1f}s.")

    # 2. Process with Watchtower Pipeline (Asymmetric Clipping, 4 Spectral Indices, Label Refinement)
    dataset = Sen2FireMultispectralDataset(
        root_dir="data/sen2fire",
        stats_json_path="configs/sen2fire_train_stats.json",
        synthetic_fallback_samples=1,
    )

    # Create dummy coarse positive from SWIR threshold to mimic coarse MODIS hotspot
    coarse_fire = (s2_raw[11] > 2500).astype(np.uint8)

    features_17ch, refined_mask, clipped_aero = dataset.process_raw_arrays(s2_raw, aero_raw, coarse_fire)

    # Standardize 17 channels with zero-leakage training statistics
    standardized = (features_17ch - dataset.mean) / dataset.std
    x_tensor = torch.from_numpy(standardized).unsqueeze(0).float()  # (1, 17, 256, 256)

    # 3. Model Inference with Isolated MC Dropout (T=10 passes)
    print("\n[Inference] Running Watchtower Attention U-Net with Isolated MC Dropout (T=10 passes)...")
    model = SSL4EOAttentionUNetResNet50(in_channels=17, num_classes=3, ssl_source="ssl4eo_s12")

    uq_out = predict_with_uncertainty(model, x_tensor, mode="mc_dropout", num_mc_passes=10, use_amp=False)
    pred_mask = uq_out["pred_mask"][0].numpy()
    pred_entropy = uq_out["predictive_entropy"][0].numpy()
    epistemic_mi = uq_out["epistemic_mi"][0].numpy()
    mean_probs = uq_out["mean_probs"][0].numpy()

    # 4. Captum Spectral Attribution
    print("[Explainable AI] Computing Captum Spectral Attributions (Integrated Gradients & LayerGradCam)...")
    attr_out = compute_spectral_feature_attributions(model, x_tensor, target_class=2, ig_steps=6)
    gradcam = attr_out["layer_gradcam"][0, 0].numpy()
    channel_shares = attr_out["channel_importance_shares"].cpu().numpy()

    # 5. Global Forest Watch Carbon Accounting
    print("[Carbon Accounting] Calculating Peat & Biomass Carbon Emissions across 3 Epistemic Regimes...")
    carbon_report = estimate_burn_scar_carbon_loss(
        mean_probs=mean_probs,
        epistemic_mi=epistemic_mi,
        transform=transform,
        crs="EPSG:32749",
        pixel_resolution_m=10.0,
    )

    # 6. Render Comprehensive Diagnostic Figure
    print(f"[Visualization] Saving real-world diagnostic figure to: {out_img_path}")
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 4)

    fig.suptitle(
        f"Watchtower Real-World Validation: Central Kalimantan Peat Fires (Sept 21, 2019)\n"
        f"Location: Pulang Pisau / Sebangau (Lat: {lat:.4f}°, Lon: {lon:.4f}°) | Sentinel-2 L2A (10m Grid)",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: True Color (TCI RGB - What human eye sees through the plane/camera)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.transpose(tci_rgb, (1, 2, 0)))
    ax1.set_title("1. True Color RGB (Mata Manusia)\nPenuh Asap Pekat Kuning/Cokelat", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: False Color (SWIR2 [B12] - NIR [B8] - Red [B4]: Tembus Asap!)
    swir2_norm = np.clip(features_17ch[11] / 1.5, 0, 1)
    nir_norm = np.clip(features_17ch[7] / 0.8, 0, 1)
    red_norm = np.clip(features_17ch[3] / 0.6, 0, 1)
    false_color = np.stack([swir2_norm, nir_norm, red_norm], axis=-1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(false_color)
    ax2.set_title("2. False Color (SWIR-NIR-Red)\nMenembus Asap! Api Terlihat Menyala", fontsize=11, fontweight="bold", color="darkred")
    ax2.axis("off")

    # Panel 3: Refined Physical Ground Truth Mask
    cmap_mask = matplotlib.colors.ListedColormap(["#222222", "#d95f02", "#e41a1c"])
    ax3 = fig.add_subplot(gs[0, 2])
    im_ref = ax3.imshow(refined_mask, cmap=cmap_mask, vmin=0, vmax=2)
    ax3.set_title("3. Physical Ground Truth Mask\n(Hitam: Asap/Latar, Oranye: Burn Scar, Merah: Api)", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Panel 4: Watchtower Predicted Mask (MAP)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(pred_mask, cmap=cmap_mask, vmin=0, vmax=2)
    ax4.set_title("4. Watchtower Prediction (MAP)\nDeteksi Segmentasi Semantik AI", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # Panel 5: Total Predictive Entropy
    ax5 = fig.add_subplot(gs[1, 0])
    im_ent = ax5.imshow(pred_entropy, cmap="inferno")
    ax5.set_title("5. Total Predictive Entropy\n(Keraguan AI di Batas Penjalaran Api)", fontsize=11, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im_ent, ax=ax5, fraction=0.046, pad=0.04)

    # Panel 6: Epistemic Uncertainty (BALD MI)
    ax6 = fig.add_subplot(gs[1, 1])
    im_mi = ax6.imshow(epistemic_mi, cmap="magma")
    ax6.set_title("6. Epistemic Uncertainty (BALD MI)\n(Mendeteksi Tutupan Asap Ekstrem)", fontsize=11, fontweight="bold", color="purple")
    ax6.axis("off")
    plt.colorbar(im_mi, ax=ax6, fraction=0.046, pad=0.04)

    # Panel 7: Captum 17-Channel Importance Shares
    ax7 = fig.add_subplot(gs[1, 2])
    y_pos = np.arange(len(SEN2FIRE_BAND_NAMES))
    ax7.barh(y_pos, channel_shares, color="#1f77b4", align="center")
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(SEN2FIRE_BAND_NAMES, fontsize=7)
    ax7.invert_yaxis()
    ax7.set_xlabel("Importance Share (Sum=1.0)", fontsize=9)
    ax7.set_title("7. Captum Feature Attribution\nKontribusi Sensor Terhadap Prediksi", fontsize=11, fontweight="bold")

    # Panel 8: GFW Carbon Accounting Table
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
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
    table = ax8.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.22, 0.24, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    ax8.set_title("8. GFW Carbon Accounting\nEstimasi Emisi Lahan Gambut Kalimantan", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Print summary metrics to console
    print("\n" + "=" * 80)
    print("HASIL VALIDASI REAL DATA KALIMANTAN (PULANG PISAU - 21 SEPT 2019):")
    print("=" * 80)
    print(f"Koordinat Geografis  : Latitude {lat:.5f}°, Longitude {lon:.5f}°")
    print(f"Luas Area Sampel     : 256 piksel x 256 piksel = 655.36 Hektar")
    print(f"Piksel Api Terdeteksi: {int((refined_mask > 0).sum())} piksel ({((refined_mask > 0).sum() * 0.01):.2f} ha)")
    print(f"Piksel Api Aktif     : {int((refined_mask == 2).sum())} piksel ({((refined_mask == 2).sum() * 0.01):.2f} ha)")
    print("\nLaporan Emisi Karbon Gambut (GFW):")
    print(f"  • Conservative Lower Bound : {carbon_report['conservative_lower_bound']['co2_equivalent_tonnes']:.1f} Ton CO2e")
    print(f"  • Nominal Mean             : {carbon_report['nominal_mean']['co2_equivalent_tonnes']:.1f} Ton CO2e")
    print(f"  • Upper Bound 95% CI       : {carbon_report['upper_bound_95ci']['co2_equivalent_tonnes']:.1f} Ton CO2e")
    print("=" * 80)
    print(f"Gambar diagnostik berhasil disimpan di: {out_img_path}")


if __name__ == "__main__":
    main()

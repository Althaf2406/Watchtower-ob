"""Cross-Sensor First-Layer Weight Surgery for SSL4EO-L (Landsat 7-ch) & SSL4EO-S12 (Sentinel-2 13-ch)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch

SSLSourceType = Literal["ssl4eo_l", "ssl4eo_s12"]


def adapt_ssl4eo_first_layer_weights(
    pretrained_weight: torch.Tensor,
    source_sensor: str = "ssl4eo_s12",
    target_in_chans: int = 17,
) -> torch.Tensor:
    """Adapt pretrained SSL4EO first-layer conv/patch-embed kernels `(Out, C_pre, kH, kW)` to `target_in_chans=17`.

    Target 17-channel layout:
      - 0: B1 Coastal (443nm)
      - 1: B2 Blue (490nm)
      - 2: B3 Green (560nm)
      - 3: B4 Red (665nm)
      - 4: B5 Red-Edge 1 (705nm)
      - 5: B6 Red-Edge 2 (740nm)
      - 6: B7 Red-Edge 3 (783nm)
      - 7: B8 NIR (842nm)
      - 8: B8A Narrow NIR (865nm)
      - 9: B9 Water Vapour (945nm)
      - 10: B11 SWIR 1 (1610nm)
      - 11: B12 SWIR 2 (2190nm)
      - 12: B13 S5P UV Aerosol Index
      - 13: NBR     = (B8 - B12) / (B8 + B12)
      - 14: AF_SWIR = (B12 - B8A) / (B12 + B8A)
      - 15: NDVI    = (B8 - B4) / (B8 + B4)
      - 16: NDWI    = (B3 - B8) / (B3 + B8)

    Weight surgery rules:
      1. Map homologous physical bands from `ssl4eo_l` (7-ch Landsat OLI SR) or `ssl4eo_s12` (13-ch Sentinel-2 L1C).
      2. Linearly interpolate Red (B4) and NIR (B8) weights for Red-Edge (B5, B6, B7) when absent (`ssl4eo_l` or RGB).
      3. Initialize B13 (UV Aerosol) from short-wave bands `(B1 + B2) / 2`.
      4. Initialize the 4 spectral index channels from symmetric band differences:
         - NBR:     `(W[B8] - W[B12]) / 2`
         - AF_SWIR: `(W[B12] - W[B8A]) / 2`
         - NDVI:    `(W[B8] - W[B4]) / 2`
         - NDWI:    `(W[B3] - W[B8]) / 2`
      5. Apply variance-preserving scaling factor `C_pre / C_target` so activation variance remains invariant.
    """
    if pretrained_weight.ndim != 4:
        raise ValueError(
            f"Expected 4D conv kernel `(Out, C_pre, kH, kW)`, got shape {tuple(pretrained_weight.shape)}"
        )

    out_ch, c_pre, kh, kw = pretrained_weight.shape
    sensor = source_sensor.lower().strip()
    if sensor not in ("ssl4eo_l", "ssl4eo_s12"):
        raise ValueError(f"Unsupported `source_sensor` {source_sensor!r}. Expected 'ssl4eo_l' or 'ssl4eo_s12'.")

    w_src = pretrained_weight.detach().clone().to(dtype=torch.float32)
    w_new = torch.zeros((out_ch, target_in_chans, kh, kw), dtype=torch.float32, device=w_src.device)

    if sensor == "ssl4eo_s12" and c_pre >= 13:
        # TorchGeo SSL4EO-S12 13-band order:
        # [0:B1, 1:B2, 2:B3, 3:B4, 4:B5, 5:B6, 6:B7, 7:B8, 8:B8A, 9:B9, 10:B10_Cirrus, 11:B11, 12:B12]
        w_new[:, 0:10] = w_src[:, 0:10]  # B1..B9
        w_new[:, 10] = w_src[:, 11]      # B11 (SWIR 1)
        w_new[:, 11] = w_src[:, 12]      # B12 (SWIR 2)
    elif sensor == "ssl4eo_l" and c_pre >= 7:
        # TorchGeo SSL4EO-L 7-band order (Landsat OLI SR):
        # [0:B1_Coastal, 1:B2_Blue, 2:B3_Green, 3:B4_Red, 4:B5_NIR, 5:B6_SWIR1, 6:B7_SWIR2]
        w_new[:, 0] = w_src[:, 0]        # B1 Coastal
        w_new[:, 1] = w_src[:, 1]        # B2 Blue
        w_new[:, 2] = w_src[:, 2]        # B3 Green
        w_new[:, 3] = w_src[:, 3]        # B4 Red
        w_new[:, 7] = w_src[:, 4]        # B8 NIR
        w_new[:, 8] = w_src[:, 4]        # B8A Narrow NIR (from Landsat NIR 865nm)
        w_new[:, 9] = 0.5 * (w_src[:, 4] + w_src[:, 5])  # B9 Water Vapour (945nm)
        w_new[:, 10] = w_src[:, 5]       # B11 SWIR 1
        w_new[:, 11] = w_src[:, 6]       # B12 SWIR 2

        # Linearly interpolate Red (idx 3) and NIR (idx 7) for Red-Edge B5 (idx 4), B6 (idx 5), B7 (idx 6)
        w_red = w_new[:, 3]
        w_nir = w_new[:, 7]
        w_new[:, 4] = 0.75 * w_red + 0.25 * w_nir  # B5 (705nm)
        w_new[:, 5] = 0.50 * w_red + 0.50 * w_nir  # B6 (740nm)
        w_new[:, 6] = 0.25 * w_red + 0.75 * w_nir  # B7 (783nm)
    elif c_pre == 3:
        # 3-channel RGB fallback (ImageNet / synthetic 3-ch): [0:Red, 1:Green, 2:Blue]
        w_red = w_src[:, 0]
        w_green = w_src[:, 1]
        w_blue = w_src[:, 2]
        w_mean = w_src.mean(dim=1)

        w_new[:, 0] = w_blue
        w_new[:, 1] = w_blue
        w_new[:, 2] = w_green
        w_new[:, 3] = w_red
        w_new[:, 7] = w_mean
        w_new[:, 8] = w_mean
        w_new[:, 9] = w_mean
        w_new[:, 10] = w_red
        w_new[:, 11] = w_red
        w_new[:, 4] = 0.75 * w_red + 0.25 * w_new[:, 7]
        w_new[:, 5] = 0.50 * w_red + 0.50 * w_new[:, 7]
        w_new[:, 6] = 0.25 * w_red + 0.75 * w_new[:, 7]
    else:
        # Generic channel mapping for arbitrary C_pre
        for ch in range(min(12, target_in_chans)):
            w_new[:, ch] = w_src[:, ch % c_pre]
        w_red = w_new[:, 3]
        w_nir = w_new[:, 7]
        w_new[:, 4] = 0.75 * w_red + 0.25 * w_nir
        w_new[:, 5] = 0.50 * w_red + 0.50 * w_nir
        w_new[:, 6] = 0.25 * w_red + 0.75 * w_nir

    # Channel 12 (B13 S5P UV Aerosol): initialize from short-wave bands (B1 + B2) / 2
    if target_in_chans > 12:
        w_new[:, 12] = 0.5 * (w_new[:, 0] + w_new[:, 1])

    # Channels 13..16 (4 Spectral Indices): initialize from symmetric band differences
    if target_in_chans > 13:
        # 13: NBR = (B8 - B12) / (B8 + B12)
        w_new[:, 13] = 0.5 * (w_new[:, 7] - w_new[:, 11])
    if target_in_chans > 14:
        # 14: AF_SWIR = (B12 - B8A) / (B12 + B8A)
        w_new[:, 14] = 0.5 * (w_new[:, 11] - w_new[:, 8])
    if target_in_chans > 15:
        # 15: NDVI = (B8 - B4) / (B8 + B4)
        w_new[:, 15] = 0.5 * (w_new[:, 7] - w_new[:, 3])
    if target_in_chans > 16:
        # 16: NDWI = (B3 - B8) / (B3 + B8)
        w_new[:, 16] = 0.5 * (w_new[:, 2] - w_new[:, 7])

    # Any additional channels beyond 17
    for extra_ch in range(17, target_in_chans):
        w_new[:, extra_ch] = w_src.mean(dim=1)

    # Apply variance-preserving scaling: C_pre / C_target
    scaling_factor = float(c_pre) / float(target_in_chans)
    w_new = w_new * scaling_factor

    return w_new.to(dtype=pretrained_weight.dtype)


def load_ssl4eo_pretrained_weights(
    architecture: Literal["resnet50", "vit_small_patch16_224"],
    ssl_source: SSLSourceType = "ssl4eo_s12",
    download: bool = False,
) -> Optional[Dict[str, Any]]:
    """Attempt to resolve TorchGeo SSL4EO MoCo state_dict (`LANDSAT_OLI_SR_MOCO` or `SENTINEL2_ALL_MOCO`).

    Returns `None` gracefully when running offline / in dry-run mode or when TorchGeo weights are not cached.
    """
    if not download:
        return None

    try:
        if architecture == "resnet50":
            from torchgeo.models import ResNet50_Weights

            weight_enum = (
                ResNet50_Weights.LANDSAT_OLI_SR_MOCO
                if ssl_source == "ssl4eo_l"
                else ResNet50_Weights.SENTINEL2_ALL_MOCO
            )
            return weight_enum.get_state_dict(progress=False)
        elif architecture == "vit_small_patch16_224":
            from torchgeo.models import ViTSmall16_Weights

            weight_enum = (
                ViTSmall16_Weights.LANDSAT_OLI_SR_MOCO
                if ssl_source == "ssl4eo_l"
                else ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            )
            return weight_enum.get_state_dict(progress=False)
    except Exception:
        return None
    return None


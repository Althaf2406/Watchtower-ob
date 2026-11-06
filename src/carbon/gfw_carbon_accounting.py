"""Uncertainty-Propagated GFW Carbon Accounting (AGB & SOC) and CodeCarbon Compute Footprint Tracker."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine, from_bounds, from_origin
from rasterio.warp import reproject
from shapely.geometry import shape

EPISTEMIC_REGIMES: Tuple[str, str, str] = (
    "conservative_lower_bound",
    "nominal_mean",
    "upper_bound_95ci",
)


def _align_biomass_raster(
    raster_source: Optional[Union[str, Path, np.ndarray]],
    target_shape: Tuple[int, int],
    dst_transform: Affine,
    dst_crs: Union[str, rasterio.crs.CRS],
    default_value: float,
) -> np.ndarray:
    """Load or reproject a GFW AGB (`Mg/ha`) or SOC (`tC/ha`) raster onto the prediction spatial grid."""
    h, w = int(target_shape[0]), int(target_shape[1])
    if raster_source is None:
        return np.full((h, w), float(default_value), dtype=np.float32)

    if isinstance(raster_source, np.ndarray):
        if raster_source.shape == (h, w):
            return raster_source.astype(np.float32)
        dst = np.zeros((h, w), dtype=np.float32)
        src_transform = from_bounds(0.0, 0.0, float(w), float(h), width=raster_source.shape[-1], height=raster_source.shape[-2])
        reproject(
            source=raster_source.astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=dst_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        return dst

    raster_path = Path(raster_source)
    if not raster_path.is_file():
        return np.full((h, w), float(default_value), dtype=np.float32)

    with rasterio.open(raster_path) as src:
        dst = np.zeros((h, w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs if src.crs is not None else dst_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)


def _vectorize_mask_to_gdf(
    binary_mask: np.ndarray,
    transform: Affine,
    crs: Union[str, rasterio.crs.CRS],
) -> gpd.GeoDataFrame:
    """Vectorize a 2D binary burn mask into a GeoDataFrame using `rasterio.features.shapes`."""
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    geoms = []
    values = []
    for geom_dict, val in shapes(mask_u8, mask=mask_u8 == 1, transform=transform):
        if int(val) == 1:
            geoms.append(shape(geom_dict))
            values.append(int(val))

    if not geoms:
        return gpd.GeoDataFrame({"burn_class": [], "geometry": []}, crs=crs)
    return gpd.GeoDataFrame({"burn_class": values, "geometry": geoms}, crs=crs)


def estimate_burn_scar_carbon_loss(
    mean_probs: np.ndarray,
    epistemic_mi: Optional[np.ndarray] = None,
    pred_variance: Optional[np.ndarray] = None,
    agb_raster: Optional[Union[str, Path, np.ndarray]] = None,
    soc_raster: Optional[Union[str, Path, np.ndarray]] = None,
    transform: Optional[Affine] = None,
    crs: str = "EPSG:32755",
    pixel_resolution_m: float = 10.0,
    agb_carbon_fraction: float = 0.47,
    burn_scar_combustion_factor: float = 0.45,
    active_flame_combustion_factor: float = 0.65,
    soc_combustion_factor: float = 0.10,
) -> Dict[str, Any]:
    """Estimate wildfire carbon & CO2e emissions across 3 epistemic confidence regimes using GFW AGB & SOC.

    Regimes evaluated:
      1. `conservative_lower_bound`: High-confidence core burn pixels (`p_fire - 1.96 * sigma >= 0.50`
         and low epistemic mutual information).
      2. `nominal_mean`: MAP semantic prediction (`Class 1: Burn Scar` or `Class 2: Active Flame Front`).
      3. `upper_bound_95ci`: Upper 95% confidence envelope (`p_fire + 1.96 * sigma >= 0.50`).

    Args:
        mean_probs: Posterior mean probabilities `(C, H, W)` or `(B, C, H, W)`.
        epistemic_mi: Optional BALD mutual information map `(H, W)`.
        pred_variance: Optional posterior variance `(C, H, W)` or `(H, W)`.
        agb_raster: Path or array for Aboveground Live Woody Biomass (`Mg/ha`). Defaults to 145.0 Mg/ha.
        soc_raster: Path or array for Soil Organic Carbon (`tC/ha`). Defaults to 65.0 tC/ha.
        transform: Affine transform of the prediction grid (defaults to 10m UTM grid).
        crs: Coordinate reference system string.
        pixel_resolution_m: Native pixel resolution in meters (10.0m for Sentinel-2).
        agb_carbon_fraction: IPCC biomass-to-carbon conversion factor (0.47 tC / Mg AGB).
        burn_scar_combustion_factor: Fraction of AGB combusted in Class 1 Burn Scars.
        active_flame_combustion_factor: Fraction of AGB combusted in Class 2 Active Flame Fronts.
        soc_combustion_factor: Fraction of surface SOC combusted.

    Returns:
        Dictionary keyed by regime (`conservative_lower_bound`, `nominal_mean`, `upper_bound_95ci`)
        plus summary metadata and polygon counts.
    """
    probs = np.asarray(mean_probs, dtype=np.float32)
    if probs.ndim == 4:
        probs = probs[0]
    if probs.shape[0] != 3:
        raise ValueError(f"Expected 3-class probabilities `(3, H, W)`, got shape {probs.shape}")

    _, h, w = probs.shape
    if transform is None:
        transform = from_bounds(
            0.0,
            0.0,
            float(w) * pixel_resolution_m,
            float(h) * pixel_resolution_m,
            width=w,
            height=h,
        )

    # Align Global Forest Watch AGB (Mg/ha) and SOC (tC/ha) rasters via rasterio.warp.reproject
    agb_grid = _align_biomass_raster(agb_raster, (h, w), transform, crs, default_value=145.0)
    soc_grid = _align_biomass_raster(soc_raster, (h, w), transform, crs, default_value=65.0)

    p_burn = probs[1]
    p_flame = probs[2]
    p_fire_total = np.clip(p_burn + p_flame, 0.0, 1.0)
    pred_class = np.argmax(probs, axis=0)

    if pred_variance is not None:
        var_arr = np.asarray(pred_variance, dtype=np.float32)
        if var_arr.ndim == 4:
            var_arr = var_arr[0]
        sigma_fire = np.sqrt(np.clip(var_arr[1] + var_arr[2] if var_arr.ndim == 3 else var_arr, 0.0, None))
    elif epistemic_mi is not None:
        mi_arr = np.asarray(epistemic_mi, dtype=np.float32)
        if mi_arr.ndim == 3:
            mi_arr = mi_arr[0]
        sigma_fire = np.sqrt(np.clip(mi_arr, 0.0, None)) * 0.35
    else:
        sigma_fire = np.sqrt(np.clip(p_fire_total * (1.0 - p_fire_total), 0.0, None)) * 0.25

    regime_masks: Dict[str, np.ndarray] = {
        "conservative_lower_bound": (p_fire_total - 1.96 * sigma_fire) >= 0.50,
        "nominal_mean": (pred_class > 0) | (p_fire_total >= 0.50),
        "upper_bound_95ci": (p_fire_total + 1.96 * sigma_fire) >= 0.50,
    }

    pixel_area_ha = (pixel_resolution_m * pixel_resolution_m) / 10000.0  # 0.01 ha for 10m x 10m pixel
    co2_ratio = 44.0 / 12.0  # 3.6667 tCO2e per tC

    results: Dict[str, Any] = {}
    for regime_name in EPISTEMIC_REGIMES:
        mask = regime_masks[regime_name]
        gdf = _vectorize_mask_to_gdf(mask, transform=transform, crs=crs)

        if not gdf.empty and gdf.crs is not None:
            vector_area_ha = float(gdf.geometry.area.sum() / 10000.0)
            num_polygons = int(len(gdf))
        else:
            vector_area_ha = float(mask.sum() * pixel_area_ha)
            num_polygons = 0

        is_flame = mask & (p_flame >= p_burn)
        is_scar = mask & (~is_flame)

        # Carbon density per pixel (tC/ha)
        agb_c_density = agb_grid * agb_carbon_fraction
        agb_loss_scar_tc = float(np.sum(agb_c_density[is_scar] * burn_scar_combustion_factor * pixel_area_ha))
        agb_loss_flame_tc = float(np.sum(agb_c_density[is_flame] * active_flame_combustion_factor * pixel_area_ha))
        agb_loss_total_tc = agb_loss_scar_tc + agb_loss_flame_tc

        soc_loss_total_tc = float(np.sum(soc_grid[mask] * soc_combustion_factor * pixel_area_ha))
        total_carbon_tc = agb_loss_total_tc + soc_loss_total_tc
        total_co2e_tonnes = total_carbon_tc * co2_ratio

        results[regime_name] = {
            "burned_area_ha": round(vector_area_ha, 4),
            "polygon_count": num_polygons,
            "agb_carbon_loss_tc": round(agb_loss_total_tc, 4),
            "soc_carbon_loss_tc": round(soc_loss_total_tc, 4),
            "total_carbon_loss_tc": round(total_carbon_tc, 4),
            "co2_equivalent_tonnes": round(total_co2e_tonnes, 4),
        }

    return results


@contextmanager
def track_pipeline_carbon_footprint(
    project_suffix: str = "Run",
    output_dir: Union[str, Path] = "logs/codecarbon",
    enabled: bool = True,
) -> Iterator[Dict[str, float]]:
    """Context manager wrapping `codecarbon.EmissionsTracker` with project prefix `Watchtower_UQ_Seg`.

    Yields a mutable dictionary populated with `emissions_kg_co2eq` upon exit.
    """
    project_name = (
        project_suffix
        if project_suffix.startswith("Watchtower_UQ_Seg")
        else f"Watchtower_UQ_Seg_{project_suffix}"
    )
    metrics_holder: Dict[str, float] = {"emissions_kg_co2eq": 0.0}

    if not enabled:
        yield metrics_holder
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tracker = None
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=str(out_path),
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
    except Exception:
        tracker = None

    try:
        yield metrics_holder
    finally:
        if tracker is not None:
            try:
                emissions = tracker.stop()
                if emissions is not None:
                    metrics_holder["emissions_kg_co2eq"] = float(emissions)
            except Exception:
                pass

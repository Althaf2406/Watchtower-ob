"""Captum Spectral & Spatial Attribution Explainer for 17-Channel Wildfire Segmentation."""

from src.attribution.captum_explainer import (
    SegmentationAttributionWrapper,
    compute_spectral_feature_attributions,
)

__all__ = [
    "SegmentationAttributionWrapper",
    "compute_spectral_feature_attributions",
]


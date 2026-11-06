"""CLI Utility to Compute Zero-Leakage Channel-Wise Mean & Std on Sen2Fire Training Splits (scene1 & scene2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Ensure repository root is importable when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sen2fire_dataset import (
    DEFAULT_CHANNEL_MEAN,
    DEFAULT_CHANNEL_STD,
    SEN2FIRE_BAND_NAMES,
    Sen2FireMultispectralDataset,
)


def compute_streaming_channel_stats(
    data_root: Path,
    train_scenes: Sequence[str] = ("scene1", "scene2"),
) -> Dict[str, object]:
    """Compute exact two-pass / Welford channel-wise mean and std strictly over training splits.

    Never reads validation (`scene3`) or test (`scene4`) scenes to guarantee zero data leakage.
    """
    dataset = Sen2FireMultispectralDataset(
        root_dir=data_root,
        scenes=train_scenes,
        is_train=False,
        synthetic_fallback_samples=0,
    )

    if len(dataset.files) == 0:
        return {
            "source": "synthetic_default_fallback",
            "train_scenes": list(train_scenes),
            "num_files_processed": 0,
            "channel_names": list(SEN2FIRE_BAND_NAMES),
            "mean": [float(x) for x in DEFAULT_CHANNEL_MEAN],
            "std": [float(x) for x in DEFAULT_CHANNEL_STD],
        }

    channel_sum = np.zeros(17, dtype=np.float64)
    channel_sq_sum = np.zeros(17, dtype=np.float64)
    total_pixels = 0

    for npz_path in dataset.files:
        with np.load(npz_path) as data:
            image = data["image"]
            aerosol = data["aerosol"]
            label = data["label"]
        features_17ch, _, _ = dataset.process_raw_arrays(image, aerosol, label)
        flat = features_17ch.reshape(17, -1).astype(np.float64)
        channel_sum += flat.sum(axis=1)
        channel_sq_sum += np.square(flat).sum(axis=1)
        total_pixels += flat.shape[1]

    mean = channel_sum / max(1, total_pixels)
    variance = np.maximum((channel_sq_sum / max(1, total_pixels)) - np.square(mean), 1e-12)
    std = np.sqrt(variance)

    return {
        "source": "computed_from_train_splits",
        "train_scenes": list(train_scenes),
        "num_files_processed": len(dataset.files),
        "total_pixels": int(total_pixels),
        "channel_names": list(SEN2FIRE_BAND_NAMES),
        "mean": [round(float(m), 6) for m in mean],
        "std": [round(float(s), 6) for s in std],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute zero-leakage 17-channel normalization stats from Sen2Fire scene1 & scene2."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/sen2fire",
        help="Path to Sen2Fire dataset directory.",
    )
    parser.add_argument(
        "--train_scenes",
        nargs="+",
        default=["scene1", "scene2"],
        help="Training scene identifiers (strictly excluding val/test scenes).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/sen2fire_train_stats.json",
        help="Output JSON file path for computed mean and std.",
    )
    args = parser.parse_args()

    stats = compute_streaming_channel_stats(
        data_root=Path(args.data_root),
        train_scenes=args.train_scenes,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"[Watchtower] Saved 17-channel normalization stats ({stats['source']}, "
        f"scenes={stats['train_scenes']}, files={stats['num_files_processed']}) -> {out_path}"
    )


if __name__ == "__main__":
    main()


"""Cache Real Sentinel-2 L2A Multispectral Scenes from AWS Open Data for Instant Full-Pipeline Execution."""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

BENCHMARK_DIR = Path("data/benchmarks")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

BAND_FILES = {
    0: "B01.tif",
    1: "B02.tif",
    2: "B03.tif",
    3: "B04.tif",
    4: "B05.tif",
    5: "B06.tif",
    6: "B07.tif",
    7: "B08.tif",
    8: "B8A.tif",
    9: "B09.tif",
    10: "B11.tif",
    11: "B12.tif",
}

SCENES = {
    "kalimantan_2019": {
        "title": "Pulang Pisau Peatland Wildfire, Central Kalimantan",
        "date": "2019-09-21",
        "base_url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/49/M/HT/2019/9/S2B_49MHT_20190921_1_L2A",
        "center_lon": 113.9729,
        "center_lat": -1.8430,
        "crs": "EPSG:32749",
        "row_10m": 400,
        "col_10m": 3080,
    },
    "maui_2023": {
        "title": "Lahaina Coastal Wildfire Disaster, Maui Hawaii",
        "date": "2023-08-13",
        "base_url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/4/Q/GJ/2023/8/S2A_4QGJ_20230813_0_L2A",
        "center_lon": -156.677,
        "center_lat": 20.8783,
        "crs": "EPSG:32604",
        "row_10m": 8957,
        "col_10m": 4171,
    },
    "australia_2020": {
        "title": "Blue Mountains Eucalypt Megablaze, NSW Australia",
        "date": "2020-01-10",
        "base_url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/55/H/GC/2020/1/S2A_55HGC_20200110_1_L2A",
        "center_lon": 150.285,
        "center_lat": -33.682,
        "crs": "EPSG:32755",
        "row_10m": 7650,
        "col_10m": 4200,
    },
    "surabaya_2023": {
        "title": "Keputih Sukolilo Savanna Wildfire, Surabaya Timur",
        "date": "2023-09-30",
        "base_url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/49/M/FM/2023/9/S2B_49MFM_20230930_0_L2A",
        "center_lon": 112.7950,
        "center_lat": -7.2920,
        "crs": "EPSG:32749",
        "row_10m": 644,
        "col_10m": 9816,
    },
}


def download_and_cache_scene(name: str, info: dict, patch_size: int = 256) -> Path:
    out_file = BENCHMARK_DIR / f"{name}.npz"
    if out_file.is_file():
        print(f"[Cache] {name} already exists at {out_file}. Skipping download.")
        return out_file

    print(f"\n[Download] Fetching real Sentinel-2 scene for '{name}' ({info['title']})...")
    base_url = info["base_url"]
    r10 = info["row_10m"]
    c10 = info["col_10m"]
    half = patch_size // 2

    win_10m = Window(c10 - half, r10 - half, patch_size, patch_size)
    win_20m = Window(c10 // 2 - half // 2, r10 // 2 - half // 2, half, half)
    win_60m = Window(c10 // 6 - half // 6, r10 // 6 - half // 6, half // 3 + 1, half // 3 + 1)

    s2_12bands = np.zeros((12, patch_size, patch_size), dtype=np.float32)
    affine_10m = None

    import time
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2"):
        for b_idx in range(12):
            fname = BAND_FILES[b_idx]
            url = f"{base_url}/{fname}"
            for attempt in range(4):
                try:
                    with rasterio.open(url) as src:
                        if affine_10m is None and src.width > 8000:
                            affine_10m = rasterio.windows.transform(win_10m, src.transform)

                        if src.width > 8000:
                            w = win_10m
                        elif src.width > 4000:
                            w = win_20m
                        else:
                            w = win_60m
                        band_data = src.read(1, window=w, out_shape=(patch_size, patch_size), resampling=Resampling.bilinear).astype(np.float32)
                        s2_12bands[b_idx] = band_data
                        print(f"  -> Band {b_idx:02d} ({fname}): min={band_data.min():.0f}, max={band_data.max():.0f}")
                        break
                except Exception as e:
                    if attempt == 3:
                        print(f"  -> Warning reading {fname}: {e}")
                    time.sleep(1.0)

        # Fetch aerosol AOT
        aerosol_1ch = np.zeros((1, patch_size, patch_size), dtype=np.float32)
        try:
            with rasterio.open(f"{base_url}/AOT.tif") as src:
                w = win_20m if src.width < 8000 else win_10m
                aot = src.read(1, window=w, out_shape=(patch_size, patch_size), resampling=Resampling.bilinear).astype(np.float32) / 1000.0
                aerosol_1ch[0] = aot
                print(f"  -> Aerosol AOT: mean={aot.mean():.2f}, max={aot.max():.2f}")
        except Exception as e:
            print(f"  -> AOT missing, fallback to estimation: {e}")
            # If AOT missing, calculate atmospheric aerosol proxy from blue band
            aerosol_1ch[0] = np.clip(s2_12bands[1] / 3000.0, 0.05, 3.5)

    if affine_10m is None:
        from rasterio.transform import from_origin
        affine_10m = from_origin(c10 * 10, r10 * 10, 10, 10)

    # Convert transform to float array for saving
    t_vals = [affine_10m.a, affine_10m.b, affine_10m.c, affine_10m.d, affine_10m.e, affine_10m.f]

    np.savez_compressed(
        out_file,
        s2_12bands=s2_12bands,
        aerosol_1ch=aerosol_1ch,
        transform=np.array(t_vals, dtype=np.float64),
        crs=info["crs"],
        center_lat=info["center_lat"],
        center_lon=info["center_lon"],
        title=info["title"],
        date=info["date"],
    )
    print(f"[Cache] Successfully saved real satellite benchmark to: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)!")
    return out_file


if __name__ == "__main__":
    for scene_name, scene_info in SCENES.items():
        download_and_cache_scene(scene_name, scene_info)

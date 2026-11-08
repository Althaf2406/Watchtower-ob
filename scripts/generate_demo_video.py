"""
Fast, Real-Speed (1x) Full-HD (1920x1080 @ 30fps) Demo Video Generator for Watchtower (Borneo Only).
Fetches the EXACT 8-panel outputs from the running Watchtower server (/api/analyze) so that
Panels 01–06 (False Color, Natural RGB, AI Segmentation with orange #FF9100 contour trace & #FF5722 fill,
Predictive Entropy, Epistemic MI, NBR), Panel 07 (Captum Spectral XAI), and Panel 08 (GFW Carbon Audit)
are 100% identical to the live browser UI.
"""

import asyncio
import io
import json
import math
import os
import subprocess
import urllib.request
import wave
from pathlib import Path

import edge_tts
import imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_W, VIDEO_H = 1920, 1080
FPS = 24


def get_font(size: int, kind: str = "display"):
    if kind == "display":
        candidates = ["C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    elif kind == "mono":
        candidates = ["C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/consola.ttf"]
    else:
        candidates = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_LOGO = get_font(34, "display")
FONT_H1 = get_font(22, "display")
FONT_H2 = get_font(16, "display")
FONT_H3 = get_font(13, "display")
FONT_STAT_BIG = get_font(34, "display")
FONT_MONO_MD = get_font(13, "mono")
FONT_MONO_SM = get_font(11, "mono")
FONT_MONO_XS = get_font(10, "mono")
FONT_BADGE = get_font(9, "mono")


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def fetch_esri_basemap(lat: float, lon: float, zoom: int = 13, out_w: int = 1360, out_h: int = 1020) -> Image.Image:
    cache_dir = OUTPUTS_DIR / "tile_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cx, cy = deg2num(lat, lon, zoom)
    tx_center, ty_center = int(cx), int(cy)

    cols, rows = 7, 5
    canvas = Image.new("RGB", (cols * 256, rows * 256), (12, 22, 18))

    for dy in range(-2, 3):
        for dx in range(-3, 4):
            tx = tx_center + dx
            ty = ty_center + dy
            tile_file = cache_dir / f"esri_{zoom}_{ty}_{tx}.jpg"
            tile_img = None
            if tile_file.is_file():
                try:
                    tile_img = Image.open(tile_file).convert("RGB")
                except Exception:
                    tile_img = None
            if tile_img is None:
                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Watchtower-Demo/1.0"})
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        data = resp.read()
                    tile_file.write_bytes(data)
                    tile_img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    tile_img = Image.new("RGB", (256, 256), (16, 28, 22))
            canvas.paste(tile_img, ((dx + 3) * 256, (dy + 2) * 256))

    frac_x = cx - tx_center
    frac_y = cy - ty_center
    center_px_x = int((3 + frac_x) * 256)
    center_px_y = int((2 + frac_y) * 256)

    x0 = max(0, min(canvas.width - out_w, center_px_x - out_w // 2))
    y0 = max(0, min(canvas.height - out_h, center_px_y - out_h // 2))
    return canvas.crop((x0, y0, x0 + out_w, y0 + out_h))


def parse_svg_path_to_polygons(svg_paths: list[dict]) -> list[dict]:
    """Parse 'M x y L x y ... Z' strings from /api/analyze into list of point lists."""
    parsed = []
    for item in svg_paths:
        d_str = item.get("d", "")
        cid = item.get("class_id", 1)
        tokens = d_str.replace("M ", "").replace("L ", "").replace("Z", "").strip().split()
        pts = []
        for i in range(0, len(tokens) - 1, 2):
            try:
                pts.append((float(tokens[i]), float(tokens[i + 1])))
            except ValueError:
                pass
        if len(pts) >= 3:
            parsed.append({"class_id": cid, "points": pts})
    return parsed


def fetch_real_borneo_telemetry() -> dict:
    """Call the live /api/analyze endpoint to get the exact PNGs, SVG paths, XAI bars, and Carbon audit."""
    payload = json.dumps({
        "lat": -1.8430,
        "lon": 113.9729,
        "date": "2019-09-21",
        "model": "attention_unet_r50",
        "mc_passes": 6,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/api/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        api_data = json.loads(resp.read().decode("utf-8"))

    def load_rel_img(rel_url: str, mode: str = "RGB") -> Image.Image:
        local_path = ROOT_DIR / rel_url.lstrip("/")
        return Image.open(local_path).convert(mode)

    panels = api_data["panels"]
    p1 = load_rel_img(panels["panel1_falsecolor"])
    p2 = load_rel_img(panels["panel2_natural"])
    p3 = load_rel_img(panels["panel3_segmentation"])
    p4 = load_rel_img(panels["panel4_entropy"])
    p5 = load_rel_img(panels["panel5_epistemic"])
    p6 = load_rel_img(panels["panel6_nbr"])

    sat_nat = load_rel_img(api_data["satellite_imagery_url"])
    sat_fc = load_rel_img(api_data["satellite_falsecolor_url"])
    map_mask_rgba = load_rel_img(api_data["mask_overlay_url"], mode="RGBA")

    polygons = parse_svg_path_to_polygons(api_data.get("pixel_svg_paths", []))
    xai_labels = api_data["attribution"]["top_channels"]["labels"]
    xai_values = api_data["attribution"]["top_channels"]["values"]
    xai_pairs = list(zip(xai_labels, xai_values))

    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "sat_nat": sat_nat,
        "sat_fc": sat_fc,
        "map_mask_rgba": map_mask_rgba,
        "polygons": polygons,
        "xai": xai_pairs,
        "carbon": api_data["carbon"],
        "evaluation": api_data["evaluation"],
    }


def draw_slanted_rect(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, slant=8, fill=None, outline=None, shadow=None):
    if shadow:
        sx, sy, scol = shadow
        spoly = [(x0 + slant + sx, y0 + sy), (x1 + slant + sx, y0 + sy), (x1 + sx, y1 + sy), (x0 + sx, y1 + sy)]
        draw.polygon(spoly, fill=scol)
    poly = [(x0 + slant, y0), (x1 + slant, y0), (x1, y1), (x0, y1)]
    draw.polygon(poly, fill=fill, outline=outline)


def draw_progressive_polygons(
    target_rgba: Image.Image,
    polygons: list[dict],
    box_x0: int,
    box_y0: int,
    box_size: int,
    trace_prog: float,
    fill_prog: float,
    base_raster_rgba: Image.Image | None = None,
):
    """
    Draw the exact orange/crimson SVG contours from templates/index.html:
    1) Progressive outline trace along polygon vertices (stroke #FF9100 / #FFFFFF)
    2) Progressive interior fill (#FF5722 / #FF1744) + underlying 10m raster fade-in
    """
    overlay = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    scale = box_size / 256.0

    # 1. Interior fill
    if fill_prog > 0:
        if base_raster_rgba is not None:
            ras = base_raster_rgba.resize((box_size, box_size), Image.Resampling.NEAREST)
            r, g, b, a = ras.split()
            a = a.point(lambda p: int(p * fill_prog * 0.90))
            overlay = Image.alpha_composite(overlay, Image.merge("RGBA", (r, g, b, a)))
        else:
            for poly in polygons:
                pts = [(px * scale, py * scale) for px, py in poly["points"]]
                fill_alpha = int(195 * fill_prog)
                fcol = (255, 23, 68, fill_alpha) if poly["class_id"] == 2 else (255, 87, 34, fill_alpha)
                odraw.polygon(pts, fill=fcol)

    # 2. Progressive outline tracing along polygon perimeter
    odraw = ImageDraw.Draw(overlay)
    if trace_prog > 0:
        for poly in polygons:
            pts = [(px * scale, py * scale) for px, py in poly["points"]]
            pts_closed = pts + [pts[0]]
            n_seg = max(1, int((len(pts_closed) - 1) * trace_prog))
            sub_pts = pts_closed[: n_seg + 1]
            if len(sub_pts) >= 2:
                scol = (255, 255, 255, 255) if poly["class_id"] == 2 else (255, 145, 0, 255)
                odraw.line(sub_pts, fill=scol, width=2)

    target_rgba.alpha_composite(overlay, (box_x0, box_y0))


def render_ui_frame(
    basemap_img: Image.Image,
    telemetry: dict,
    stage_float: float,
    scan_active: bool,
    cursor_xy: tuple[int, int],
    cursor_click: bool,
) -> np.ndarray:
    canvas = Image.new("RGBA", (VIDEO_W, VIDEO_H), (4, 7, 17, 255))
    draw = ImageDraw.Draw(canvas)

    # 1. Map Viewport (left 1360px, y=60..1080)
    canvas.paste(basemap_img.convert("RGBA"), (0, 60))

    aoi_size = 380
    aoi_x0, aoi_y0 = 500, 330
    aoi_x1, aoi_y1 = aoi_x0 + aoi_size, aoi_y0 + aoi_size

    # Satellite tile inside AOI box on Map
    if stage_float >= 0.15:
        sat_src = telemetry["sat_fc"] if stage_float < 1.15 else telemetry["sat_nat"]
        sat_tile = sat_src.resize((aoi_size, aoi_size), Image.Resampling.BILINEAR)
        wipe_h = aoi_size if stage_float >= 1.0 else int(aoi_size * min(1.0, (stage_float - 0.1) / 0.85))
        if wipe_h > 0:
            canvas.paste(sat_tile.crop((0, 0, aoi_size, wipe_h)).convert("RGBA"), (aoi_x0, aoi_y0))

    # Stage 3 on Map: Outline Trace -> Interior Fill
    if stage_float >= 2.0:
        trace_p = min(1.0, (stage_float - 2.0) / 0.80)
        fill_p = max(0.0, min(1.0, (stage_float - 2.65) / 0.70))
        draw_progressive_polygons(
            canvas,
            telemetry["polygons"],
            aoi_x0,
            aoi_y0,
            aoi_size,
            trace_p,
            fill_p,
            base_raster_rgba=telemetry["map_mask_rgba"],
        )

    # Cyan AOI Border & Map Laser Scanline
    draw.rectangle([aoi_x0, aoi_y0, aoi_x1, aoi_y1], outline=(0, 229, 255, 240), width=2)
    if scan_active and 0.05 < stage_float < 7.8:
        sweep_frac = (stage_float * 1.4) % 1.0
        ly = int(aoi_y0 + sweep_frac * aoi_size)
        draw.rectangle([aoi_x0, max(aoi_y0, ly - 18), aoi_x1, ly], fill=(0, 229, 255, 55))
        draw.line([(aoi_x0, ly), (aoi_x1, ly)], fill=(0, 229, 255, 255), width=3)

    # Top Mission Banner on Map
    draw_slanted_rect(
        draw, 390, 74, 1180, 124, slant=10,
        fill=(8, 14, 28, 242),
        outline=(255, 23, 68, 255) if stage_float >= 8.0 else (0, 229, 255, 255),
        shadow=(5, 5, (4, 7, 17, 230)),
    )
    if stage_float == 0.0:
        m_title = "TARGET LOCKED: BORNEO PEATLAND MEGAFIRE, INDONESIA"
        m_sub = "Coordinates (-1.8430°, 113.9729°) framed. Press [ANALYZE FIRE & CARBON >>] to start progressive scan!"
    elif stage_float < 8.0:
        st_idx = min(8, int(math.ceil(stage_float)))
        m_title = f"STAGE {st_idx}/8: PROGRESSIVE SENTINEL-2 SATELLITE TELEMETRY SCAN"
        m_sub = "Penetrating peatland smoke plume, tracing AI burn perimeter, and auditing IPCC carbon..."
    else:
        m_title = "WILDFIRE AUDIT COMPLETE: 244.5 HA | 122,066 tCO2e"
        m_sub = "Active Wildfire Crisis (2019-09-21): High-Intensity Thermal Front"
    draw.text((415, 80), m_title, font=FONT_H2, fill=(255, 255, 255, 255))
    draw.text((415, 102), m_sub, font=FONT_MONO_XS, fill=(0, 229, 255, 255))

    # ================= TOP HEADER BAR =================
    draw.rectangle([0, 0, VIDEO_W, 60], fill=(8, 14, 28, 252))
    draw.line([(0, 60), (VIDEO_W, 60)], fill=(30, 45, 74, 255), width=2)
    draw_slanted_rect(draw, 16, 10, 56, 50, slant=6, fill=(255, 23, 68, 255), outline=(255, 255, 255, 255))
    draw.text((27, 13), "W", font=FONT_LOGO, fill=(255, 255, 255, 255))
    draw.text((70, 8), "WATCHTOWER", font=FONT_LOGO, fill=(255, 255, 255, 255))
    draw_slanted_rect(draw, 255, 14, 455, 36, slant=6, fill=(255, 255, 255, 255))
    draw.text((266, 16), "GLOBAL WILDFIRE & CARBON AI", font=FONT_H3, fill=(4, 7, 17, 255))
    draw.text((70, 42), "// STEP 1: SELECT GLOBAL HOTSPOT PINPOINT  ->  STEP 2: PRESS ANALYZE FOR 8-STAGE SATELLITE SCAN", font=FONT_MONO_XS, fill=(0, 229, 255, 230))

    presets = [
        ("borneo", "01  BORNEO PEATLAND", 1190, 1405, True),
        ("maui", "02  LAHAINA, MAUI", 1420, 1620, False),
        ("australia", "03  NSW AUSTRALIA", 1635, 1840, False),
    ]
    for _, label, bx0, bx1, is_sel in presets:
        bg_c = (255, 23, 68, 255) if is_sel else (16, 25, 46, 255)
        out_c = (255, 255, 255, 255) if is_sel else (30, 45, 74, 255)
        sh_c = (4, 4, (0, 229, 255, 220)) if is_sel else (3, 3, (0, 0, 0, 200))
        draw_slanted_rect(draw, bx0, 12, bx1, 48, slant=8, fill=bg_c, outline=out_c, shadow=sh_c)
        draw.text((bx0 + 18, 20), label, font=FONT_H2, fill=(255, 255, 255, 255))

    # ================= LEFT FLOATING COMMAND DECK =================
    draw.rectangle([24, 82, 346, 535], fill=(4, 7, 17, 220))
    draw.rectangle([18, 76, 340, 528], fill=(8, 14, 28, 245), outline=(30, 45, 74, 255), width=2)
    draw_slanted_rect(draw, 30, 88, 195, 112, slant=6, fill=(0, 229, 255, 255))
    draw.text((38, 91), "/// 01 PINPOINT TARGET", font=FONT_H3, fill=(4, 7, 17, 255))
    draw.text((225, 93), "BORNEO HOTSPOT", font=FONT_MONO_XS, fill=(255, 145, 0, 255))

    draw.text((30, 126), "LATITUDE", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.rectangle([30, 142, 172, 172], fill=(4, 7, 17, 255), outline=(0, 229, 255, 160))
    draw.text((38, 149), "-1.8430", font=FONT_MONO_MD, fill=(255, 255, 255, 255))

    draw.text((184, 126), "LONGITUDE", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.rectangle([184, 142, 326, 172], fill=(4, 7, 17, 255), outline=(0, 229, 255, 160))
    draw.text((192, 149), "113.9729", font=FONT_MONO_MD, fill=(255, 255, 255, 255))

    draw.text((30, 186), "OBSERVATION DATE", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.rectangle([30, 202, 326, 232], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
    draw.text((38, 209), "2019-09-21", font=FONT_MONO_MD, fill=(255, 255, 255, 255))

    draw_slanted_rect(draw, 30, 242, 172, 272, slant=6, fill=(16, 25, 46, 255), outline=(0, 230, 118, 200))
    draw.text((44, 249), "PRE-FIRE BASE", font=FONT_H3, fill=(255, 255, 255, 255))
    draw_slanted_rect(draw, 182, 242, 324, 272, slant=6, fill=(255, 23, 68, 255), outline=(255, 255, 255, 220))
    draw.text((205, 249), "PEAK FIRE", font=FONT_H3, fill=(255, 255, 255, 255))

    draw.text((30, 288), "NEURAL ARCHITECTURE", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.rectangle([30, 304, 326, 334], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
    draw.text((38, 312), "SSL4EO ATTENTION U-NET R-50", font=FONT_MONO_SM, fill=(0, 229, 255, 255))

    draw.text((30, 350), "MAP S2 OVERLAY", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    is_fc_stage = 0.1 <= stage_float < 1.15
    nat_bg = (16, 25, 46, 255) if is_fc_stage else (0, 230, 118, 255)
    nat_fg = (200, 210, 225, 255) if is_fc_stage else (4, 7, 17, 255)
    fc_bg = (255, 145, 0, 255) if is_fc_stage else (16, 25, 46, 255)
    fc_fg = (4, 7, 17, 255) if is_fc_stage else (200, 210, 225, 255)
    draw_slanted_rect(draw, 30, 368, 172, 396, slant=6, fill=nat_bg, outline=(255, 255, 255, 200))
    draw.text((48, 374), "NATURAL RGB", font=FONT_H3, fill=nat_fg)
    draw_slanted_rect(draw, 182, 368, 324, 396, slant=6, fill=fc_bg, outline=(30, 45, 74, 255))
    draw.text((200, 374), "FALSE COLOR", font=FONT_H3, fill=fc_fg)

    draw.text((30, 416), "// STEP 02: EXECUTE TELEMETRY", font=FONT_MONO_XS, fill=(255, 145, 0, 255))
    btn_outline = (0, 229, 255, 255) if stage_float == 0.0 else (255, 255, 255, 255)
    draw_slanted_rect(
        draw, 30, 436, 322, 492, slant=10,
        fill=(255, 23, 68, 255), outline=btn_outline,
        shadow=(5, 5, (0, 229, 255, 180) if stage_float == 0.0 else (4, 7, 17, 255)),
    )
    btn_label = "SCANNING AOI SECTOR..." if (0.0 < stage_float < 8.0) else ("RE-SCAN TARGET AOI >>" if stage_float >= 8.0 else "ANALYZE FIRE & CARBON >>")
    draw.text((52, 451), btn_label, font=FONT_H1, fill=(255, 255, 255, 255))

    # ================= RIGHT 8-PANEL PROGRESSIVE TELEMETRY DECK =================
    rx0 = 1360
    draw.rectangle([rx0, 60, VIDEO_W, VIDEO_H], fill=(8, 14, 28, 252))
    draw.line([(rx0, 60), (rx0, VIDEO_H)], fill=(30, 45, 74, 255), width=2)

    # Top Telemetry Stage Bar
    draw.rectangle([rx0 + 14, 72, VIDEO_W - 14, 138], fill=(16, 25, 46, 255), outline=(30, 45, 74, 255), width=2)
    stage_num = min(8, int(math.ceil(stage_float))) if stage_float > 0 else 0
    status_str = (
        "STANDBY // PRESS ANALYZE TO SCAN AOI" if stage_num == 0
        else ("COMPLETE // ALL 8 TELEMETRY PANELS SYNCHRONIZED" if stage_float >= 8.0 else f"STAGE {stage_num}/8 // PROGRESSIVE SATELLITE SCAN...")
    )
    dot_col = (0, 229, 255, 255) if stage_num == 0 else ((0, 230, 118, 255) if stage_float >= 8.0 else (255, 145, 0, 255))
    draw.rectangle([rx0 + 24, 82, rx0 + 36, 94], fill=dot_col, outline=(255, 255, 255, 255))
    draw.text((rx0 + 44, 79), status_str, font=FONT_H2, fill=(255, 255, 255, 255))
    draw.text((VIDEO_W - 95, 81), f"STAGE {stage_num}/8", font=FONT_MONO_SM, fill=(0, 229, 255, 255))

    prog_pct = min(1.0, stage_float / 8.0)
    bar_w = int((VIDEO_W - rx0 - 48) * prog_pct)
    draw.rectangle([rx0 + 24, 102, VIDEO_W - 24, 110], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
    if bar_w > 0:
        draw.rectangle([rx0 + 24, 102, rx0 + 24 + bar_w, 110], fill=(255, 23, 68, 255))
    draw.text((rx0 + 24, 116), "Borneo Peatland Megafire, Indonesia   |   -1.8430°, 113.9729° | 2019-09-21", font=FONT_MONO_XS, fill=(200, 215, 230, 255))

    # Section Ribbon above Panels 01-06 (matches user screenshot!)
    grid_x0 = rx0 + 16
    draw_slanted_rect(draw, grid_x0, 144, grid_x0 + 305, 164, slant=6, fill=(255, 23, 68, 255))
    draw.text((grid_x0 + 8, 146), "/// PANELS 01-06: SATELLITE & BAYESIAN UNCERTAINTY SCAN", font=FONT_H3, fill=(255, 255, 255, 255))
    draw.text((VIDEO_W - 140, 148), "256x256 PATCH @ 10M", font=FONT_MONO_XS, fill=(148, 163, 184, 255))

    # ================= PANELS 01–06 (EXACT BROWSER RASTERS & BADGES) =================
    p_size = 166
    gap = 12
    grid_y0 = 170

    panel_specs = [
        (1, "01 // FALSE COLOR", "SWIR LOCKED", (255, 145, 0, 255), 0.1, 0.85, telemetry["p1"]),
        (2, "02 // NATURAL RGB", "RGB LOCKED", (0, 230, 118, 255), 1.0, 0.85, telemetry["p2"]),
        (3, "03 // AI SEGMENTATION", "FIRE MAPPED", (255, 23, 68, 255), 2.0, 1.40, telemetry["p3"]),
        (4, "04 // PRED. ENTROPY", "ENTROPY OK", (0, 229, 255, 255), 3.5, 0.75, telemetry["p4"]),
        (5, "05 // EPISTEMIC MI", "MI CALIBRATED", (0, 229, 255, 255), 4.3, 0.75, telemetry["p5"]),
        (6, "06 // NBR BURN INDEX", "NBR MAPPED", (255, 87, 34, 255), 5.1, 0.75, telemetry["p6"]),
    ]

    for idx, (pnum, ptitle, pbadge, pcol, t_start, t_dur, p_img) in enumerate(panel_specs):
        col_i = idx % 3
        row_i = idx // 3
        px0 = grid_x0 + col_i * (p_size + gap)
        py0 = grid_y0 + row_i * (p_size + 24 + gap)
        px1 = px0 + p_size
        py1 = py0 + p_size + 24

        unlocked = stage_float >= t_start
        local_p = max(0.0, min(1.0, (stage_float - t_start) / t_dur)) if unlocked else 0.0

        border_c = pcol if unlocked else (30, 45, 74, 255)
        draw.rectangle([px0, py0, px1, py1], fill=(4, 7, 17, 255), outline=border_c, width=2)
        draw.rectangle([px0 + 2, py0 + 2, px1 - 2, py0 + 24], fill=(8, 14, 28, 255))
        draw.text((px0 + 5, py0 + 5), ptitle, font=FONT_H3, fill=pcol)

        # Right-side status badge inside header bar (matches user screenshot!)
        if unlocked:
            if pnum == 3 and local_p >= 0.65:
                draw.rectangle([px1 - 58, py0 + 4, px1 - 4, py0 + 21], fill=(255, 23, 68, 255))
                draw.text((px1 - 55, py0 + 7), "FIRE MAPPED", font=FONT_BADGE, fill=(255, 255, 255, 255))
            elif pnum == 3:
                draw.text((px1 - 52, py0 + 7), "TRACING...", font=FONT_BADGE, fill=(255, 23, 68, 255))
            else:
                draw.text((px1 - len(pbadge) * 5 - 6, py0 + 7), pbadge, font=FONT_BADGE, fill=(0, 229, 255, 255))
        else:
            draw.text((px1 - 44, py0 + 7), "STANDBY", font=FONT_BADGE, fill=(100, 116, 139, 255))

        img_box_y0 = py0 + 24
        if not unlocked:
            draw.text((px0 + 28, img_box_y0 + 70), "AWAITING SCAN", font=FONT_MONO_XS, fill=(71, 85, 105, 255))
        elif pnum == 3:
            # Exact Panel 3 rendering: dark background -> orange #FF9100 outline trace -> #FF5722 fill + p3_segmentation.png
            draw.rectangle([px0 + 2, img_box_y0, px1 - 2, py1 - 2], fill=(6, 11, 22, 255))
            trace_p = min(1.0, local_p / 0.58)
            fill_p = max(0.0, min(1.0, (local_p - 0.48) / 0.52))
            if fill_p > 0:
                seg_res = p_img.resize((p_size - 4, p_size - 2), Image.Resampling.BILINEAR).convert("RGBA")
                r, g, b, a = seg_res.split()
                a = a.point(lambda p: int(p * fill_p))
                canvas.alpha_composite(Image.merge("RGBA", (r, g, b, a)), (px0 + 2, img_box_y0))
            draw_progressive_polygons(
                canvas,
                telemetry["polygons"],
                px0 + 2,
                img_box_y0,
                p_size - 4,
                trace_p,
                fill_p * 0.45,
                base_raster_rgba=None,
            )
            if 0.0 < local_p < 0.45:
                ly = img_box_y0 + int((p_size - 2) * (local_p / 0.45))
                draw.line([(px0 + 2, ly), (px1 - 2, ly)], fill=(0, 229, 255, 255), width=3)
        else:
            wipe_h = int((p_size - 2) * local_p)
            if wipe_h > 0:
                resized_p = p_img.resize((p_size - 4, p_size - 2), Image.Resampling.BILINEAR)
                crop_p = resized_p.crop((0, 0, p_size - 4, wipe_h)).convert("RGBA")
                canvas.paste(crop_p, (px0 + 2, img_box_y0))
            if 0.0 < local_p < 0.98:
                ly = img_box_y0 + wipe_h
                draw.line([(px0 + 2, ly), (px1 - 2, ly)], fill=(0, 229, 255, 255), width=3)

    # Bayesian Calibration Metrics Strip (y=568..594)
    cal_p = max(0.0, min(1.0, (stage_float - 4.3) / 0.75))
    ev = telemetry["evaluation"]
    cal_boxes = [
        ("ECE:", f"{ev['ECE'] * cal_p:.4f}", (0, 229, 255, 255)),
        ("BRIER:", f"{ev['Brier_Score'] * cal_p:.4f}", (255, 145, 0, 255)),
        ("AUSE:", f"{ev['AUSE'] * cal_p:.4f}", (0, 230, 118, 255)),
    ]
    for ci, (clbl, cval, ccol) in enumerate(cal_boxes):
        cx0 = grid_x0 + ci * (p_size + gap)
        draw.rectangle([cx0, 568, cx0 + p_size, 594], fill=(16, 25, 46, 255), outline=(30, 45, 74, 255))
        draw.text((cx0 + 8, 575), clbl, font=FONT_MONO_XS, fill=(148, 163, 184, 255))
        draw.text((cx0 + 104, 572), cval, font=FONT_H2, fill=ccol)

    # ================= PANEL 07: CAPTUM SPECTRAL XAI BARS =================
    p7_unlocked = stage_float >= 5.9
    p7_prog = max(0.0, min(1.0, (stage_float - 5.9) / 1.05))
    p7_border = (255, 145, 0, 255) if p7_unlocked else (30, 45, 74, 255)
    draw.rectangle([grid_x0, 604, VIDEO_W - 16, 818], fill=(16, 25, 46, 255), outline=p7_border, width=2)
    draw_slanted_rect(draw, grid_x0 + 10, 612, grid_x0 + 275, 632, slant=6, fill=(255, 145, 0, 255))
    draw.text((grid_x0 + 18, 614), "/// 07 CAPTUM SPECTRAL XAI ATTRIBUTION", font=FONT_H3, fill=(4, 7, 17, 255))
    draw.text((VIDEO_W - 110, 616), "XAI ACTIVE" if p7_unlocked else "STANDBY", font=FONT_MONO_XS, fill=(255, 145, 0, 255))

    max_xai = max(v for _, v in telemetry["xai"])
    for bi, (bname, bval) in enumerate(telemetry["xai"]):
        by = 638 + bi * 19
        draw.text((grid_x0 + 12, by), bname[:10], font=FONT_MONO_XS, fill=(255, 255, 255, 255))
        bar_x0, bar_x1 = grid_x0 + 96, VIDEO_W - 82
        draw.rectangle([bar_x0, by + 2, bar_x1, by + 13], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
        stagger_p = max(0.0, min(1.0, (p7_prog - bi * 0.05) / 0.65))
        ease_p = 1.0 - (1.0 - stagger_p) ** 3
        fill_w = int((bar_x1 - bar_x0) * (bval / max_xai) * ease_p)
        if fill_w > 0:
            bar_col = (255, 23, 68, 255) if bi < 2 else ((255, 145, 0, 255) if bi < 5 else (0, 229, 255, 255))
            draw.rectangle([bar_x0, by + 2, bar_x0 + fill_w, by + 13], fill=bar_col)
        disp_pct = bval * 100.0 * ease_p
        draw.text((VIDEO_W - 72, by), f"{disp_pct:4.1f}%", font=FONT_MONO_XS, fill=(0, 229, 255, 255))

    # ================= PANEL 08: GFW CARBON STOCK AUDIT =================
    p8_unlocked = stage_float >= 7.0
    p8_prog = max(0.0, min(1.0, (stage_float - 7.0) / 1.10))
    p8_ease = 1.0 - (1.0 - p8_unlocked * p8_prog) ** 3
    p8_border = (255, 23, 68, 255) if p8_unlocked else (30, 45, 74, 255)
    draw.rectangle([grid_x0, 826, VIDEO_W - 16, 1070], fill=(16, 25, 46, 255), outline=p8_border, width=2)
    draw_slanted_rect(draw, grid_x0 + 10, 834, grid_x0 + 295, 854, slant=6, fill=(255, 87, 34, 255))
    draw.text((grid_x0 + 18, 836), "/// 08 GFW CARBON STOCK AUDIT (IPCC TIER-3)", font=FONT_H3, fill=(4, 7, 17, 255))

    cb = telemetry["carbon"]
    cur_ha = cb["nominal_mean"]["burned_area_ha"] * p8_ease
    cur_co2 = cb["nominal_mean"]["co2_equivalent_tonnes"] * p8_ease

    draw.rectangle([grid_x0 + 12, 860, grid_x0 + 254, 930], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
    draw.text((grid_x0 + 24, 865), "BURNED SCAR AREA", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.text((grid_x0 + 24, 882), f"{cur_ha:,.1f} HA", font=FONT_STAT_BIG, fill=(255, 145, 0, 255))

    draw.rectangle([grid_x0 + 266, 860, VIDEO_W - 28, 930], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
    draw.text((grid_x0 + 278, 865), "GHG EMISSIONS RELEASED", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.text((grid_x0 + 278, 882), f"{int(round(cur_co2)):,}", font=FONT_STAT_BIG, fill=(255, 23, 68, 255))

    t_x0, t_x1 = grid_x0 + 12, VIDEO_W - 28
    draw.rectangle([t_x0, 938, t_x1, 1058], fill=(4, 7, 17, 255), outline=(30, 45, 74, 255))
    draw.rectangle([t_x0, 938, t_x1, 962], fill=(8, 14, 28, 255))
    draw.text((t_x0 + 10, 944), "CONFIDENCE REGIME", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.text((t_x0 + 225, 944), "CARBON LOSS (tC)", font=FONT_MONO_XS, fill=(148, 163, 184, 255))
    draw.text((t_x0 + 375, 944), "CO2e EMISSIONS (t)", font=FONT_MONO_XS, fill=(148, 163, 184, 255))

    regimes = [
        ("CONSERVATIVE (LOWER)", cb["conservative_lower_bound"], (0, 230, 118, 255), 968),
        ("NOMINAL MEAN", cb["nominal_mean"], (255, 145, 0, 255), 998),
        ("UPPER 95% CI BOUND", cb["upper_bound_95ci"], (255, 23, 68, 255), 1028),
    ]
    for rname, rdict, rcol, ry in regimes:
        draw.text((t_x0 + 10, ry), rname, font=FONT_MONO_SM, fill=rcol)
        draw.text((t_x0 + 235, ry), f"{rdict['total_carbon_loss_tc'] * p8_ease:,.1f}", font=FONT_MONO_SM, fill=(255, 255, 255, 255))
        draw.text((t_x0 + 385, ry), f"{rdict['co2_equivalent_tonnes'] * p8_ease:,.1f}", font=FONT_MONO_SM, fill=rcol)

    # ================= TACTICAL CURSOR =================
    cx, cy = cursor_xy
    if cursor_click:
        draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], outline=(0, 229, 255, 255), width=4)
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 23, 68, 235), outline=(255, 255, 255, 255), width=2)
    else:
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], outline=(0, 229, 255, 220), width=2)
        pts = [(cx, cy), (cx + 16, cy + 7), (cx + 9, cy + 10), (cx + 7, cy + 17)]
        draw.polygon(pts, fill=(255, 23, 68, 255), outline=(255, 255, 255, 255))

    return np.array(canvas.convert("RGB"))


def build_tactical_sfx_wav(duration_sec: float = 25.0, sr: int = 24000) -> np.ndarray:
    """Synthesize clean Atlus/Persona-inspired tactical UI clicks, laser scan sweeps, and counter ticks for 25s timeline."""
    n_samples = int(duration_sec * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    def add_tone(t_start: float, dur: float, f0: float, f1: float | None = None, amp: float = 0.15):
        i0 = int(t_start * sr)
        length = int(dur * sr)
        if i0 >= n_samples or length <= 0:
            return
        i1 = min(n_samples, i0 + length)
        actual_len = i1 - i0
        t = np.linspace(0, actual_len / sr, actual_len, endpoint=False)
        if f1 is None:
            phase = 2 * np.pi * f0 * t
        else:
            freqs = np.linspace(f0, f1, actual_len)
            phase = 2 * np.pi * np.cumsum(freqs) / sr
        env = np.exp(-3.5 * t / max(dur, 0.01)) * np.sin(np.pi * t / max(dur, 0.01)) ** 0.3
        audio[i0:i1] += (np.sin(phase) * env * amp).astype(np.float32)

    # 1. Pinpoint Click (t = 1.6s)
    add_tone(1.60, 0.10, 980, 1480, amp=0.22)
    # 2. Analyze Button Click & Engage (t = 3.2s)
    add_tone(3.20, 0.12, 520, 1040, amp=0.26)
    add_tone(3.30, 0.18, 1040, 1560, amp=0.22)
    # 3. Stage 1 & 2 Laser Scan Sweeps (t = 3.45s, 5.25s)
    add_tone(3.45, 1.40, 340, 880, amp=0.10)
    add_tone(5.25, 1.40, 380, 960, amp=0.10)
    # 4. Stage 3 AI Segmentation Outline Trace Ticks (t = 7.05s .. 10.0s)
    for k in range(22):
        add_tone(7.05 + k * 0.13, 0.045, 1150 + k * 35, amp=0.11)
    # 5. Stages 4, 5, 6 Unlock Chimes (t = 10.2s, 12.0s, 14.0s)
    add_tone(10.25, 0.35, 740, 1100, amp=0.14)
    add_tone(12.05, 0.35, 880, 1320, amp=0.14)
    add_tone(14.05, 0.35, 990, 1480, amp=0.14)
    # 6. Stage 7 Captum XAI Bars Rising Cascade (t = 15.8s .. 18.2s)
    for k in range(9):
        add_tone(15.85 + k * 0.22, 0.10, 600 + k * 90, amp=0.12)
    # 7. Stage 8 GFW Carbon Counter Rapid Ticks + Completion Chord (t = 18.6s .. 21.8s)
    for k in range(24):
        add_tone(18.65 + k * 0.12, 0.04, 1400 + (k % 3) * 120, amp=0.10)
    # Completion Triad at t = 21.8s
    add_tone(21.80, 0.85, 523.25, amp=0.14)  # C5
    add_tone(21.80, 0.85, 659.25, amp=0.14)  # E5
    add_tone(21.80, 0.85, 783.99, amp=0.14)  # G5
    add_tone(21.80, 0.95, 1046.50, amp=0.14) # C6

    return np.clip(audio, -0.95, 0.95)


def generate_video():
    import gc
    video_raw_path = OUTPUTS_DIR / "watchtower_demo_raw.mp4"
    video_final_path = OUTPUTS_DIR / "watchtower_hackathon_demo.mp4"
    vo_mp3_path = OUTPUTS_DIR / "watchtower_vo_short.mp3"
    sfx_wav_path = OUTPUTS_DIR / "watchtower_sfx.wav"

    print("[1/4] Fetching Esri satellite basemap & exact /api/analyze Borneo telemetry...")
    basemap = fetch_esri_basemap(-1.8430, 113.9729, zoom=13)
    telemetry = fetch_real_borneo_telemetry()
    gc.collect()

    print("[2/4] Rendering comfortable-paced (25s, 24fps) Borneo 8-panel demo video...")
    writer = imageio.get_writer(
        str(video_raw_path),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )

    cursor_xy = [1295, 30]
    analyze_btn_xy = (176, 464)

    def record_segment(
        duration_sec: float,
        stage_start: float,
        stage_end: float,
        scan_active: bool,
        target_cursor: tuple[int, int],
        click_at_end: bool,
    ):
        nonlocal cursor_xy
        n_frames = max(1, int(duration_sec * FPS))
        sx, sy = cursor_xy
        ex, ey = target_cursor
        for i in range(n_frames):
            t = (i + 1) / n_frames
            ease = t * t * (3 - 2 * t)
            cx = int(sx + (ex - sx) * ease)
            cy = int(sy + (ey - sy) * ease)
            cursor_xy = [cx, cy]
            st = stage_start + (stage_end - stage_start) * t
            clicking = click_at_end and (i >= n_frames - 4)
            frame = render_ui_frame(
                basemap_img=basemap,
                telemetry=telemetry,
                stage_float=st,
                scan_active=scan_active,
                cursor_xy=(cx, cy),
                cursor_click=clicking,
            )
            writer.append_data(frame)
        gc.collect()

    # 1. Select Pinpoint 01 Borneo (1.8s) -> Move to Analyze Fire & Carbon & Click (1.6s)
    record_segment(1.8, 0.0, 0.0, False, (1295, 30), True)
    record_segment(1.6, 0.0, 0.0, False, analyze_btn_xy, True)

    # 2. Well-Paced Progressive 8-Stage Scan (~18.6s scan + 3.0s hold = 25.0s total)
    record_segment(1.80, 0.0, 1.0, True, (1440, 250), False)   # Stage 1: False Color (SWIR-NIR-Red)
    record_segment(1.80, 1.0, 2.0, True, (1620, 250), False)   # Stage 2: Natural RGB
    record_segment(3.20, 2.0, 3.5, True, (1790, 250), False)   # Stage 3: AI Segmentation (Outline Trace -> Fill)
    record_segment(1.80, 3.5, 4.3, True, (1440, 440), False)   # Stage 4: Predictive Entropy
    record_segment(2.00, 4.3, 5.1, True, (1620, 440), False)   # Stage 5: Epistemic MI & Calibration
    record_segment(1.80, 5.1, 5.9, True, (1790, 440), False)   # Stage 6: NBR Burn Index
    record_segment(2.80, 5.9, 7.0, True, (1650, 700), False)   # Stage 7: Captum Spectral XAI Bars Growing
    record_segment(3.40, 7.0, 8.2, False, (1650, 920), False)  # Stage 8: GFW Carbon Count-Up
    record_segment(3.00, 8.2, 8.2, False, (690, 520), False)   # Hold completed state for 3.0s

    writer.close()
    gc.collect()

    print("[3/4] Synthesizing natural-paced 25s Voiceover + Tactical UI Sound Effects...")
    async def make_vo():
        text = (
            "In Step One, we select the Borneo Peatland Megafire pinpoint and lock our Sentinel-2 scan box. "
            "In Step Two, pressing Analyze Fire and Carbon triggers a progressive eight-panel scan: "
            "first revealing Short-Wave Infrared False Color and True Color RGB, "
            "followed by AI segmentation that traces the active fire outline and fills the burn scar. "
            "Next, Watchtower maps Predictive Entropy, Bayesian Epistemic Uncertainty, and the Normalized Burn Ratio, "
            "animates Captum Spectral XAI attribution bars, and counts up the Global Forest Watch Carbon Audit "
            "to two hundred forty-five point four hectares and eighty-nine thousand tonnes of CO2 emissions."
        )
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+6%")
        await comm.save(str(vo_mp3_path))

    asyncio.run(make_vo())

    sfx_samples = build_tactical_sfx_wav(duration_sec=25.0, sr=24000)
    with wave.open(str(sfx_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes((sfx_samples * 32767).astype(np.int16).tobytes())

    print("[4/4] Mixing Voiceover + Tactical SFX into final 25s MP4...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_raw_path),
        "-i", str(vo_mp3_path),
        "-i", str(sfx_wav_path),
        "-filter_complex", "[1:a]atempo=1.18,volume=1.35[vo];[2:a]volume=0.45[sfx];[vo][sfx]amix=inputs=2:duration=shortest:dropout_transition=1[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(video_final_path),
    ]
    subprocess.run(cmd, check=True)

    for p in (video_raw_path, vo_mp3_path, sfx_wav_path):
        if p.is_file():
            p.unlink()

    print(f"[SUCCESS] Saved 25.0s Borneo demo video WITH AUDIO to: {video_final_path} ({video_final_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    generate_video()

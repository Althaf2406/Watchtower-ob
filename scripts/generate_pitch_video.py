"""
Watchtower — Full Motion-Graphics Hackathon Pitching Video Generator.
Every single frame is dynamically animated (NO static PowerPoint slides):
- Scene 1 (Act 01): 3D Rotating Earth Globe + Orbiting Sentinel-2 Satellite with Pulsing Laser Scan Cone + Kinetic Typography
- Scene 2 (Act 02): Dynamic Moving Laser Split-Screen Smoke-Cutter (Optical RGB Blindness vs 2190nm SWIR + AI Fire Mask)
- Scene 3 (Act 03): Animated 16-Band Spectral Wave Prism -> Pulsing Attention U-Net Neural Nodes -> Bayesian MC Particle Cloud -> Captum XAI Equalizer
- Scene 4 (Act 04): Live Borneo Screen Recording framed in an Animated Tactical Orbital HUD with Dynamic Stage Callouts & Audio Waveform
- Scene 5 (Act 05): Rotating Tech-Stack Orbital Constellation + Animated Circular & Linear Scientific Benchmark Gauges (ECE, Brier, AUSE, VRAM)
- Scene 6 (Act 06): Global Telemetry Arcs Connecting Borneo, Maui & Australia on 3D Globe + Kinetic Closing Emblem
"""

import asyncio
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

import edge_tts
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

USER_RECORDING_PATH = Path(
    r"C:\Users\altha\.gemini\antigravity\brain\a59a8596-90f0-416d-b572-bff212dd9ba5\.user_uploaded\media_1794099531929_470a4f64.mp4"
)
USER_LOGO_PATH = Path(
    r"C:\Users\altha\.gemini\antigravity\brain\a59a8596-90f0-416d-b572-bff212dd9ba5\.user_uploaded\media_1794109703636_78fabae8.png"
)
ARTIFACT_DIR = Path(r"C:\Users\altha\.gemini\antigravity\brain\a59a8596-90f0-416d-b572-bff212dd9ba5")

# Render canvas at 1280x720 @ 15fps for fast procedural drawing, upscaled by FFmpeg to 1920x1080 @ 30fps
RW, RH = 1280, 720
RFPS = 15


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


F_HERO = get_font(46, "display")
F_TITLE = get_font(32, "display")
F_SUB = get_font(20, "display")
F_CARD = get_font(18, "display")
F_BODY = get_font(14, "sans")
F_MONO_LG = get_font(16, "mono")
F_MONO_MD = get_font(12, "mono")
F_MONO_SM = get_font(10, "mono")
F_STAT = get_font(30, "display")

# Palette
C_BG = (4, 7, 17)
C_PANEL = (10, 17, 34)
C_CARD = (16, 26, 48)
C_BORDER = (34, 52, 86)
C_CRIMSON = (255, 23, 68)
C_ORANGE = (255, 87, 34)
C_GOLD = (255, 145, 0)
C_CYAN = (0, 229, 255)
C_EMERALD = (0, 230, 118)
C_WHITE = (255, 255, 255)
C_MUTED = (148, 163, 184)

# Pre-generate deterministic starfield
RNG = np.random.default_rng(42)
STARS = [(int(RNG.integers(0, RW)), int(RNG.integers(55, RH - 45)), int(RNG.integers(90, 235)), float(RNG.uniform(0.5, 2.2))) for _ in range(95)]

# Simplified continent polygons (lat, lon) for 3D globe rendering
CONTINENT_POLYS = [
    # Southeast Asia / Borneo / Maritime
    [(-8, 95), (6, 95), (22, 105), (22, 122), (8, 126), (-8, 138), (-10, 115), (-4, 108)],
    # Australia
    [(-12, 130), (-15, 145), (-28, 153), (-38, 146), (-34, 115), (-20, 114)],
    # Eurasia
    [(35, -10), (60, 10), (70, 60), (65, 140), (35, 140), (22, 115), (10, 75), (25, 35)],
    # Africa
    [(35, -5), (32, 32), (10, 50), (-34, 25), (-34, 18), (5, 8), (15, -17)],
    # North America
    [(70, -160), (70, -60), (45, -55), (25, -80), (15, -95), (32, -117), (55, -135)],
    # South America
    [(10, -75), (-5, -35), (-23, -42), (-54, -68), (-18, -72), (5, -80)],
]

HOTSPOT_COORDS = [
    ("BORNEO PEATLAND", -1.8430, 113.9729, C_CRIMSON),
    ("NSW AUSTRALIA", -33.6702, 150.3640, C_GOLD),
    ("LAHAINA, MAUI", 20.8783, -156.6825, C_CYAN),
]

_GLOBE_CACHE = {}


def get_real_earth_globe_renderer(radius: int = 215, tilt_deg: float = -2.0):
    """
    Precompute 3D spherical ray-casting lookup tables for a real Esri World Imagery satellite globe.
    Returns a fast function `render_globe_patch(rot_lon_deg) -> Image.Image (RGBA)`.
    """
    key = (radius, round(tilt_deg, 2))
    if key in _GLOBE_CACHE:
        return _GLOBE_CACHE[key]

    tex_path = ASSETS_DIR / "pitch_textures" / "earth_satellite_z3.jpg"
    if not tex_path.is_file():
        tex_path = OUTPUTS_DIR / "earth_satellite_z3.jpg"
    if tex_path.is_file():
        tex_im = Image.open(tex_path).convert("RGB")
    else:
        tex_im = Image.new("RGB", (2048, 2048), (8, 24, 48))

    # Enhance brightness & landmass contrast so islands like Borneo, Java, Sumatra & Sulawesi pop vividly
    tex_np = np.asarray(tex_im, dtype=np.float32)
    tex_np = np.clip((tex_np - 18.0) * 1.45 + 24.0, 0, 255).astype(np.uint8)
    tex_h, tex_w, _ = tex_np.shape

    diam = radius * 2
    y_grid, x_grid = np.mgrid[0:diam, 0:diam].astype(np.float32)
    u = (x_grid - radius + 0.5) / radius
    v = (radius - 0.5 - y_grid) / radius
    r2 = u * u + v * v
    mask = r2 <= 1.0

    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    tilt = math.radians(tilt_deg)
    # Inverse tilt rotation (from screen normal to world coordinates)
    y_w = v * math.cos(tilt) + z * math.sin(tilt)
    z_w = -v * math.sin(tilt) + z * math.cos(tilt)
    x_w = u

    lat_rad = np.arcsin(np.clip(y_w, -0.9999, 0.9999))
    lon_rel_deg = np.degrees(np.arctan2(x_w, z_w))  # [-180, 180]

    # Convert latitude to Web Mercator Y index in [0, tex_h - 1]
    merc_y = (1.0 - np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0)) / np.pi) * 0.5
    merc_y_idx = np.clip((merc_y * tex_h).astype(np.int32), 0, tex_h - 1)

    # 3D directional lighting (sunlight from top-left-front) + cyan atmospheric limb glow
    light = np.clip(0.22 + 0.82 * (0.25 * u + 0.25 * v + 0.92 * z), 0.18, 1.15)[..., None]
    rim = (np.clip(1.0 - z, 0.0, 1.0) ** 2.6)[..., None]
    alpha = (mask.astype(np.uint8)) * 255

    def render_patch(rot_lon_deg: float) -> Image.Image:
        lon_deg = (lon_rel_deg + rot_lon_deg + 180.0) % 360.0
        merc_x_idx = np.clip((lon_deg / 360.0 * tex_w).astype(np.int32), 0, tex_w - 1)
        sampled = tex_np[merc_y_idx, merc_x_idx].astype(np.float32)
        shaded = sampled * light + rim * np.array([0.0, 195.0, 255.0], dtype=np.float32) * 0.65
        rgb = np.where(mask[..., None], np.clip(shaded, 0, 255).astype(np.uint8), 0)
        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, mode="RGBA")

    _GLOBE_CACHE[key] = render_patch
    return render_patch


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def ease_out_back(x: float) -> float:
    x = max(0.0, min(1.0, x))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((x - 1.0) ** 3) + c1 * ((x - 1.0) ** 2)


_LOGO_CACHE = {}


def get_watchtower_logo(size: int) -> Image.Image:
    """Load the official split orange/cyan Watchtower 'W' logo with transparent background."""
    if size in _LOGO_CACHE:
        return _LOGO_CACHE[size]
    logo_path = USER_LOGO_PATH if USER_LOGO_PATH.exists() else (ASSETS_DIR / "favicon.png")
    raw = Image.open(logo_path).convert("RGBA")
    arr = np.array(raw)
    # Key out the dark navy background so the geometric orange/cyan W emblem floats cleanly
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    bg_mask = (r < 22) & (g < 28) & (b < 45)
    arr[:, :, 3] = np.where(bg_mask, 0, 255).astype(np.uint8)
    cleaned = Image.fromarray(arr, mode="RGBA").resize((size, size), Image.Resampling.LANCZOS)
    _LOGO_CACHE[size] = cleaned
    return cleaned


def draw_slanted(
    draw: ImageDraw.ImageDraw,
    x0,
    y0,
    x1,
    y1,
    slant=10,
    fill=None,
    outline=None,
    width=2,
    shadow=None,
):
    if shadow is not None:
        sx, sy, scol = shadow
        spoly = [(x0 + slant + sx, y0 + sy), (x1 + slant + sx, y0 + sy), (x1 + sx, y1 + sy), (x0 + sx, y1 + sy)]
        draw.polygon(spoly, fill=scol)
    poly = [(x0 + slant, y0), (x1 + slant, y0), (x1, y1), (x0, y1)]
    draw.polygon(poly, fill=fill, outline=outline, width=width)


def draw_motion_hud_shell(im: Image.Image, draw: ImageDraw.ImageDraw, t: float, badge: str, title: str, caption: str):
    """Animated starfield, moving grid, Website-Demo-Matched Top Command Header with Official Watchtower Logo, and bottom audio waveform + caption."""
    # Moving cyber grid
    offset_x = int((t * 28) % 64)
    offset_y = int((t * 14) % 64)
    for gx in range(-64 + offset_x, RW, 64):
        draw.line([(gx, 56), (gx, RH - 44)], fill=(9, 16, 32), width=1)
    for gy in range(56 + offset_y, RH - 44, 64):
        draw.line([(0, gy), (RW, gy)], fill=(9, 16, 32), width=1)

    # Twinkling drifting stars
    for sx, sy, sbr, sspd in STARS:
        px = int((sx - t * 18 * sspd) % RW)
        twinkle = int(sbr * (0.65 + 0.35 * math.sin(t * 4.0 + sx)))
        draw.rectangle([px, sy, px + 1, sy + 1], fill=(twinkle, twinkle, min(255, twinkle + 30)))

    # Top Header Bar (Exact match to Website Demo templates/index.html navbar)
    draw.rectangle([0, 0, RW, 56], fill=(10, 17, 34))

    # Animated diagonal Atlus hazard stripes across header background
    stripe_shift = int((t * 30) % 28)
    for k in range(-2, 24):
        x_s = RW - 620 + k * 28 + stripe_shift
        draw.polygon([(x_s, 0), (x_s + 12, 0), (x_s - 28, 56), (x_s - 40, 56)], fill=(28, 12, 28))

    draw.line([(0, 56), (RW, 56)], fill=C_BORDER, width=2)

    # 1. Official Watchtower Logo in slanted container with crimson offset shadow
    draw_slanted(
        draw,
        14,
        8,
        54,
        48,
        slant=5,
        fill=(11, 18, 34),
        outline=C_CRIMSON,
        width=2,
        shadow=(3, 3, C_CRIMSON),
    )
    logo_im = get_watchtower_logo(34)
    im.paste(logo_im, (19, 11), logo_im)

    # 2. Brand Title ("WATCHTOWER" with crimson drop-shadow) + "GLOBAL WILDFIRE & CARBON AI" + "SENTINEL-2 10M"
    draw.text((66, 7), "WATCHTOWER", font=F_SUB, fill=C_CRIMSON)
    draw.text((64, 5), "WATCHTOWER", font=F_SUB, fill=C_WHITE)

    draw_slanted(draw, 184, 7, 366, 26, slant=4, fill=C_WHITE)
    draw.polygon([(188, 7), (193, 7), (189, 26), (184, 26)], fill=C_CYAN)
    draw.text((197, 10), "GLOBAL WILDFIRE & CARBON AI", font=F_MONO_SM, fill=C_BG)

    draw_slanted(draw, 372, 7, 474, 26, slant=4, fill=C_CRIMSON)
    draw.text((382, 10), "SENTINEL-2 10M", font=F_MONO_SM, fill=C_WHITE)

    # Sub-line prompt matching website demo's "// STEP 1 -> 2 -> 3" sub-header
    sub_prefix = f"// {badge}:"
    draw.text((64, 32), sub_prefix, font=F_MONO_MD, fill=C_CYAN)
    prefix_w = len(sub_prefix) * 7 + 8
    draw.text((64 + prefix_w, 32), title, font=F_MONO_MD, fill=C_GOLD)

    # Bottom Subtitle + Live Audio Equalizer Bar
    draw.rectangle([0, RH - 44, RW, RH], fill=(6, 10, 22))
    draw.line([(0, RH - 44), (RW, RH - 44)], fill=C_CYAN, width=2)

    # Animated mini audio waveform bars on bottom-left
    for bi in range(14):
        bh = int(4 + 12 * abs(math.sin(t * 9.5 + bi * 0.7)))
        bx = 16 + bi * 6
        bcol = C_CRIMSON if bi % 3 == 0 else C_CYAN
        draw.rectangle([bx, RH - 22 - bh, bx + 4, RH - 22 + bh], fill=bcol)

    draw.text((112, RH - 31), caption, font=F_BODY, fill=C_WHITE)


def project_3d_globe(lat_deg: float, lon_deg: float, rot_lon_deg: float, tilt_deg: float, cx: int, cy: int, radius: float):
    """Project (lat, lon) onto 2D screen matching the exact 3D sphere ray-caster."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg - rot_lon_deg)
    tilt = math.radians(tilt_deg)

    # 3D world unit sphere coordinates
    x_w = math.cos(lat) * math.sin(lon)
    y_w = math.sin(lat)
    z_w = math.cos(lat) * math.cos(lon)

    # Forward tilt rotation to screen space (inverse of y_w = v*cos + z*sin, z_w = -v*sin + z*cos)
    v = y_w * math.cos(tilt) - z_w * math.sin(tilt)
    z_s = y_w * math.sin(tilt) + z_w * math.cos(tilt)

    sx = int(cx + radius * x_w)
    sy = int(cy - radius * v)
    return sx, sy, z_s


def draw_rotating_globe_and_satellite(
    im: Image.Image,
    draw: ImageDraw.ImageDraw,
    t: float,
    cx: int,
    cy: int,
    radius: int,
    rot_lon_deg: float,
    scan_progress: float,
    show_arcs: bool = False,
):
    """Draw a photorealistic 3D satellite Earth globe locked onto Borneo with an orbiting Sentinel-2 satellite."""
    tilt_deg = -2.0
    # Outer orbital atmosphere glow rings
    for r_off, col in [(radius + 22, (0, 65, 95)), (radius + 10, (0, 120, 165)), (radius + 3, C_CYAN)]:
        draw.ellipse([cx - r_off, cy - r_off, cx + r_off, cy + r_off], outline=col, width=1)

    # Paste photorealistic 3D ray-casted satellite Earth sphere!
    globe_patch_rgba = get_real_earth_globe_renderer(radius=radius, tilt_deg=tilt_deg)(rot_lon_deg)
    im.paste(globe_patch_rgba, (cx - radius, cy - radius), globe_patch_rgba)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=C_CYAN, width=2)

    # Subtle Latitude & Longitude 3D graticule lines over the real satellite globe
    for lat_g in range(-60, 75, 30):
        pts = []
        for lon_g in range(-180, 185, 8):
            sx, sy, z = project_3d_globe(lat_g, lon_g, rot_lon_deg, tilt_deg, cx, cy, radius)
            if z > 0.05:
                pts.append((sx, sy))
            else:
                if len(pts) >= 2:
                    draw.line(pts, fill=(0, 180, 220), width=1)
                pts = []
        if len(pts) >= 2:
            draw.line(pts, fill=(0, 180, 220), width=1)

    for lon_g in range(-180, 180, 30):
        pts = []
        for lat_g in range(-80, 85, 8):
            sx, sy, z = project_3d_globe(lat_g, lon_g, rot_lon_deg, tilt_deg, cx, cy, radius)
            if z > 0.05:
                pts.append((sx, sy))
            else:
                if len(pts) >= 2:
                    draw.line(pts, fill=(0, 180, 220), width=1)
                pts = []
        if len(pts) >= 2:
            draw.line(pts, fill=(0, 180, 220), width=1)

    # Highlight real Borneo (Kalimantan) coastline contour on the satellite globe
    borneo_coast = [
        (-4.1, 114.6), (-3.4, 111.8), (-1.2, 109.2), (1.9, 109.6),
        (4.5, 114.1), (6.9, 116.8), (5.2, 119.0), (1.1, 118.8),
        (-2.1, 116.5), (-4.1, 114.6),
    ]
    b_pts = []
    for blat, blon in borneo_coast:
        sx, sy, z = project_3d_globe(blat, blon, rot_lon_deg, tilt_deg, cx, cy, radius)
        if z > 0.05:
            b_pts.append((sx, sy))
    if len(b_pts) >= 4:
        draw.line(b_pts, fill=C_GOLD, width=2)

    # Pulsing Wildfire Hotspots on Globe (Borneo is ALWAYS index 0 and primary target!)
    visible_hotspots = []
    borneo_xy = (cx, cy)
    for idx, (hname, hlat, hlon, hcol) in enumerate(HOTSPOT_COORDS):
        sx, sy, z = project_3d_globe(hlat, hlon, rot_lon_deg, tilt_deg, cx, cy, radius)
        if z > 0.05:
            visible_hotspots.append((hname, sx, sy, hcol))
            if idx == 0:
                borneo_xy = (sx, sy)
                # Extra-prominent tactical target crosshair on Borneo!
                pulse_r = int(10 + 14 * ((t * 2.8) % 1.0))
                draw.ellipse([sx - pulse_r, sy - pulse_r, sx + pulse_r, sy + pulse_r], outline=C_CRIMSON, width=2)
                draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=C_CRIMSON, outline=C_WHITE, width=2)
                # Crosshair ticks
                draw.line([(sx - 18, sy), (sx - 8, sy)], fill=C_CYAN, width=2)
                draw.line([(sx + 8, sy), (sx + 18, sy)], fill=C_CYAN, width=2)
                draw.line([(sx, sy - 18), (sx, sy - 8)], fill=C_CYAN, width=2)
                draw.line([(sx, sy + 8), (sx, sy + 18)], fill=C_CYAN, width=2)
                # Callout badge for Borneo
                draw.line([(sx + 10, sy - 10), (sx + 34, sy - 28), (sx + 195, sy - 28)], fill=C_CRIMSON, width=2)
                draw.rectangle([sx + 34, sy - 50, sx + 205, sy - 28], fill=(8, 12, 24), outline=C_CRIMSON, width=2)
                draw.text((sx + 40, sy - 46), "BORNEO (-1.84°S, 113.97°E)", font=F_MONO_SM, fill=C_WHITE)
            else:
                pulse_r = int(5 + 7 * ((t * 2.2) % 1.0))
                draw.ellipse([sx - pulse_r, sy - pulse_r, sx + pulse_r, sy + pulse_r], outline=hcol, width=1)
                draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=C_WHITE, outline=hcol, width=1)
                draw.line([(sx + 6, sy - 6), (sx + 24, sy - 18), (sx + 125, sy - 18)], fill=hcol, width=1)
                draw.rectangle([sx + 24, sy - 34, sx + 132, sy - 18], fill=(6, 10, 22), outline=hcol, width=1)
                draw.text((sx + 28, sy - 31), hname, font=F_MONO_SM, fill=C_WHITE)

    # Telemetry arcs between Borneo and other visible hotspots if requested
    if show_arcs and len(visible_hotspots) >= 2:
        _, x1, y1, _ = visible_hotspots[0]
        for i in range(1, len(visible_hotspots)):
            _, x2, y2, _ = visible_hotspots[i]
            mx, my = (x1 + x2) // 2, min(y1, y2) - 48
            arc_pts = []
            for s in range(21):
                u = s / 20.0
                px = int((1 - u) ** 2 * x1 + 2 * (1 - u) * u * mx + u ** 2 * x2)
                py = int((1 - u) ** 2 * y1 + 2 * (1 - u) * u * my + u ** 2 * y2)
                arc_pts.append((px, py))
            draw.line(arc_pts, fill=C_CYAN, width=2)

    # Orbiting Sentinel-2 Satellite positioned in upper-left/top orbit so its laser locks directly onto BORNEO!
    sat_x = int(cx - 125 + 45 * math.sin(t * 0.85))
    sat_y = int(cy - radius - 22 + 12 * math.cos(t * 0.85))

    # Target point is ALWAYS Borneo!
    tx, ty = borneo_xy

    if scan_progress > 0:
        spread = 26
        cone_poly = [(sat_x, sat_y), (tx - spread, ty - 10), (tx + spread, ty + 10)]
        draw.polygon(cone_poly, outline=C_CYAN, width=1)
        for k in range(5):
            prog = ((t * 2.5 + k * 0.2) % 1.0)
            lx0 = int(sat_x + (tx - spread - sat_x) * prog)
            ly0 = int(sat_y + (ty - 10 - sat_y) * prog)
            lx1 = int(sat_x + (tx + spread - sat_x) * prog)
            ly1 = int(sat_y + (ty + 10 - sat_y) * prog)
            draw.line([(lx0, ly0), (lx1, ly1)], fill=C_CYAN if k % 2 == 0 else C_CRIMSON, width=2)

        sq_r = int(22 + 6 * math.sin(t * 6.0))
        draw.rectangle([tx - sq_r, ty - sq_r, tx + sq_r, ty + sq_r], outline=C_CYAN, width=2)
        scan_y = int(ty - sq_r + ((t * 3.0) % 1.0) * (2 * sq_r))
        draw.line([(tx - sq_r, scan_y), (tx + sq_r, scan_y)], fill=C_CRIMSON, width=3)

    # Draw Sentinel-2 Satellite Body & Solar Wings
    draw.polygon([(sat_x - 52, sat_y - 14), (sat_x - 16, sat_y - 6), (sat_x - 16, sat_y + 6), (sat_x - 52, sat_y + 14)], fill=(18, 75, 130), outline=C_CYAN, width=2)
    draw.polygon([(sat_x + 16, sat_y - 6), (sat_x + 52, sat_y - 14), (sat_x + 52, sat_y + 14), (sat_x + 16, sat_y + 6)], fill=(18, 75, 130), outline=C_CYAN, width=2)
    draw.rectangle([sat_x - 14, sat_y - 12, sat_x + 14, sat_y + 12], fill=C_GOLD, outline=C_WHITE, width=2)
    draw.ellipse([sat_x - 6, sat_y - 6, sat_x + 6, sat_y + 6], fill=C_CRIMSON, outline=C_WHITE, width=1)
    draw.text((sat_x - 42, sat_y - 28), "SENTINEL-2 L2A (10M)", font=F_MONO_SM, fill=C_CYAN)
    return borneo_xy


def render_scene1_frame(t_local: float, dur: float, caption: str, sat_preview: Image.Image) -> np.ndarray:
    """
    Scene 1 (Act 01): WordBoundary-synced Orbital Satellite Scanning Real 3D Satellite Globe -> Locked onto Borneo!
    WordBoundary landmarks:
    - 0.00s..7.40s: Global wildfire crisis hook (Globe rotates toward Borneo)
    - 7.42s ("Meet Watchtower"): Main Watchtower title & orbital lock engage
    - 11.99s ("scans the globe with ten-meter Sentinel-2"): Satellite fires laser scan cone & Borneo hologram card zooms out
    - 15.13s ("real-time"), 16.77s ("Bayesian"), 18.28s ("IPCC"): 3 bottom KPI cards pop in sequentially
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(im, draw, t_local, "ACT 01 // ORBITAL AI", "WHAT IS WATCHTOWER?", caption)

    # Rotate real Earth globe smoothly from Indian Ocean (55°E) to lock dead-center on Borneo (113.97°E) right as "Meet Watchtower" is spoken
    lock_p = ease_out_cubic(t_local / 7.5)
    rot_lon = 55.0 + 58.97 * lock_p + math.sin(t_local * 0.8) * 1.2
    # Laser scan cone activates as "Meet Watchtower... scans the globe" is spoken (t >= 7.4s)
    scan_p = ease_out_cubic((t_local - 7.4) / 2.2) if t_local >= 7.4 else 0.0
    borneo_xy = draw_rotating_globe_and_satellite(
        im, draw, t_local, cx=915, cy=375, radius=215, rot_lon_deg=rot_lon, scan_progress=scan_p
    )

    # Kinetic Left Title Reveal
    slide_in = ease_out_back(t_local / 1.1)
    lx = int(-420 + 465 * slide_in)

    is_watchtower_introduced = t_local >= 7.35
    draw_slanted(draw, lx, 82, lx + 310, 112, slant=8, fill=C_CRIMSON if is_watchtower_introduced else C_ORANGE)
    draw.text(
        (lx + 14, 86),
        "/// ORBITAL WILDFIRE & CARBON AI" if is_watchtower_introduced else "/// GLOBAL WILDFIRE & CARBON CRISIS",
        font=F_CARD,
        fill=C_WHITE,
    )

    draw.text((lx, 124), "REAL-TIME WILDFIRE", font=F_HERO, fill=C_WHITE)
    draw.text((lx, 174), "& CARBON INTELLIGENCE", font=F_HERO, fill=C_GOLD)
    draw.text((lx, 224), "FROM SENTINEL-2 ORBIT", font=F_HERO, fill=C_CYAN if is_watchtower_introduced else C_MUTED)

    # Before 11.8s ("scans the globe"), show Global Crisis Telemetry; at 11.8s, zoom out the Borneo Sentinel-2 Hologram Card!
    if t_local < 11.8:
        draw.rectangle([lx, 295, lx + 545, 505], fill=C_PANEL, outline=C_BORDER, width=2)
        draw.text((lx + 20, 312), "ORBITAL SURVEILLANCE STATUS: ACQUIRING TARGET...", font=F_CARD, fill=C_GOLD)
        draw.text((lx + 20, 348), "• Millions of forest hectares destroyed annually", font=F_BODY, fill=C_WHITE)
        draw.text((lx + 20, 378), "• Billions of tonnes of unverified CO2 released", font=F_BODY, fill=C_WHITE)
        draw.text((lx + 20, 408), "• Rotating to lock onto Borneo Peatland (-1.84°S, 113.97°E)...", font=F_BODY, fill=C_CYAN)
        prog_bar = int(505 * min(1.0, t_local / 11.5))
        draw.rectangle([lx + 20, 452, lx + 525, 476], fill=C_BG, outline=C_BORDER, width=1)
        draw.rectangle([lx + 20, 452, lx + 20 + prog_bar, 476], fill=C_CRIMSON if is_watchtower_introduced else C_GOLD)
    else:
        pop_p = ease_out_back((t_local - 11.8) / 0.9)
        bx_globe, by_globe = borneo_xy
        draw.line([(bx_globe, by_globe), (lx + 545, 305)], fill=C_CYAN, width=2)
        draw.line([(bx_globe, by_globe), (lx + 545, 495)], fill=C_CYAN, width=2)

        draw.rectangle([lx, 295, lx + 545, 505], fill=C_CARD, outline=C_CYAN, width=2)
        im.paste(sat_preview, (lx + 12, 307))
        draw.rectangle([lx + 12, 307, lx + 192, 487], outline=C_CRIMSON, width=2)
        sy = 307 + int(((t_local * 90) % 180))
        draw.line([(lx + 12, sy), (lx + 192, sy)], fill=C_CYAN, width=2)

        draw.text((lx + 208, 312), "TARGET LOCKED: BORNEO PEATLAND", font=F_CARD, fill=C_CRIMSON)
        draw.text((lx + 208, 340), "• 12 Sentinel-2 Bands + 4 Burn Indices", font=F_BODY, fill=C_WHITE)
        draw.text((lx + 208, 366), "• Smoke-Penetrating SWIR (2190nm @ 10m)", font=F_BODY, fill=C_GOLD)
        draw.text((lx + 208, 392), "• Attention U-Net + Bayesian MC Dropout", font=F_BODY, fill=C_CYAN)
        draw.text((lx + 208, 418), "• Captum Spectral XAI + IPCC Tier-3 CO2e", font=F_BODY, fill=C_EMERALD)

        co2_val = int(min(111900, max(0.0, t_local - 12.0) * 16500))
        draw.rectangle([lx + 208, 448, lx + 528, 490], fill=C_BG, outline=C_GOLD, width=1)
        draw.text((lx + 218, 458), f"AUDITED EMISSIONS: {co2_val:,} tCO2e", font=F_MONO_LG, fill=C_GOLD)

    # 3 Animated KPI Counters synced to "real-time burn segmentation" (15.13s), "Bayesian uncertainty" (16.77s), "IPCC Tier-3" (18.28s)
    kpis = [
        ("REAL-TIME AI", "10m Burn Segmentation", C_CYAN, 15.05),
        ("BAYESIAN UQ", "Calibrated Confidence", C_GOLD, 16.70),
        ("IPCC TIER-3", "Audited Carbon (tCO2e)", C_CRIMSON, 18.20),
    ]
    for ki, (kval, klbl, kcol, kdelay) in enumerate(kpis):
        kx = 45 + ki * 186
        if t_local >= kdelay:
            kp = ease_out_back((t_local - kdelay) / 0.65)
            ky = int(680 - 155 * min(1.0, kp))
            draw_slanted(draw, kx, ky, kx + 172, ky + 110, slant=8, fill=C_CARD, outline=kcol, width=2)
            draw.text((kx + 14, ky + 16), kval, font=F_SUB, fill=kcol)
            draw.text((kx + 14, ky + 56), klbl, font=F_MONO_SM, fill=C_WHITE)
        else:
            ky = 525
            draw_slanted(draw, kx, ky, kx + 172, ky + 110, slant=8, fill=C_PANEL, outline=C_BORDER, width=1)
            draw.text((kx + 14, ky + 16), kval, font=F_SUB, fill=(65, 80, 105))
            draw.text((kx + 14, ky + 56), klbl, font=F_MONO_SM, fill=(65, 80, 105))

    return np.array(im)


def render_scene2_frame(
    t_local: float,
    dur: float,
    caption: str,
    img_rgb_smoke: Image.Image,
    img_swir_fire: Image.Image,
    img_epistemic: Image.Image,
) -> np.ndarray:
    """
    Scene 2 (Act 02): Why Watchtower Matters — Dynamic Moving Laser Split-Screen Smoke-Cutter
    + 3 Animated Problem-to-Solution Motion Panels.
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(im, draw, t_local + 22.0, "ACT 02 // WHY IT MATTERS", "SOLVING 3 CRITICAL WILDFIRE & CARBON BLINDSPOTS", caption)

    # Left Interactive Split-Screen Scanner Box (x=42..562, y=78..650)
    bx0, by0, bw, bh = 42, 82, 520, 520
    # Oscillating laser scanner position (sweeps smoothly across the 520px satellite window)
    split_frac = 0.50 + 0.38 * math.sin(t_local * 1.45)
    split_px = int(bw * split_frac)

    # Left of laser: Optical RGB Blinded by Smoke; Right of laser: Watchtower SWIR + AI Fire Mask
    im.paste(img_rgb_smoke, (bx0, by0))
    right_crop = img_swir_fire.crop((split_px, 0, bw, bh))
    im.paste(right_crop, (bx0 + split_px, by0))

    draw.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=C_CYAN, width=2)

    # Glowing Vertical Laser Split Bar
    lx = bx0 + split_px
    draw.rectangle([max(bx0, lx - 14), by0, min(bx0 + bw, lx + 14), by0 + bh], outline=C_CYAN, width=1)
    draw.line([(lx, by0), (lx, by0 + bh)], fill=C_WHITE, width=4)
    draw.line([(lx - 3, by0), (lx - 3, by0 + bh)], fill=C_CRIMSON, width=2)
    draw.line([(lx + 3, by0), (lx + 3, by0 + bh)], fill=C_CYAN, width=2)

    # Labels on Left & Right of Split Scanner
    draw.rectangle([bx0 + 10, by0 + 10, bx0 + 235, by0 + 42], fill=(10, 14, 24), outline=C_MUTED, width=1)
    draw.text((bx0 + 18, by0 + 18), "OPTICAL RGB: BLINDED BY SMOKE", font=F_MONO_SM, fill=C_MUTED)

    draw.rectangle([bx0 + bw - 255, by0 + 10, bx0 + bw - 10, by0 + 42], fill=(28, 8, 18), outline=C_CRIMSON, width=2)
    draw.text((bx0 + bw - 245, by0 + 18), "WATCHTOWER SWIR + AI: CLEAR", font=F_MONO_SM, fill=C_GOLD)

    draw.rectangle([bx0, by0 + bh + 8, bx0 + bw, by0 + bh + 48], fill=C_CARD, outline=C_BORDER, width=1)
    draw.text((bx0 + 14, by0 + bh + 18), "LIVE SWIR SMOKE-PENETRATION SCANNER // 2190nm BAND B12 + AI MASK", font=F_MONO_SM, fill=C_CYAN)

    # Right Column: 3 Animated Problem -> Solution Kinetic Cards
    # WordBoundary exact timestamps inside Scene 2 (which starts at global 00:21.88):
    # - "First..." is spoken at t_local = 2.30s (global 00:24.18)
    # - "Second..." is spoken at t_local = 9.10s (global 00:30.98)
    # - "And third..." is spoken at t_local = 15.87s (global 00:37.75)
    cards = [
        (
            2.25,
            "GAP 01 // TOXIC SMOKE BLINDNESS",
            "Standard RGB satellites see only white smoke during peatland megafires.",
            "WATCHTOWER FIX: 16-Channel SWIR (2190nm) + Active Fire ratio cuts through dense smoke.",
            C_CRIMSON,
        ),
        (
            9.05,
            "GAP 02 // BLACK-BOX AI HALLUCINATIONS",
            "Deterministic U-Nets output uncalibrated masks with zero confidence bounds.",
            "WATCHTOWER FIX: Bayesian MC Dropout (T=20) + Temperature Scaling maps Epistemic MI.",
            C_CYAN,
        ),
        (
            15.80,
            "GAP 03 // MONTHS-LONG CARBON DELAYS",
            "Manual post-fire forest & peatland emission reporting takes months to audit.",
            "WATCHTOWER FIX: Automated GFW Above-Ground + Peat Carbon audit in 3 confidence regimes.",
            C_GOLD,
        ),
    ]

    for idx, (t_appear, ctitle, cprob, cfix, ccol) in enumerate(cards):
        cy0 = 82 + idx * 188
        is_lit = t_local >= t_appear
        is_current = is_lit and (idx == 2 or t_local < cards[idx + 1][0])
        active_p = ease_out_back((t_local - t_appear) / 0.75) if is_lit else 0.0
        cx0 = int(600 + (1.0 - min(1.0, active_p)) * 45) if is_lit else 645

        border_w = 4 if is_current else (2 if is_lit else 1)
        draw.rectangle(
            [cx0, cy0, 1245, cy0 + 170],
            fill=C_CARD if is_lit else (8, 12, 24),
            outline=ccol if is_lit else (24, 36, 58),
            width=border_w,
        )
        draw_slanted(draw, cx0 + 14, cy0 + 12, cx0 + 370, cy0 + 42, slant=8, fill=ccol if is_lit else (24, 36, 58))
        draw.text((cx0 + 26, cy0 + 16), ctitle, font=F_CARD, fill=C_BG if is_lit else (85, 98, 118))

        draw.text((cx0 + 18, cy0 + 56), f"PROBLEM: {cprob}", font=F_BODY, fill=C_WHITE if is_lit else (65, 76, 94))
        draw.text((cx0 + 18, cy0 + 88), cfix, font=F_BODY, fill=ccol if is_lit else (55, 66, 84))

        # Animated progress/confidence meter inside each card
        bar_w = int(600 * min(1.0, max(0.0, (t_local - t_appear) / 2.5))) if is_lit else 0
        draw.rectangle([cx0 + 18, cy0 + 128, cx0 + 618, cy0 + 148], fill=C_BG, outline=C_BORDER if is_lit else (20, 30, 48), width=1)
        if bar_w > 0:
            draw.rectangle([cx0 + 18, cy0 + 128, cx0 + 18 + bar_w, cy0 + 148], fill=ccol)

    return np.array(im)


def render_scene3_frame(
    t_local: float,
    dur: float,
    caption: str,
    img_seg: Image.Image,
    img_mi: Image.Image,
) -> np.ndarray:
    """
    Scene 3 (Act 03): How Watchtower Works — Animated 4-Stage Neural & Geospatial Dataflow Pipeline!
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(im, draw, t_local + 44.0, "ACT 03 // HOW IT WORKS", "4-STAGE ORBITAL AI & BAYESIAN CARBON ARCHITECTURE", caption)

    # 4 Animated Pipeline Nodes across the screen — synced to WordBoundary timestamps:
    # Stage 1 ("fuse twelve Sentinel-2"): 2.20s
    # Stage 2 ("Attention U-Net"): 10.50s
    # Stage 3 ("Next, Monte Carlo Dropout"): 13.40s
    # Stage 4 ("Captum Integrated Gradients" @ 18.47s, "Global Forest Watch" @ 21.75s): 18.35s
    node_specs = [
        (40, 310, "01 // 16-CH INGESTION", "12 Sentinel-2 + 4 Indices", C_GOLD, 2.20),
        (350, 620, "02 // ATTENTION U-NET", "SSL4EO-S12 ResNet-50", C_CRIMSON, 10.50),
        (660, 930, "03 // BAYESIAN MC UQ", "T=20 Dropout + Temp Scale", C_CYAN, 13.40),
        (970, 1240, "04 // XAI & IPCC CARBON", "Captum IG + GFW Tier-3", C_EMERALD, 18.35),
    ]

    # Animated data-stream conduits connecting the 4 stages
    for i in range(3):
        x_start = node_specs[i][1]
        x_end = node_specs[i + 1][0]
        next_unlock = node_specs[i + 1][5]
        y_mid = 335
        conduit_active = t_local >= next_unlock - 0.8
        draw.line([(x_start, y_mid), (x_end, y_mid)], fill=C_CYAN if conduit_active else C_BORDER, width=6)
        if conduit_active:
            p_pos = (t_local * 2.4 + i * 0.3) % 1.0
            px = int(x_start + (x_end - x_start) * p_pos)
            draw.ellipse([px - 7, y_mid - 7, px + 7, y_mid + 7], fill=C_CYAN, outline=C_WHITE, width=2)

    for idx, (nx0, nx1, ntitle, nsub, ncol, t_unlock) in enumerate(node_specs):
        ny0, ny1 = 86, 565
        is_lit = t_local >= t_unlock
        is_current = is_lit and (idx == 3 or t_local < node_specs[idx + 1][5])
        card_col = ncol if is_lit else (28, 40, 64)
        draw.rectangle(
            [nx0, ny0, nx1, ny1],
            fill=C_CARD if is_lit else (8, 12, 24),
            outline=card_col,
            width=4 if is_current else (2 if is_lit else 1),
        )
        draw_slanted(draw, nx0 + 10, ny0 + 10, nx1 - 18, ny0 + 38, slant=6, fill=card_col)
        draw.text((nx0 + 18, ny0 + 14), ntitle, font=F_CARD, fill=C_BG if is_lit else (85, 98, 118))
        draw.text((nx0 + 14, ny0 + 48), nsub, font=F_MONO_SM, fill=C_WHITE if is_lit else (65, 78, 96))
        draw.line([(nx0 + 12, ny0 + 70), (nx1 - 12, ny0 + 70)], fill=card_col, width=1)

        if idx == 0:
            bands = [
                ("B12 SWIR-2 (2190nm)", C_CRIMSON),
                ("B11 SWIR-1 (1610nm)", C_ORANGE),
                ("B08 NIR (842nm)", C_GOLD),
                ("NBR Burn Ratio", C_CYAN),
                ("dNBR Temporal Delta", C_EMERALD),
                ("AF_SWIR Thermal", C_CRIMSON),
                ("NDVI Canopy Index", C_EMERALD),
                ("SCL Cloud Bitmask", C_CYAN),
            ]
            for bi, (blbl, bcol) in enumerate(bands):
                by = ny0 + 92 + bi * 46
                draw.text((nx0 + 14, by - 12), blbl, font=F_MONO_SM, fill=C_WHITE if is_lit else (65, 78, 96))
                wave_pts = []
                amp = 7.0 if is_lit else 1.5
                for wx in range(nx0 + 14, nx1 - 14, 6):
                    wy = by + 12 + int(amp * math.sin(t_local * 5.0 + wx * 0.08 + bi))
                    wave_pts.append((wx, wy))
                draw.line(wave_pts, fill=bcol if is_lit else (35, 48, 70), width=2)

        elif idx == 1:
            layers = [(90, 26), (70, 36), (48, 48), (32, 62), (48, 48), (70, 36), (90, 26)]
            cx_u = (nx0 + nx1) // 2
            for li, (lw, lh) in enumerate(layers):
                ly = ny0 + 88 + li * 42
                pulse = int(4 * math.sin(t_local * 6.0 - li * 0.7)) if is_lit else 0
                lcol = (C_CRIMSON if li >= 3 else C_GOLD) if is_lit else (35, 48, 70)
                draw.rectangle([cx_u - lw - pulse, ly, cx_u + lw + pulse, ly + 28], fill=C_BG, outline=lcol, width=2)
                if li < 3:
                    target_y = ny0 + 88 + (6 - li) * 42 + 14
                    draw.arc([cx_u + lw + 4, ly + 10, cx_u + 118, target_y], start=-90, end=90, fill=C_CYAN if is_lit else (30, 42, 62), width=2)
                    if is_lit:
                        att_y = int(ly + 14 + ((t_local * 1.8 + li * 0.3) % 1.0) * (target_y - ly - 14))
                        draw.ellipse([cx_u + 105, att_y - 4, cx_u + 113, att_y + 4], fill=C_WHITE, outline=C_CYAN)

            if is_lit:
                im.paste(img_seg.resize((230, 82)), (nx0 + 20, ny1 - 96))
            draw.rectangle([nx0 + 20, ny1 - 96, nx0 + 250, ny1 - 14], outline=C_CRIMSON if is_lit else (35, 48, 70), width=2)

        elif idx == 2:
            draw.text((nx0 + 14, ny0 + 82), "MC DROPOUT POSTERIOR SAMPLING:", font=F_MONO_SM, fill=C_CYAN if is_lit else (65, 78, 96))
            for mc_i in range(6):
                my = ny0 + 112 + mc_i * 26
                prog = ((t_local - t_unlock) * 1.5 + mc_i * 0.17) % 1.0 if is_lit else 0.0
                mw = int(235 * prog)
                draw.rectangle([nx0 + 14, my, nx1 - 14, my + 16], fill=C_BG, outline=C_BORDER, width=1)
                if mw > 0:
                    draw.rectangle([nx0 + 14, my, nx0 + 14 + mw, my + 16], fill=C_CYAN if mc_i % 2 == 0 else C_GOLD)
                draw.text((nx0 + 20, my + 2), f"PASS T={mc_i*3+2:02d} // p(y|x,w_{mc_i+1})", font=F_MONO_SM, fill=C_WHITE if is_lit else (65, 78, 96))

            if is_lit:
                im.paste(img_mi.resize((230, 125)), (nx0 + 20, ny0 + 280))
            draw.rectangle([nx0 + 20, ny0 + 280, nx0 + 250, ny0 + 405], outline=C_CYAN if is_lit else (35, 48, 70), width=2)
            draw.text((nx0 + 20, ny0 + 416), "CALIBRATED ECE: 0.0080" if is_lit else "AWAITING MC PASSES...", font=F_MONO_MD, fill=C_EMERALD if is_lit else (65, 78, 96))
            draw.text((nx0 + 20, ny0 + 438), "BRIER: 0.0142 | AUSE: 0.0290", font=F_MONO_SM, fill=C_WHITE if is_lit else (55, 68, 86))

        elif idx == 3:
            draw.text((nx0 + 14, ny0 + 82), "CAPTUM INTEGRATED GRADIENTS:", font=F_MONO_SM, fill=C_GOLD if is_lit else (65, 78, 96))
            xai_bars = [("B12_SWIR2", 0.95, C_CRIMSON), ("NBR_Index", 0.86, C_ORANGE), ("AF_SWIR", 0.78, C_GOLD), ("dNBR", 0.68, C_CYAN), ("B11_SWIR1", 0.60, C_EMERALD)]
            xai_p = ease_out_cubic((t_local - 18.40) / 1.8) if is_lit else 0.0
            for xi, (xlbl, xval, xcol) in enumerate(xai_bars):
                xy = ny0 + 106 + xi * 32
                anim_w = int(155 * xval * xai_p)
                draw.text((nx0 + 14, xy), xlbl, font=F_MONO_SM, fill=C_WHITE if is_lit else (65, 78, 96))
                draw.rectangle([nx0 + 92, xy + 2, nx1 - 14, xy + 16], fill=C_BG, outline=C_BORDER, width=1)
                if anim_w > 0:
                    draw.rectangle([nx0 + 92, xy + 2, nx0 + 92 + anim_w, xy + 16], fill=xcol)

            # Live Ticking IPCC Tier-3 Carbon Counter Box synced to "Global Forest Watch" @ 21.75s!
            gfw_lit = t_local >= 21.50
            gfw_p = ease_out_cubic((t_local - 21.50) / 2.8) if gfw_lit else 0.0
            draw.rectangle([nx0 + 14, ny0 + 280, nx1 - 14, ny1 - 14], fill=C_BG, outline=C_EMERALD if gfw_lit else (35, 48, 70), width=2)
            draw.text((nx0 + 24, ny0 + 292), "GFW IPCC TIER-3 AUDIT", font=F_MONO_MD, fill=C_EMERALD if gfw_lit else (65, 78, 96))
            cur_ha = 245.4 * gfw_p
            cur_co2 = int(111900 * gfw_p)
            draw.text((nx0 + 24, ny0 + 320), f"{cur_ha:,.1f} HA", font=F_STAT, fill=C_GOLD if gfw_lit else (65, 78, 96))
            draw.text((nx0 + 24, ny0 + 362), f"{cur_co2:,} tCO2e", font=F_STAT, fill=C_CRIMSON if gfw_lit else (65, 78, 96))
            draw.text((nx0 + 24, ny0 + 412), "• Conservative: 90,707 tCO2e", font=F_MONO_SM, fill=C_EMERALD if gfw_lit else (55, 68, 86))
            draw.text((nx0 + 24, ny0 + 432), "• Nominal Mean: 111,900 tCO2e", font=F_MONO_SM, fill=C_GOLD if gfw_lit else (55, 68, 86))
            draw.text((nx0 + 24, ny0 + 452), "• 95% CI Upper: 129,858 tCO2e", font=F_MONO_SM, fill=C_CRIMSON if gfw_lit else (55, 68, 86))

    # Bottom Animated Formula Strip
    draw.rectangle([40, 580, 1240, 660], fill=C_PANEL, outline=C_CYAN, width=2)
    draw.text((58, 592), "MATHEMATICAL RIGOR //  L_total = 0.5·Focal(γ=2) + 0.5·Tversky(α=0.3, β=0.7)   |   I[y; w | x, D] = H(E_w[p]) - E_w[H(p)]", font=F_MONO_MD, fill=C_CYAN)
    draw.text((58, 622), "CARBON PROPAGATION //  E_CO2e = Area_burn(±1σ) × (AGB_GFW × CF_canopy + Peat_C × CF_peat) × 3.667 molecular ratio", font=F_MONO_MD, fill=C_GOLD)

    return np.array(im)


def render_scene5_frame(t_local: float, dur: float, caption: str) -> np.ndarray:
    """
    Scene 5 (Act 05): Orbital Tech-Stack Constellation + Animated Radial & Linear Scientific Gauges!
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(im, draw, t_local + 105.0, "ACT 05 // TECH STACK", "TECHNOLOGY CONSTELLATION & BENCHMARK VALIDATION", caption)

    # Left: Orbital Tech-Stack Constellation (x=40..720, y=80..655)
    cx, cy = 380, 365
    for r_orb in (95, 175, 245):
        draw.ellipse([cx - r_orb, cy - r_orb, cx + r_orb, cy + r_orb], outline=(20, 38, 68), width=1)

    # Tech nodes synced to WordBoundary timestamps:
    # "PyTorch" @ 2.37s, "TorchGeo" @ 3.18s, "Captum" @ 4.06s, "FastAPI" @ 4.75s, "Leaflet" @ 5.45s, "Bayesian" @ 8.16s
    tech_nodes = [
        ("PYTORCH 2.x + TORCHGEO", "16-Ch Tensor Engine", C_CRIMSON, 0.0, 2.30),
        ("SSL4EO ATTENTION U-NET", "Self-Supervised R-50", C_GOLD, math.pi / 3, 3.15),
        ("CAPTUM INTEGRATED GRAD", "16-Band Spectral XAI", C_EMERALD, 2 * math.pi / 3, 4.00),
        ("FASTAPI + LEAFLET GIS", "Real-Time Web HUD", C_CYAN, math.pi, 4.70),
        ("GLOBAL FOREST WATCH", "IPCC Tier-3 Biomass", C_ORANGE, 4 * math.pi / 3, 5.50),
        ("BAYESIAN MC DROPOUT", "T=20 + Temp Scaling", C_CYAN, 5 * math.pi / 3, 8.05),
    ]

    for ni, (ntitle, nsub, ncol, base_ang, t_node) in enumerate(tech_nodes):
        ang = base_ang + t_local * 0.25
        nx = int(cx + 215 * math.cos(ang))
        ny = int(cy + 185 * math.sin(ang))
        node_lit = t_local >= t_node

        draw.line([(cx, cy), (nx, ny)], fill=ncol if node_lit else (25, 38, 60), width=2)
        if node_lit:
            pkt_p = (t_local * 1.8 + ni * 0.2) % 1.0
            px = int(cx + (nx - cx) * pkt_p)
            py = int(cy + (ny - cy) * pkt_p)
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=C_WHITE, outline=ncol, width=2)

        draw.rectangle([nx - 105, ny - 32, nx + 105, ny + 32], fill=C_CARD if node_lit else (8, 12, 24), outline=ncol if node_lit else (28, 40, 62), width=2 if node_lit else 1)
        draw.text((nx - 95, ny - 22), ntitle, font=F_MONO_SM, fill=ncol if node_lit else (70, 82, 102))
        draw.text((nx - 95, ny + 2), nsub, font=F_BODY, fill=C_WHITE if node_lit else (60, 72, 90))

    # Central Watchtower AI Core with Official Split Orange/Cyan W Logo
    pulse_c = int(8 * math.sin(t_local * 5.0))
    draw.ellipse([cx - 66 - pulse_c, cy - 66 - pulse_c, cx + 66 + pulse_c, cy + 66 + pulse_c], fill=C_PANEL, outline=C_CRIMSON, width=3)
    core_logo = get_watchtower_logo(40)
    im.paste(core_logo, (cx - 20, cy - 46), core_logo)
    draw.text((cx - 46, cy - 4), "WATCHTOWER", font=F_CARD, fill=C_WHITE)
    draw.text((cx - 32, cy + 18), "AI CORE", font=F_MONO_SM, fill=C_CYAN)

    # Right: Animated Radial & Linear Scientific Benchmark Gauges (synced to "Our Bayesian calibration achieves an Expected Calibration Error..." @ 8.16s..15.01s)
    draw.rectangle([730, 82, 1245, 655], fill=C_PANEL, outline=C_BORDER, width=2)
    draw_slanted(draw, 748, 98, 1185, 128, slant=8, fill=C_CRIMSON if t_local >= 8.10 else (28, 40, 62))
    draw.text((762, 102), "/// VERIFIED SCIENTIFIC BENCHMARKS", font=F_CARD, fill=C_WHITE if t_local >= 8.10 else (85, 98, 118))

    gauges = [
        ("EXPECTED CALIBRATION ERROR (ECE)", 0.0080, "0.0080", 0.92, C_CYAN, "Post-hoc Temperature Scaling (T_opt)", 9.50),
        ("CALIBRATED BRIER SCORE", 0.0142, "0.0142", 0.88, C_GOLD, "Proper scoring rule accuracy", 11.20),
        ("SPARSIFICATION ERROR (AUSE)", 0.0290, "0.0290", 0.85, C_EMERALD, "Uncertainty-error rank correlation", 12.60),
        ("PEAK GPU VRAM FOOTPRINT", 1.74, "< 1.8 GB", 0.44, C_CRIMSON, "FP16 mixed-precision edge inference", 14.50),
    ]
    for gi, (glbl, gnum, gdisp, gfill, gcol, gsub, t_g) in enumerate(gauges):
        gy = 148 + gi * 122
        g_lit = t_local >= t_g
        g_prog = ease_out_cubic((t_local - t_g) / 1.5) if g_lit else 0.0
        draw.rectangle([748, gy, 1226, gy + 106], fill=C_CARD if g_lit else (8, 12, 24), outline=gcol if g_lit else (28, 40, 62), width=2 if g_lit else 1)

        gx_c, gy_c, gr = 802, gy + 53, 36
        draw.ellipse([gx_c - gr, gy_c - gr, gx_c + gr, gy_c + gr], outline=C_BORDER, width=5)
        if g_prog > 0:
            sweep_deg = int(360 * gfill * g_prog)
            draw.arc([gx_c - gr, gy_c - gr, gx_c + gr, gy_c + gr], start=-90, end=-90 + sweep_deg, fill=gcol, width=6)

        draw.text((856, gy + 12), glbl, font=F_MONO_SM, fill=C_WHITE if g_lit else (70, 82, 102))
        draw.text((856, gy + 32), gdisp if g_prog > 0.85 else (f"{gnum * g_prog:.4f}" if g_lit else "---"), font=F_STAT, fill=gcol if g_lit else (55, 68, 86))
        draw.text((856, gy + 74), gsub, font=F_BODY, fill=C_MUTED if g_lit else (55, 68, 86))

    return np.array(im)


def render_scene6_frame(t_local: float, dur: float, caption: str) -> np.ndarray:
    """
    Scene 6 (Act 06): Global Impact Finale — Real 3D Satellite Globe Locked on Borneo & Indo-Pacific!
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(im, draw, t_local + 122.0, "ACT 06 // GLOBAL IMPACT", "READY FOR GLOBAL WILDFIRE & CLIMATE DEPLOYMENT", caption)

    # Keep globe centered on Borneo & Indo-Pacific (114°E .. 126°E) so Borneo is ALWAYS prominently locked!
    rot_lon = 113.97 + 10.0 * math.sin(t_local * 0.45)
    draw_rotating_globe_and_satellite(
        im, draw, t_local + 10.0, cx=900, cy=365, radius=210, rot_lon_deg=rot_lon, scan_progress=1.0, show_arcs=True
    )

    # Left Kinetic Finale Summary
    p_in = ease_out_back(t_local / 1.0)
    lx = int(-380 + 425 * min(1.0, p_in))

    draw_slanted(draw, lx, 90, lx + 420, 122, slant=8, fill=C_CRIMSON)
    draw.text((lx + 16, 94), "/// GREEN SHIFT HACKS // OFFICIAL PITCH", font=F_CARD, fill=C_WHITE)

    draw_slanted(draw, lx, 134, lx + 48, 182, slant=6, fill=(11, 18, 34), outline=C_CRIMSON, width=2, shadow=(3, 3, C_CRIMSON))
    hero_logo = get_watchtower_logo(40)
    im.paste(hero_logo, (lx + 6, 138), hero_logo)
    draw.text((lx + 62, 138), "WATCHTOWER", font=F_HERO, fill=C_CRIMSON)
    draw.text((lx + 60, 136), "WATCHTOWER", font=F_HERO, fill=C_WHITE)
    draw.text((lx, 190), "UNCERTAINTY-CALIBRATED", font=F_TITLE, fill=C_GOLD)
    draw.text((lx, 228), "WILDFIRE & CARBON AI", font=F_TITLE, fill=C_CYAN)

    pillars = [
        ("01 // SMOKE-PENETRATING SWIR AI", "Sees active fire fronts through dense tropical peat smoke.", C_CRIMSON),
        ("02 // BAYESIAN & EXPLAINABLE", "Calibrated MC Dropout uncertainty + Captum Spectral XAI.", C_CYAN),
        ("03 // INSTANT IPCC CARBON AUDIT", "Real-time Above-Ground & Peatland CO2e accounting.", C_EMERALD),
    ]
    for pi, (ptitle, pdesc, pcol) in enumerate(pillars):
        py = 290 + pi * 108
        draw_slanted(draw, lx, py, lx + 555, py + 92, slant=8, fill=C_CARD, outline=pcol, width=2)
        draw.text((lx + 18, py + 12), ptitle, font=F_CARD, fill=pcol)
        draw.text((lx + 18, py + 46), pdesc, font=F_BODY, fill=C_WHITE)

    return np.array(im)


def stream_scene_to_mp4(ff: str, out_mp4: Path, dur: float, render_fn, preview_png_path: Path | None = None):
    """Stream procedural motion-graphic frames directly into an FFmpeg rawvideo pipe (-> 1920x1080 @ 30fps MP4)."""
    n_frames = max(1, int(round(dur * RFPS)))
    cmd = [
        ff,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{RW}x{RH}",
        "-r",
        str(RFPS),
        "-i",
        "-",
        "-vf",
        "scale=1920:1080:flags=bicubic,fps=30",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-threads",
        "2",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mid_frame = int(n_frames * 0.65)
    for fi in range(n_frames):
        t_local = fi / RFPS
        rgb = render_fn(t_local, dur)
        if preview_png_path is not None and fi == mid_frame:
            Image.fromarray(rgb).resize((1920, 1080), Image.Resampling.BICUBIC).save(str(preview_png_path))
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg pipe failed for {out_mp4}")


def build_motion_graphics_pitch():
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    sr = 24000

    # Load real satellite patch assets for motion graphics
    def load_img(name: str, sz: tuple[int, int]) -> Image.Image:
        p = ASSETS_DIR / "pitch_textures" / name
        if not p.is_file():
            p = OUTPUTS_DIR / name
        if p.is_file():
            return Image.open(p).convert("RGB").resize(sz, Image.Resampling.BILINEAR)
        return Image.new("RGB", sz, (20, 30, 48))

    sat_preview = load_img("run_7837_p1_falsecolor.png", (180, 180))
    img_rgb_smoke = load_img("run_7837_p2_natural.png", (520, 520))
    img_swir_fire = load_img("run_7837_p3_segmentation.png", (520, 520))
    img_epistemic = load_img("run_7837_p5_epistemic.png", (520, 520))
    img_seg_sm = load_img("run_7837_p3_segmentation.png", (230, 82))
    img_mi_sm = load_img("run_7837_p5_epistemic.png", (230, 125))

    scenes = [
        (
            "scene1_globe_scan",
            "Every year, catastrophic wildfires destroy millions of hectares of forest and release billions of tonnes of CO2. Meet Watchtower: an AI-driven orbital wildfire and carbon intelligence platform that scans the globe with ten-meter Sentinel-2 satellites to deliver real-time burn segmentation, Bayesian uncertainty, and IPCC Tier-3 carbon audits.",
            "Watchtower scans the globe with 10m Sentinel-2 telemetry for real-time AI wildfire & carbon auditing.",
            lambda t, d, cap: render_scene1_frame(t, d, cap, sat_preview),
            ARTIFACT_DIR / "pitch_mg_scene1_globe.png",
        ),
        (
            "scene2_smoke_cutter",
            "Why does Watchtower matter? First, toxic peatland smoke completely blinds standard optical RGB satellites during active megafires. Second, conventional black-box AI models output uncalibrated masks with zero confidence bounds. And third, manual post-fire carbon emission audits take months to verify.",
            "Solving 3 Blindspots: 2190nm SWIR Smoke Penetration, Bayesian Calibration & Instant Carbon Accounting.",
            lambda t, d, cap: render_scene2_frame(t, d, cap, img_rgb_smoke, img_swir_fire, img_epistemic),
            ARTIFACT_DIR / "pitch_mg_scene2_smokecutter.png",
        ),
        (
            "scene3_neural_pipeline",
            "How does Watchtower work? We fuse twelve Sentinel-2 multispectral bands with four physical burn indices into a sixteen-channel tensor that cuts through smoke, feeding an SSL4EO-pretrained Attention U-Net. Next, Monte Carlo Dropout and Temperature Scaling quantify Bayesian epistemic uncertainty, Captum Integrated Gradients explains top spectral bands, and our Global Forest Watch engine audits CO2 emissions.",
            "Pipeline: 16-Ch Sentinel-2 -> SSL4EO Attention U-Net -> Bayesian MC Dropout -> Captum XAI & GFW Carbon.",
            lambda t, d, cap: render_scene3_frame(t, d, cap, img_seg_sm, img_mi_sm),
            ARTIFACT_DIR / "pitch_mg_scene3_pipeline.png",
        ),
        (
            "scene4_live_demo",
            "Here is Watchtower in action on the Borneo Peatland Megafire. Selecting the hotspot and pressing Analyze triggers our progressive eight-panel scan: revealing Short-Wave Infrared False Color and True Color RGB, tracing the active fire outline and filling the AI burn scar segmentation, mapping Predictive Entropy, Bayesian Epistemic Uncertainty, and Normalized Burn Ratio, animating Captum Spectral XAI bars, and counting up the Global Forest Watch Carbon Audit to two hundred forty-five point four hectares and one hundred eleven thousand nine hundred tonnes of CO2 emissions.",
            "Live System Demo: Progressive 8-Stage Scan on Borneo Peatland (245.4 HA | 111,900 tCO2e).",
            None,
            None,
        ),
        (
            "scene5_tech_constellation",
            "Under the hood, Watchtower is powered by PyTorch, TorchGeo, Captum, FastAPI, and Leaflet GIS. Our Bayesian calibration achieves an Expected Calibration Error of zero point zero zero eight while running under one point eight gigabytes of VRAM.",
            "Tech Constellation: PyTorch, TorchGeo, SSL4EO U-Net, Captum XAI & FastAPI (ECE 0.0080, < 1.8 GB VRAM).",
            lambda t, d, cap: render_scene5_frame(t, d, cap),
            ARTIFACT_DIR / "pitch_mg_scene5_tech.png",
        ),
        (
            "scene6_global_finale",
            "From tropical peatlands in Borneo to Maui and New South Wales, Watchtower empowers rapid wildfire response and transparent climate accounting. Watchtower: uncertainty-calibrated wildfire and carbon intelligence from orbit.",
            "Watchtower: Uncertainty-calibrated wildfire and carbon intelligence from orbit.",
            lambda t, d, cap: render_scene6_frame(t, d, cap),
            ARTIFACT_DIR / "pitch_mg_scene6_globe.png",
        ),
    ]

    print("[1/4] Synthesizing English Pitching Voiceover for all 6 Motion-Graphic Scenes...", flush=True)
    async def synth_vo():
        for sc_id, vo_txt, _, _, _ in scenes:
            mp3_p = OUTPUTS_DIR / f"mg_{sc_id}.mp3"
            if not mp3_p.is_file() or mp3_p.stat().st_size < 1000:
                await edge_tts.Communicate(vo_txt, voice="en-US-ChristopherNeural", rate="+11%").save(str(mp3_p))

    asyncio.run(synth_vo())

    scene_audios = []
    scene_durs = []
    for sc_id, _, _, _, _ in scenes:
        mp3_p = OUTPUTS_DIR / f"mg_{sc_id}.mp3"
        wav_p = OUTPUTS_DIR / f"mg_{sc_id}.wav"
        subprocess.run([ff, "-y", "-i", str(mp3_p), "-ar", str(sr), "-ac", "1", str(wav_p)], capture_output=True, check=True)
        with wave.open(str(wav_p), "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        padded = np.concatenate([np.zeros(int(0.15 * sr), dtype=np.float32), raw, np.zeros(int(0.35 * sr), dtype=np.float32)])
        scene_audios.append(padded)
        scene_durs.append(len(padded) / sr)
        print(f"  - {sc_id}: {len(padded)/sr:.2f}s")

    total_dur = sum(scene_durs)
    print(f"  Total Motion-Graphic Pitch Duration: {total_dur:.2f}s")

    # Build Master Audio (Voiceover + Sci-Fi Orbital Radar & UI Sound Design)
    full_vo = np.concatenate(scene_audios)
    n_samples = len(full_vo)
    sfx = np.zeros(n_samples, dtype=np.float32)

    def add_tone(t_sec: float, d: float, f0: float, f1: float | None = None, amp: float = 0.14):
        i0 = int(t_sec * sr)
        length = int(d * sr)
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
        env = np.exp(-3.0 * t / max(d, 0.01)) * (np.sin(np.pi * t / max(d, 0.01)) ** 0.3)
        sfx[i0:i1] += (np.sin(phase) * env * amp).astype(np.float32)

    cur_t = 0.0
    for idx, (sc_id, _, _, _, _) in enumerate(scenes):
        dur = scene_durs[idx]
        # Scene transition whoosh/chime
        add_tone(cur_t + 0.02, 0.22, 440, 1120, amp=0.16)
        if sc_id == "scene1_globe_scan":
            # "Meet Watchtower" @ 7.42s, "scans the globe" @ 11.99s, KPIs @ 15.05s, 16.70s, 18.20s
            add_tone(cur_t + 7.40, 0.25, 580, 1160, amp=0.16)
            add_tone(cur_t + 11.85, 0.45, 340, 880, amp=0.16)
            add_tone(cur_t + 15.05, 0.16, 740, 1110, amp=0.14)
            add_tone(cur_t + 16.70, 0.16, 880, 1320, amp=0.14)
            add_tone(cur_t + 18.20, 0.16, 990, 1485, amp=0.14)
        elif sc_id == "scene2_smoke_cutter":
            # "First" @ 2.30s, "Second" @ 9.10s, "third" @ 15.87s
            add_tone(cur_t + 2.25, 0.25, 660, 1100, amp=0.18)
            add_tone(cur_t + 9.05, 0.25, 780, 1240, amp=0.18)
            add_tone(cur_t + 15.80, 0.25, 880, 1420, amp=0.18)
        elif sc_id == "scene3_neural_pipeline":
            # Stage 1 @ 2.20s, Stage 2 @ 10.50s, Stage 3 @ 13.40s, Stage 4 @ 18.35s, GFW Carbon @ 21.50s
            add_tone(cur_t + 2.20, 0.22, 620, 1080, amp=0.16)
            add_tone(cur_t + 10.50, 0.22, 740, 1220, amp=0.16)
            add_tone(cur_t + 13.40, 0.22, 860, 1360, amp=0.16)
            add_tone(cur_t + 18.35, 0.22, 980, 1480, amp=0.16)
            for k in range(12):
                add_tone(cur_t + 21.60 + k * 0.14, 0.04, 1380 + (k % 3) * 120, amp=0.10)
        elif sc_id == "scene4_live_demo":
            # Exact WordBoundary-synced SFX for Scene 4 Live Borneo Demo
            add_tone(cur_t + 5.65, 0.14, 520, 1040, amp=0.22)   # "Analyze" @ 5.70s
            add_tone(cur_t + 6.20, 0.65, 280, 620, amp=0.09)    # Map laser sweep
            add_tone(cur_t + 8.65, 0.14, 880, 1320, amp=0.18)   # "Short-Wave Infrared False Color" @ 8.72s
            add_tone(cur_t + 10.40, 0.14, 980, 1480, amp=0.18)  # "True Color RGB" @ 10.47s
            for k in range(12):                                 # "tracing the active fire outline" @ 11.86s
                add_tone(cur_t + 11.85 + k * 0.18, 0.04, 1200 + k * 40, amp=0.11)
            add_tone(cur_t + 16.30, 0.28, 740, 1110, amp=0.15)  # "Predictive Entropy" @ 16.35s
            add_tone(cur_t + 17.35, 0.28, 880, 1320, amp=0.15)  # "Bayesian Epistemic" @ 17.39s
            add_tone(cur_t + 19.45, 0.28, 990, 1485, amp=0.15)  # "Normalized Burn Ratio" @ 19.51s
            for k in range(8):                                  # "animating Captum Spectral XAI" @ 21.04s
                add_tone(cur_t + 21.05 + k * 0.18, 0.07, 580 + k * 95, amp=0.12)
            for k in range(16):                                 # "counting up Global Forest Watch" @ 23.75s
                add_tone(cur_t + 23.80 + k * 0.18, 0.04, 1380 + (k % 3) * 130, amp=0.10)
            add_tone(cur_t + 27.00, 0.85, 523.25, amp=0.12)
            add_tone(cur_t + 27.00, 0.85, 659.25, amp=0.12)
            add_tone(cur_t + 27.00, 0.85, 783.99, amp=0.12)
        elif sc_id == "scene5_tech_constellation":
            for tn_t in [2.30, 3.15, 4.00, 4.70, 5.50, 8.05]:
                add_tone(cur_t + tn_t, 0.12, 820, 1280, amp=0.12)
            for g_t in [9.50, 11.20, 12.60, 14.50]:
                add_tone(cur_t + g_t, 0.20, 680, 1180, amp=0.14)
        cur_t += dur

    # Finale chord
    add_tone(total_dur - 1.1, 0.95, 523.25, amp=0.15)
    add_tone(total_dur - 1.1, 0.95, 659.25, amp=0.15)
    add_tone(total_dur - 1.1, 0.95, 783.99, amp=0.15)
    add_tone(total_dur - 1.1, 1.00, 1046.50, amp=0.15)

    mixed_audio = np.clip(full_vo * 1.25 + sfx * 0.38, -0.98, 0.98)
    master_wav = OUTPUTS_DIR / "mg_master_audio.wav"
    with wave.open(str(master_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((mixed_audio * 32767).astype(np.int16).tobytes())

    print("[2/4] Streaming 6 Procedural Motion-Graphic Scenes to Full-HD (1920x1080 @ 30fps) MP4 segments...")
    seg_mp4s = []
    mg_only_mp4s = []
    mg_only_audios = []

    for idx, (sc_id, _, cap_txt, fn, prev_png) in enumerate(scenes):
        dur = scene_durs[idx]
        seg_mp4 = OUTPUTS_DIR / f"mg_part_{idx:02d}_{sc_id}.mp4"
        seg_mp4s.append(seg_mp4)

        if sc_id != "scene4_live_demo":
            mg_only_mp4s.append(seg_mp4)
            mg_only_audios.append(scene_audios[idx])
            stream_scene_to_mp4(ff, seg_mp4, dur, lambda t, d, _fn=fn, _c=cap_txt: _fn(t, d, _c), prev_png)
        else:
            # Piecewise time-map user's 15.43s screen recording so every panel unlock hits the exact spoken WordBoundary!
            vo_pts = [0.00, 5.65, 8.65, 10.40, 11.80, 16.25, 17.35, 19.45, 21.00, 23.70, 27.50, dur]
            vid_pts = [0.00, 0.25, 6.05, 6.80, 7.55, 8.95, 9.65, 10.35, 11.05, 11.95, 13.30, 15.40]

            import imageio
            reader = imageio.get_reader(str(USER_RECORDING_PATH))
            src_fps = 30.0
            out_fps = 20
            n_out = max(1, int(round(dur * out_fps)))
            cmd_demo = [
                ff,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1918x1014",
                "-r",
                str(out_fps),
                "-i",
                "-",
                "-vf",
                "scale=1920:1014,pad=1920:1080:0:33:color=#040711,fps=30",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-threads",
                "2",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(seg_mp4),
            ]
            proc_d = subprocess.Popen(cmd_demo, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cur_src_idx = 0
            cur_frame = reader.get_data(0)
            for fo in range(n_out):
                t_vo = fo / out_fps
                t_vid = float(np.interp(t_vo, vo_pts, vid_pts))
                target_idx = int(round(t_vid * src_fps))
                while cur_src_idx < target_idx:
                    try:
                        cur_frame = reader.get_next_data()
                        cur_src_idx += 1
                    except Exception:
                        break
                proc_d.stdin.write(cur_frame.tobytes())
            proc_d.stdin.close()
            proc_d.wait()
            reader.close()
        print(f"  - Rendered motion-graphic scene {idx+1}/{len(scenes)}: {sc_id} ({dur:.2f}s)", flush=True)

    # Concat Full Motion-Graphic Pitch + Live Demo
    concat_full = OUTPUTS_DIR / "mg_concat_full.txt"
    concat_full.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_mp4s), encoding="utf-8")

    final_pitch_mp4 = OUTPUTS_DIR / "watchtower_pitch_video.mp4"
    print("[3/4] Muxing Full Motion-Graphic Pitch Video -> outputs/watchtower_pitch_video.mp4...")
    subprocess.run(
        [
            ff,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_full),
            "-i",
            str(master_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(final_pitch_mp4),
        ],
        capture_output=True,
        check=True,
    )

    # Also build Motion-Graphics Only version (without Scene 4 screen recording)
    concat_mg = OUTPUTS_DIR / "mg_concat_only.txt"
    concat_mg.write_text("\n".join(f"file '{p.as_posix()}'" for p in mg_only_mp4s), encoding="utf-8")
    mg_vo = np.clip(np.concatenate(mg_only_audios) * 1.25, -0.98, 0.98)
    mg_wav = OUTPUTS_DIR / "mg_only_audio.wav"
    with wave.open(str(mg_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((mg_vo * 32767).astype(np.int16).tobytes())

    mg_only_mp4 = OUTPUTS_DIR / "watchtower_pitch_motion_only.mp4"
    subprocess.run(
        [
            ff,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_mg),
            "-i",
            str(mg_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(mg_only_mp4),
        ],
        capture_output=True,
        check=True,
    )

    shutil.copyfile(str(final_pitch_mp4), str(ARTIFACT_DIR / "watchtower_pitch_video.mp4"))

    for p in OUTPUTS_DIR.glob("mg_*"):
        if p.is_file():
            p.unlink()

    print(f"[4/4] SUCCESS! Saved Motion-Graphic Pitch Video: {final_pitch_mp4} ({final_pitch_mp4.stat().st_size/(1024*1024):.2f} MB)")
    print(f"      Saved Pure Motion-Graphic Intro (without demo): {mg_only_mp4} ({mg_only_mp4.stat().st_size/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_motion_graphics_pitch()

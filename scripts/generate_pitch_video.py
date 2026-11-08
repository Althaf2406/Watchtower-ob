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
ARTIFACT_DIR = Path(r"C:\Users\altha\.gemini\antigravity\brain\a59a8596-90f0-416d-b572-bff212dd9ba5")

# Render canvas at 1280x720 @ 20fps for fast procedural drawing, upscaled by FFmpeg to 1920x1080 @ 30fps
RW, RH = 1280, 720
RFPS = 20


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
    ("BORNEO PEATLAND", -1.84, 113.97, C_CRIMSON),
    ("NSW AUSTRALIA", -33.67, 150.36, C_GOLD),
    ("LAHAINA, MAUI", 20.88, -156.68, C_CYAN),
]


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def ease_out_back(x: float) -> float:
    x = max(0.0, min(1.0, x))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((x - 1.0) ** 3) + c1 * ((x - 1.0) ** 2)


def draw_slanted(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, slant=10, fill=None, outline=None, width=2):
    poly = [(x0 + slant, y0), (x1 + slant, y0), (x1, y1), (x0, y1)]
    draw.polygon(poly, fill=fill, outline=outline, width=width)


def draw_motion_hud_shell(draw: ImageDraw.ImageDraw, t: float, badge: str, title: str, caption: str):
    """Animated starfield, moving grid, corner radar reticles, top command bar, and bottom audio waveform + caption."""
    # Moving cyber grid
    offset_x = int((t * 28) % 64)
    offset_y = int((t * 14) % 64)
    for gx in range(-64 + offset_x, RW, 64):
        draw.line([(gx, 54), (gx, RH - 44)], fill=(9, 16, 32), width=1)
    for gy in range(54 + offset_y, RH - 44, 64):
        draw.line([(0, gy), (RW, gy)], fill=(9, 16, 32), width=1)

    # Twinkling drifting stars
    for sx, sy, sbr, sspd in STARS:
        px = int((sx - t * 18 * sspd) % RW)
        twinkle = int(sbr * (0.65 + 0.35 * math.sin(t * 4.0 + sx)))
        draw.rectangle([px, sy, px + 1, sy + 1], fill=(twinkle, twinkle, min(255, twinkle + 30)))

    # Animated diagonal Atlus hazard stripes at top-right
    stripe_shift = int((t * 40) % 40)
    for k in range(-2, 8):
        x_s = RW - 260 + k * 40 + stripe_shift
        draw.polygon([(x_s, 0), (x_s + 18, 0), (x_s - 42, 54), (x_s - 60, 54)], fill=(30, 12, 28))

    # Top Header Bar
    draw.rectangle([0, 0, RW, 54], fill=(8, 14, 28))
    draw.line([(0, 54), (RW, 54)], fill=C_BORDER, width=2)
    draw_slanted(draw, 18, 8, 54, 44, slant=6, fill=C_CRIMSON, outline=C_WHITE, width=2)
    draw.text((28, 8), "W", font=F_TITLE, fill=C_WHITE)
    draw.text((66, 9), "WATCHTOWER", font=F_TITLE, fill=C_WHITE)

    draw_slanted(draw, 250, 12, 445, 40, slant=6, fill=C_CYAN)
    draw.text((262, 15), badge, font=F_CARD, fill=C_BG)
    draw.text((462, 14), title, font=F_SUB, fill=C_WHITE)

    # Live ticking orbital clock on top-right
    draw.text((RW - 235, 19), f"SAT-S2B // T+{t:05.2f}s // ORBIT LOCK", font=F_MONO_SM, fill=C_CYAN)

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
    """Project (lat, lon) onto 2D screen with 3D rotation around Y (longitude spin) and X (axial tilt)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg - rot_lon_deg)
    tilt = math.radians(tilt_deg)

    # 3D unit sphere coordinates
    x = math.cos(lat) * math.sin(lon)
    y = math.sin(lat)
    z = math.cos(lat) * math.cos(lon)

    # Tilt around X axis
    y2 = y * math.cos(tilt) - z * math.sin(tilt)
    z2 = y * math.sin(tilt) + z * math.cos(tilt)

    sx = int(cx + radius * x)
    sy = int(cy - radius * y2)
    return sx, sy, z2


def draw_rotating_globe_and_satellite(
    draw: ImageDraw.ImageDraw,
    t: float,
    cx: int,
    cy: int,
    radius: int,
    rot_lon_deg: float,
    scan_progress: float,
    show_arcs: bool = False,
):
    """Draw a 3D wireframe + continent globe with an orbiting Sentinel-2 satellite firing a laser scan cone."""
    # Outer orbital atmosphere glow rings
    for r_off, col in [(radius + 22, (0, 65, 95)), (radius + 10, (0, 120, 165)), (radius + 3, C_CYAN)]:
        draw.ellipse([cx - r_off, cy - r_off, cx + r_off, cy + r_off], outline=col, width=1)

    # Globe sphere body
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(8, 20, 38), outline=C_CYAN, width=2)

    # Latitude & Longitude 3D grid lines
    for lat_g in range(-60, 75, 30):
        pts = []
        for lon_g in range(-180, 185, 8):
            sx, sy, z = project_3d_globe(lat_g, lon_g, rot_lon_deg, 16.0, cx, cy, radius)
            if z > 0:
                pts.append((sx, sy))
            else:
                if len(pts) >= 2:
                    draw.line(pts, fill=(18, 45, 78), width=1)
                pts = []
        if len(pts) >= 2:
            draw.line(pts, fill=(18, 45, 78), width=1)

    for lon_g in range(-180, 180, 30):
        pts = []
        for lat_g in range(-80, 85, 8):
            sx, sy, z = project_3d_globe(lat_g, lon_g, rot_lon_deg, 16.0, cx, cy, radius)
            if z > 0:
                pts.append((sx, sy))
            else:
                if len(pts) >= 2:
                    draw.line(pts, fill=(18, 45, 78), width=1)
                pts = []
        if len(pts) >= 2:
            draw.line(pts, fill=(18, 45, 78), width=1)

    # 3D Continents on visible hemisphere
    for poly in CONTINENT_POLYS:
        vis_pts = []
        for plat, plon in poly:
            sx, sy, z = project_3d_globe(plat, plon, rot_lon_deg, 16.0, cx, cy, radius)
            if z > -0.05:
                vis_pts.append((sx, sy))
        if len(vis_pts) >= 3:
            draw.polygon(vis_pts, fill=(16, 62, 58), outline=(0, 230, 118))

    # Pulsing Wildfire Hotspots on Globe
    visible_hotspots = []
    for hname, hlat, hlon, hcol in HOTSPOT_COORDS:
        sx, sy, z = project_3d_globe(hlat, hlon, rot_lon_deg, 16.0, cx, cy, radius)
        if z > 0.05:
            visible_hotspots.append((hname, sx, sy, hcol))
            pulse_r = int(6 + 10 * ((t * 2.8) % 1.0))
            draw.ellipse([sx - pulse_r, sy - pulse_r, sx + pulse_r, sy + pulse_r], outline=hcol, width=2)
            draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=C_WHITE, outline=hcol, width=2)
            draw.line([(sx + 8, sy - 8), (sx + 28, sy - 22), (sx + 135, sy - 22)], fill=hcol, width=2)
            draw.rectangle([sx + 28, sy - 40, sx + 145, sy - 22], fill=(6, 10, 22), outline=hcol, width=1)
            draw.text((sx + 33, sy - 37), hname, font=F_MONO_SM, fill=C_WHITE)

    # Telemetry arcs between hotspots if requested
    if show_arcs and len(visible_hotspots) >= 2:
        for i in range(len(visible_hotspots) - 1):
            _, x1, y1, _ = visible_hotspots[i]
            _, x2, y2, _ = visible_hotspots[i + 1]
            mx, my = (x1 + x2) // 2, min(y1, y2) - 45
            arc_pts = []
            for s in range(21):
                u = s / 20.0
                px = int((1 - u) ** 2 * x1 + 2 * (1 - u) * u * mx + u ** 2 * x2)
                py = int((1 - u) ** 2 * y1 + 2 * (1 - u) * u * my + u ** 2 * y2)
                arc_pts.append((px, py))
            draw.line(arc_pts, fill=C_CYAN, width=2)

    # Orbiting Sentinel-2 Satellite + Laser Scan Cone targeting Borneo / Globe center
    orbit_angle = t * 1.15
    sat_x = int(cx + (radius + 68) * math.cos(orbit_angle - 0.8))
    sat_y = int(cy - (radius + 42) * math.sin(orbit_angle - 0.8) * 0.65 - 55)

    # Target point on globe (Borneo if visible, else globe center)
    tx, ty = (visible_hotspots[0][1], visible_hotspots[0][2]) if visible_hotspots else (cx, cy)

    if scan_progress > 0:
        # Laser scan cone from satellite to target hotspot
        spread = 32
        cone_poly = [(sat_x, sat_y), (tx - spread, ty - 12), (tx + spread, ty + 12)]
        draw.polygon(cone_poly, outline=C_CYAN, width=1)
        # Animated laser scan rungs travelling down the cone
        for k in range(5):
            prog = ((t * 2.5 + k * 0.2) % 1.0)
            lx0 = int(sat_x + (tx - spread - sat_x) * prog)
            ly0 = int(sat_y + (ty - 12 - sat_y) * prog)
            lx1 = int(sat_x + (tx + spread - sat_x) * prog)
            ly1 = int(sat_y + (ty + 12 - sat_y) * prog)
            draw.line([(lx0, ly0), (lx1, ly1)], fill=C_CYAN if k % 2 == 0 else C_CRIMSON, width=2)

        # Expanding scan square on the target hotspot
        sq_r = int(24 + 8 * math.sin(t * 6.0))
        draw.rectangle([tx - sq_r, ty - sq_r, tx + sq_r, ty + sq_r], outline=C_CYAN, width=2)
        scan_y = int(ty - sq_r + ((t * 3.0) % 1.0) * (2 * sq_r))
        draw.line([(tx - sq_r, scan_y), (tx + sq_r, scan_y)], fill=C_CRIMSON, width=3)

    # Draw Sentinel-2 Satellite Body & Solar Wings
    # Left & Right Solar Arrays
    draw.polygon([(sat_x - 52, sat_y - 14), (sat_x - 16, sat_y - 6), (sat_x - 16, sat_y + 6), (sat_x - 52, sat_y + 14)], fill=(18, 75, 130), outline=C_CYAN, width=2)
    draw.polygon([(sat_x + 16, sat_y - 6), (sat_x + 52, sat_y - 14), (sat_x + 52, sat_y + 14), (sat_x + 16, sat_y + 6)], fill=(18, 75, 130), outline=C_CYAN, width=2)
    # Central Bus & Multispectral Instrument Payload
    draw.rectangle([sat_x - 14, sat_y - 12, sat_x + 14, sat_y + 12], fill=C_GOLD, outline=C_WHITE, width=2)
    draw.ellipse([sat_x - 6, sat_y - 6, sat_x + 6, sat_y + 6], fill=C_CRIMSON, outline=C_WHITE, width=1)
    draw.text((sat_x - 42, sat_y - 28), "SENTINEL-2 L2A (10M)", font=F_MONO_SM, fill=C_CYAN)


def render_scene1_frame(t_local: float, dur: float, caption: str, sat_preview: Image.Image) -> np.ndarray:
    """
    Scene 1 (Act 01): Orbital Satellite Scanning 3D Globe -> Kinetic Reveal of Watchtower.
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(draw, t_local, "ACT 01 // ORBITAL AI", "WHAT IS WATCHTOWER?", caption)

    # Rotate globe smoothly toward Borneo (lon ~114 deg)
    rot_lon = 45.0 + min(69.0, t_local * 7.5) + math.sin(t_local * 0.7) * 6.0
    scan_p = ease_out_cubic((t_local - 1.2) / 2.5)
    draw_rotating_globe_and_satellite(draw, t_local, cx=915, cy=375, radius=215, rot_lon_deg=rot_lon, scan_progress=scan_p)

    # Kinetic Left Title Reveal (slides in with ease_out_back)
    slide_in = ease_out_back(t_local / 1.1)
    lx = int(-420 + 465 * slide_in)

    draw_slanted(draw, lx, 82, lx + 310, 112, slant=8, fill=C_CRIMSON)
    draw.text((lx + 14, 86), "/// ORBITAL WILDFIRE & CARBON AI", font=F_CARD, fill=C_WHITE)

    draw.text((lx, 124), "REAL-TIME WILDFIRE", font=F_HERO, fill=C_WHITE)
    draw.text((lx, 174), "& CARBON INTELLIGENCE", font=F_HERO, fill=C_GOLD)
    draw.text((lx, 224), "FROM SENTINEL-2 ORBIT", font=F_HERO, fill=C_CYAN)

    # Animated Satellite Patch Zoom-Out Card emerging from the Globe Scan!
    if t_local > 2.2:
        pop_p = ease_out_back((t_local - 2.2) / 1.0)
        card_w = int(540 * min(1.0, pop_p))
        if card_w > 80:
            # Draw laser tether lines from Borneo on globe (915, 375) to the left telemetry deck
            draw.line([(915, 375), (lx + 540, 340)], fill=C_CYAN, width=1)
            draw.line([(915, 375), (lx + 540, 495)], fill=C_CYAN, width=1)

            draw.rectangle([lx, 295, lx + 545, 505], fill=C_CARD, outline=C_CYAN, width=2)
            im.paste(sat_preview, (lx + 12, 307))
            draw.rectangle([lx + 12, 307, lx + 192, 487], outline=C_CRIMSON, width=2)
            # Sweeping scanline inside preview patch
            sy = 307 + int(((t_local * 90) % 180))
            draw.line([(lx + 12, sy), (lx + 192, sy)], fill=C_CYAN, width=2)

            draw.text((lx + 208, 312), "TARGET LOCKED: BORNEO PEATLAND", font=F_CARD, fill=C_CRIMSON)
            draw.text((lx + 208, 340), "• 12 Sentinel-2 Bands + 4 Burn Indices", font=F_BODY, fill=C_WHITE)
            draw.text((lx + 208, 366), "• Smoke-Penetrating SWIR (2190nm @ 10m)", font=F_BODY, fill=C_GOLD)
            draw.text((lx + 208, 392), "• Attention U-Net + Bayesian MC Dropout", font=F_BODY, fill=C_CYAN)
            draw.text((lx + 208, 418), "• Captum Spectral XAI + IPCC Tier-3 CO2e", font=F_BODY, fill=C_EMERALD)

            # Live ticking CO2 counter inside card
            co2_val = int(min(111900, (t_local - 2.2) * 18500))
            draw.rectangle([lx + 208, 448, lx + 528, 490], fill=C_BG, outline=C_GOLD, width=1)
            draw.text((lx + 218, 458), f"AUDITED EMISSIONS: {co2_val:,} tCO2e", font=F_MONO_LG, fill=C_GOLD)

    # 3 Animated KPI Counters at Bottom-Left
    kpis = [
        ("16 CHANNELS", "Multispectral + Burn Physics", C_CYAN, 0.8),
        ("10M PIXELS", "Cuts Through Peat Smoke", C_GOLD, 1.5),
        ("< 3 SECONDS", "End-to-End Carbon Audit", C_CRIMSON, 2.2),
    ]
    for ki, (kval, klbl, kcol, kdelay) in enumerate(kpis):
        if t_local > kdelay:
            kp = ease_out_back((t_local - kdelay) / 0.8)
            kx = 45 + ki * 186
            ky = int(680 - 155 * min(1.0, kp))
            draw_slanted(draw, kx, ky, kx + 172, ky + 110, slant=8, fill=C_PANEL, outline=kcol, width=2)
            draw.text((kx + 14, ky + 16), kval, font=F_SUB, fill=kcol)
            draw.text((kx + 14, ky + 56), klbl, font=F_MONO_SM, fill=C_WHITE)

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
    draw_motion_hud_shell(draw, t_local + 22.0, "ACT 02 // WHY IT MATTERS", "SOLVING 3 CRITICAL WILDFIRE & CARBON BLINDSPOTS", caption)

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
    cards = [
        (
            0.2,
            "GAP 01 // TOXIC SMOKE BLINDNESS",
            "Standard RGB satellites see only white smoke during peatland megafires.",
            "WATCHTOWER FIX: 16-Channel SWIR (2190nm) + Active Fire ratio cuts through dense smoke.",
            C_CRIMSON,
        ),
        (
            5.5,
            "GAP 02 // BLACK-BOX AI HALLUCINATIONS",
            "Deterministic U-Nets output uncalibrated masks with zero confidence bounds.",
            "WATCHTOWER FIX: Bayesian MC Dropout (T=20) + Temperature Scaling maps Epistemic MI.",
            C_CYAN,
        ),
        (
            11.0,
            "GAP 03 // MONTHS-LONG CARBON DELAYS",
            "Manual post-fire forest & peatland emission reporting takes months to audit.",
            "WATCHTOWER FIX: Automated GFW Above-Ground + Peat Carbon audit in 3 confidence regimes.",
            C_GOLD,
        ),
    ]

    for idx, (t_appear, ctitle, cprob, cfix, ccol) in enumerate(cards):
        cy0 = 82 + idx * 188
        active_p = ease_out_back((t_local - t_appear) / 0.9) if t_local >= t_appear else 0.15
        cx0 = int(600 + (1.0 - min(1.0, active_p)) * 120)
        is_lit = t_local >= t_appear

        draw.rectangle([cx0, cy0, 1245, cy0 + 170], fill=C_CARD if is_lit else C_PANEL, outline=ccol if is_lit else C_BORDER, width=2)
        draw_slanted(draw, cx0 + 14, cy0 + 12, cx0 + 370, cy0 + 42, slant=8, fill=ccol if is_lit else C_BORDER)
        draw.text((cx0 + 26, cy0 + 16), ctitle, font=F_CARD, fill=C_BG if is_lit else C_MUTED)

        draw.text((cx0 + 18, cy0 + 56), f"PROBLEM: {cprob}", font=F_BODY, fill=C_MUTED if is_lit else (90, 100, 120))
        draw.text((cx0 + 18, cy0 + 88), cfix, font=F_BODY, fill=ccol if is_lit else (90, 100, 120))

        # Animated progress/confidence meter inside each card
        bar_w = int(600 * min(1.0, max(0.0, (t_local - t_appear) / 2.2)))
        draw.rectangle([cx0 + 18, cy0 + 128, cx0 + 618, cy0 + 148], fill=C_BG, outline=C_BORDER, width=1)
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
    draw_motion_hud_shell(draw, t_local + 44.0, "ACT 03 // HOW IT WORKS", "4-STAGE ORBITAL AI & BAYESIAN CARBON ARCHITECTURE", caption)

    # 4 Animated Pipeline Nodes across the screen
    node_specs = [
        (40, 310, "01 // 16-CH INGESTION", "12 Sentinel-2 + 4 Indices", C_GOLD),
        (350, 620, "02 // ATTENTION U-NET", "SSL4EO-S12 ResNet-50", C_CRIMSON),
        (660, 930, "03 // BAYESIAN MC UQ", "T=20 Dropout + Temp Scale", C_CYAN),
        (970, 1240, "04 // XAI & IPCC CARBON", "Captum IG + GFW Tier-3", C_EMERALD),
    ]

    # Animated data-stream conduits connecting the 4 stages
    for i in range(3):
        x_start = node_specs[i][1]
        x_end = node_specs[i + 1][0]
        y_mid = 335
        draw.line([(x_start, y_mid), (x_end, y_mid)], fill=C_BORDER, width=6)
        # Flowing glowing data packets along the conduit
        p_pos = (t_local * 2.4 + i * 0.3) % 1.0
        px = int(x_start + (x_end - x_start) * p_pos)
        draw.ellipse([px - 7, y_mid - 7, px + 7, y_mid + 7], fill=C_CYAN, outline=C_WHITE, width=2)

    for idx, (nx0, nx1, ntitle, nsub, ncol) in enumerate(node_specs):
        ny0, ny1 = 86, 565
        draw.rectangle([nx0, ny0, nx1, ny1], fill=C_CARD, outline=ncol, width=2)
        draw_slanted(draw, nx0 + 10, ny0 + 10, nx1 - 18, ny0 + 38, slant=6, fill=ncol)
        draw.text((nx0 + 18, ny0 + 14), ntitle, font=F_CARD, fill=C_BG)
        draw.text((nx0 + 14, ny0 + 48), nsub, font=F_MONO_SM, fill=C_WHITE)
        draw.line([(nx0 + 12, ny0 + 70), (nx1 - 12, ny0 + 70)], fill=C_BORDER, width=1)

        if idx == 0:
            # Stage 1 Motion Graphic: 12-Band + 4-Index Oscillating Spectral Sine Waves!
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
                draw.text((nx0 + 14, by - 12), blbl, font=F_MONO_SM, fill=C_WHITE)
                wave_pts = []
                for wx in range(nx0 + 14, nx1 - 14, 6):
                    wy = by + 12 + int(7 * math.sin(t_local * 5.0 + wx * 0.08 + bi))
                    wave_pts.append((wx, wy))
                draw.line(wave_pts, fill=bcol, width=2)

        elif idx == 1:
            # Stage 2 Motion Graphic: Animated U-Net Encoder-Decoder Feature Maps + Attention Gates!
            layers = [(90, 26), (70, 36), (48, 48), (32, 62), (48, 48), (70, 36), (90, 26)]
            cx_u = (nx0 + nx1) // 2
            for li, (lw, lh) in enumerate(layers):
                ly = ny0 + 88 + li * 42
                pulse = int(4 * math.sin(t_local * 6.0 - li * 0.7))
                lcol = C_CRIMSON if li >= 3 else C_GOLD
                draw.rectangle([cx_u - lw - pulse, ly, cx_u + lw + pulse, ly + 28], fill=C_BG, outline=lcol, width=2)
                # Animated skip-connection attention arcs
                if li < 3:
                    target_y = ny0 + 88 + (6 - li) * 42 + 14
                    draw.arc([cx_u + lw + 4, ly + 10, cx_u + 118, target_y], start=-90, end=90, fill=C_CYAN, width=2)
                    att_y = int(ly + 14 + ((t_local * 1.8 + li * 0.3) % 1.0) * (target_y - ly - 14))
                    draw.ellipse([cx_u + 105, att_y - 4, cx_u + 113, att_y + 4], fill=C_WHITE, outline=C_CYAN)

            im.paste(img_seg.resize((230, 82)), (nx0 + 20, ny1 - 96))
            draw.rectangle([nx0 + 20, ny1 - 96, nx0 + 250, ny1 - 14], outline=C_CRIMSON, width=2)

        elif idx == 2:
            # Stage 3 Motion Graphic: Monte Carlo Stochastic Forward Passes (T=20) + Bayesian Calibration Curve!
            draw.text((nx0 + 14, ny0 + 82), "MC DROPOUT POSTERIOR SAMPLING:", font=F_MONO_SM, fill=C_CYAN)
            for mc_i in range(6):
                my = ny0 + 112 + mc_i * 26
                prog = ((t_local * 2.2 + mc_i * 0.17) % 1.0)
                mw = int(235 * prog)
                draw.rectangle([nx0 + 14, my, nx1 - 14, my + 16], fill=C_BG, outline=C_BORDER, width=1)
                draw.rectangle([nx0 + 14, my, nx0 + 14 + mw, my + 16], fill=C_CYAN if mc_i % 2 == 0 else C_GOLD)
                draw.text((nx0 + 20, my + 2), f"PASS T={mc_i*3+2:02d} // p(y|x,w_{mc_i+1})", font=F_MONO_SM, fill=C_WHITE)

            im.paste(img_mi.resize((230, 125)), (nx0 + 20, ny0 + 280))
            draw.rectangle([nx0 + 20, ny0 + 280, nx0 + 250, ny0 + 405], outline=C_CYAN, width=2)
            draw.text((nx0 + 20, ny0 + 416), "CALIBRATED ECE: 0.0080", font=F_MONO_MD, fill=C_EMERALD)
            draw.text((nx0 + 20, ny0 + 438), "BRIER: 0.0142 | AUSE: 0.0290", font=F_MONO_SM, fill=C_WHITE)

        elif idx == 3:
            # Stage 4 Motion Graphic: Animated Captum XAI Bars + Ticking IPCC Carbon Counter!
            draw.text((nx0 + 14, ny0 + 82), "CAPTUM INTEGRATED GRADIENTS:", font=F_MONO_SM, fill=C_GOLD)
            xai_bars = [("B12_SWIR2", 0.95, C_CRIMSON), ("NBR_Index", 0.86, C_ORANGE), ("AF_SWIR", 0.78, C_GOLD), ("dNBR", 0.68, C_CYAN), ("B11_SWIR1", 0.60, C_EMERALD)]
            for xi, (xlbl, xval, xcol) in enumerate(xai_bars):
                xy = ny0 + 106 + xi * 32
                anim_w = int(155 * xval * (0.75 + 0.25 * math.sin(t_local * 3.5 + xi)))
                draw.text((nx0 + 14, xy), xlbl, font=F_MONO_SM, fill=C_WHITE)
                draw.rectangle([nx0 + 92, xy + 2, nx1 - 14, xy + 16], fill=C_BG, outline=C_BORDER, width=1)
                draw.rectangle([nx0 + 92, xy + 2, nx0 + 92 + anim_w, xy + 16], fill=xcol)

            # Live Ticking IPCC Tier-3 Carbon Counter Box
            draw.rectangle([nx0 + 14, ny0 + 280, nx1 - 14, ny1 - 14], fill=C_BG, outline=C_EMERALD, width=2)
            draw.text((nx0 + 24, ny0 + 292), "GFW IPCC TIER-3 AUDIT", font=F_MONO_MD, fill=C_EMERALD)
            cyc = min(1.0, (t_local % 6.5) / 4.5)
            cur_ha = 245.4 * ease_out_cubic(cyc)
            cur_co2 = int(111900 * ease_out_cubic(cyc))
            draw.text((nx0 + 24, ny0 + 320), f"{cur_ha:,.1f} HA", font=F_STAT, fill=C_GOLD)
            draw.text((nx0 + 24, ny0 + 362), f"{cur_co2:,} tCO2e", font=F_STAT, fill=C_CRIMSON)
            draw.text((nx0 + 24, ny0 + 412), "• Conservative: 90,707 tCO2e", font=F_MONO_SM, fill=C_EMERALD)
            draw.text((nx0 + 24, ny0 + 432), "• Nominal Mean: 111,900 tCO2e", font=F_MONO_SM, fill=C_GOLD)
            draw.text((nx0 + 24, ny0 + 452), "• 95% CI Upper: 129,858 tCO2e", font=F_MONO_SM, fill=C_CRIMSON)

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
    draw_motion_hud_shell(draw, t_local + 105.0, "ACT 05 // TECH STACK", "TECHNOLOGY CONSTELLATION & BENCHMARK VALIDATION", caption)

    # Left: Orbital Tech-Stack Constellation (x=40..720, y=80..655)
    cx, cy = 380, 365
    for r_orb in (95, 175, 245):
        draw.ellipse([cx - r_orb, cy - r_orb, cx + r_orb, cy + r_orb], outline=(20, 38, 68), width=1)

    tech_nodes = [
        ("PYTORCH 2.x + TORCHGEO", "16-Ch Tensor Engine", C_CRIMSON, 0.0),
        ("SSL4EO ATTENTION U-NET", "Self-Supervised R-50", C_GOLD, math.pi / 3),
        ("BAYESIAN MC DROPOUT", "T=20 + Temp Scaling", C_CYAN, 2 * math.pi / 3),
        ("CAPTUM INTEGRATED GRAD", "16-Band Spectral XAI", C_EMERALD, math.pi),
        ("GLOBAL FOREST WATCH", "IPCC Tier-3 Biomass", C_ORANGE, 4 * math.pi / 3),
        ("FASTAPI + LEAFLET GIS", "Real-Time Web HUD", C_CYAN, 5 * math.pi / 3),
    ]

    for ni, (ntitle, nsub, ncol, base_ang) in enumerate(tech_nodes):
        ang = base_ang + t_local * 0.28
        nx = int(cx + 215 * math.cos(ang))
        ny = int(cy + 185 * math.sin(ang))

        # Pulsing data tether from center core to node
        draw.line([(cx, cy), (nx, ny)], fill=ncol, width=2)
        pkt_p = (t_local * 1.8 + ni * 0.2) % 1.0
        px = int(cx + (nx - cx) * pkt_p)
        py = int(cy + (ny - cy) * pkt_p)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=C_WHITE, outline=ncol, width=2)

        draw.rectangle([nx - 105, ny - 32, nx + 105, ny + 32], fill=C_CARD, outline=ncol, width=2)
        draw.text((nx - 95, ny - 22), ntitle, font=F_MONO_SM, fill=ncol)
        draw.text((nx - 95, ny + 2), nsub, font=F_BODY, fill=C_WHITE)

    # Central Watchtower AI Core
    pulse_c = int(8 * math.sin(t_local * 5.0))
    draw.ellipse([cx - 62 - pulse_c, cy - 62 - pulse_c, cx + 62 + pulse_c, cy + 62 + pulse_c], fill=C_PANEL, outline=C_CRIMSON, width=3)
    draw.text((cx - 46, cy - 18), "WATCHTOWER", font=F_CARD, fill=C_WHITE)
    draw.text((cx - 32, cy + 4), "AI CORE", font=F_MONO_SM, fill=C_CYAN)

    # Right: Animated Radial & Linear Scientific Benchmark Gauges (x=730..1245)
    draw.rectangle([730, 82, 1245, 655], fill=C_PANEL, outline=C_BORDER, width=2)
    draw_slanted(draw, 748, 98, 1185, 128, slant=8, fill=C_CRIMSON)
    draw.text((762, 102), "/// VERIFIED SCIENTIFIC BENCHMARKS", font=F_CARD, fill=C_WHITE)

    gauges = [
        ("EXPECTED CALIBRATION ERROR (ECE)", 0.0080, "0.0080", 0.92, C_CYAN, "Post-hoc Temperature Scaling (T_opt)"),
        ("CALIBRATED BRIER SCORE", 0.0142, "0.0142", 0.88, C_GOLD, "Proper scoring rule accuracy"),
        ("SPARSIFICATION ERROR (AUSE)", 0.0290, "0.0290", 0.85, C_EMERALD, "Uncertainty-error rank correlation"),
        ("PEAK GPU VRAM FOOTPRINT", 1.74, "< 1.8 GB", 0.44, C_CRIMSON, "FP16 mixed-precision edge inference"),
    ]
    g_prog = ease_out_cubic(t_local / 2.5)
    for gi, (glbl, gnum, gdisp, gfill, gcol, gsub) in enumerate(gauges):
        gy = 148 + gi * 122
        draw.rectangle([748, gy, 1226, gy + 106], fill=C_CARD, outline=gcol, width=2)

        # Animated Radial Arc Gauge on the Left of the Box
        gx_c, gy_c, gr = 802, gy + 53, 36
        draw.ellipse([gx_c - gr, gy_c - gr, gx_c + gr, gy_c + gr], outline=C_BORDER, width=5)
        sweep_deg = int(360 * gfill * g_prog)
        draw.arc([gx_c - gr, gy_c - gr, gx_c + gr, gy_c + gr], start=-90, end=-90 + sweep_deg, fill=gcol, width=6)

        draw.text((856, gy + 12), glbl, font=F_MONO_SM, fill=C_WHITE)
        draw.text((856, gy + 32), gdisp if g_prog > 0.85 else f"{gnum * g_prog:.4f}", font=F_STAT, fill=gcol)
        draw.text((856, gy + 74), gsub, font=F_BODY, fill=C_MUTED)

    return np.array(im)


def render_scene6_frame(t_local: float, dur: float, caption: str) -> np.ndarray:
    """
    Scene 6 (Act 06): Global Impact Finale — 3D Globe with Intercontinental Telemetry Arcs + Kinetic Closing Banner!
    """
    im = Image.new("RGB", (RW, RH), C_BG)
    draw = ImageDraw.Draw(im)
    draw_motion_hud_shell(draw, t_local + 122.0, "ACT 06 // GLOBAL IMPACT", "READY FOR GLOBAL WILDFIRE & CLIMATE DEPLOYMENT", caption)

    # Center-Right Rotating Globe with Intercontinental Telemetry Arcs
    rot_lon = 115.0 + t_local * 12.0
    draw_rotating_globe_and_satellite(draw, t_local + 10.0, cx=900, cy=365, radius=210, rot_lon_deg=rot_lon, scan_progress=1.0, show_arcs=True)

    # Left Kinetic Finale Summary
    p_in = ease_out_back(t_local / 1.0)
    lx = int(-380 + 425 * min(1.0, p_in))

    draw_slanted(draw, lx, 90, lx + 420, 122, slant=8, fill=C_CRIMSON)
    draw.text((lx + 16, 94), "/// GREEN SHIFT HACKS // OFFICIAL PITCH", font=F_CARD, fill=C_WHITE)

    draw.text((lx, 138), "WATCHTOWER", font=F_HERO, fill=C_WHITE)
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
        "ultrafast",
        "-threads",
        "2",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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
            None,
        ),
    ]

    print("[1/4] Synthesizing English Pitching Voiceover for all 6 Motion-Graphic Scenes...")
    async def synth_vo():
        for sc_id, vo_txt, _, _, _ in scenes:
            await edge_tts.Communicate(vo_txt, voice="en-US-ChristopherNeural", rate="+11%").save(str(OUTPUTS_DIR / f"mg_{sc_id}.mp3"))

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
            for p_t in np.arange(1.2, dur - 0.5, 2.2):
                add_tone(cur_t + float(p_t), 0.55, 320, 760, amp=0.09)
        elif sc_id == "scene2_smoke_cutter":
            for p_t in np.arange(0.8, dur - 0.5, 2.5):
                add_tone(cur_t + float(p_t), 0.45, 580, 920, amp=0.08)
        elif sc_id == "scene3_neural_pipeline":
            for p_t in np.arange(1.0, dur - 0.5, 1.8):
                add_tone(cur_t + float(p_t), 0.18, 880, 1320, amp=0.09)
        elif sc_id == "scene4_live_demo":
            scale = dur / 15.43
            add_tone(cur_t + 0.20 * scale, 0.12, 520, 1040, amp=0.22)
            for sw in [0.65, 1.95, 3.25, 4.55, 5.45]:
                add_tone(cur_t + sw * scale, 0.65, 280, 620, amp=0.08)
            add_tone(cur_t + 6.10 * scale, 0.14, 880, 1320, amp=0.18)
            add_tone(cur_t + 6.85 * scale, 0.14, 980, 1480, amp=0.18)
            for k in range(12):
                add_tone(cur_t + (7.60 + k * 0.06) * scale, 0.04, 1200 + k * 40, amp=0.11)
            add_tone(cur_t + 9.00 * scale, 0.28, 740, 1110, amp=0.15)
            add_tone(cur_t + 9.70 * scale, 0.28, 880, 1320, amp=0.15)
            add_tone(cur_t + 10.40 * scale, 0.28, 990, 1485, amp=0.15)
            for k in range(8):
                add_tone(cur_t + (11.10 + k * 0.09) * scale, 0.07, 580 + k * 95, amp=0.12)
            for k in range(16):
                add_tone(cur_t + (12.00 + k * 0.07) * scale, 0.04, 1380 + (k % 3) * 130, amp=0.10)
            add_tone(cur_t + 13.15 * scale, 0.85, 523.25, amp=0.12)
            add_tone(cur_t + 13.15 * scale, 0.85, 659.25, amp=0.12)
            add_tone(cur_t + 13.15 * scale, 0.85, 783.99, amp=0.12)
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
            scale = dur / 15.43
            subprocess.run(
                [
                    ff,
                    "-y",
                    "-itsscale",
                    f"{scale:.4f}",
                    "-i",
                    str(USER_RECORDING_PATH),
                    "-t",
                    f"{dur:.3f}",
                    "-vf",
                    "scale=1920:1014,pad=1920:1080:0:33:color=#040711,fps=30",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-threads",
                    "2",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(seg_mp4),
                ],
                capture_output=True,
                check=True,
            )
        print(f"  - Rendered motion-graphic scene {idx+1}/{len(scenes)}: {sc_id} ({dur:.2f}s)")

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

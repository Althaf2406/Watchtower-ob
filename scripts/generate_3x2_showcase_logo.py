"""Script to generate official 3:2 ratio (1800x1200 px) Project Logo & Showcase Card."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def create_gradient_canvas(width: int = 1800, height: int = 1200) -> Image.Image:
    """Create a high-tech dark navy gradient background with subtle radar lines."""
    # Create radial gradient from center
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height * 0.45
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    factor = np.clip(dist / max_dist, 0.0, 1.0)

    # Color 1 (center glow): #0D1B36, Color 2 (outer edge): #040813
    r = (13 * (1.0 - factor) + 4 * factor).astype(np.uint8)
    g = (27 * (1.0 - factor) + 8 * factor).astype(np.uint8)
    b = (54 * (1.0 - factor) + 19 * factor).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    base = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(base)

    # Subtle high-tech grid lines
    grid_color = (0, 229, 255, 20)  # semi-transparent cyan
    grid_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid_img)

    for gx in range(150, width, 150):
        gdraw.line([(gx, 0), (gx, height)], fill=(0, 229, 255, 12), width=1)
    for gy in range(120, height, 120):
        gdraw.line([(0, gy), (width, gy)], fill=(0, 229, 255, 12), width=1)

    # Concentric orbital radar rings centered at (cx, cy)
    for radius in (280, 420, 580):
        gdraw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            outline=(0, 229, 255, 18),
            width=1,
        )

    # Crosshair ticks
    tick_len = 30
    gdraw.line([(cx - radius - tick_len, cy), (cx - radius + tick_len, cy)], fill=(0, 229, 255, 40), width=2)
    gdraw.line([(cx + radius - tick_len, cy), (cx + radius + tick_len, cy)], fill=(0, 229, 255, 40), width=2)
    gdraw.line([(cx, cy - radius - tick_len), (cx, cy - radius + tick_len)], fill=(0, 229, 255, 40), width=2)
    gdraw.line([(cx, cy + radius - tick_len), (cx, cy + radius + tick_len)], fill=(0, 229, 255, 40), width=2)

    # Composite grid
    base.paste(Image.alpha_composite(Image.new("RGBA", (width, height), (0, 0, 0, 0)), grid_img).convert("RGB"), (0, 0), mask=grid_img.split()[3])
    return base


def main() -> None:
    width, height = 1800, 1200  # Exact 3:2 aspect ratio
    assets_dir = Path("assets")
    outputs_dir = Path("outputs")
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logo_path = assets_dir / "watchtower_logo.jpg"
    if not logo_path.exists():
        raise FileNotFoundError(f"Logo not found at {logo_path}")

    # Load system fonts
    font_bold_huge = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 64)
    font_bold_mid = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 26)
    font_bold_tag = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 16)
    font_regular_pill = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 20)

    # =========================================================================
    # VARIANT 1: OFFICIAL 3:2 LOGO SHOWCASE CARD (ICON + TITLE + BACKRONYM)
    # =========================================================================
    card = create_gradient_canvas(width, height)
    draw = ImageDraw.Draw(card)

    # Load and process the selected Geometric Prism "W" logo icon
    icon = Image.open(logo_path).convert("RGB")
    icon_size = 460  # Pristine resolution
    icon_resized = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)

    # Create smooth rounded mask for the icon
    mask = Image.new("L", (icon_size, icon_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), (icon_size, icon_size)], radius=96, fill=255)

    # Create neon glow halo behind the icon
    glow_size = icon_size + 60
    glow_img = Image.new("RGBA", (glow_size, glow_size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_img)
    glow_draw.rounded_rectangle([(30, 30), (glow_size - 30, glow_size - 30)], radius=110, outline=(0, 229, 255, 120), width=6)
    glow_blurred = glow_img.filter(ImageFilter.GaussianBlur(16))

    icon_x = (width - icon_size) // 2
    icon_y = 110

    # Paste halo & rounded icon
    card.paste(glow_blurred.convert("RGB"), (icon_x - 30, icon_y - 30), mask=glow_blurred.split()[3])
    card.paste(icon_resized, (icon_x, icon_y), mask=mask)

    # Subtle cyan border ring around icon
    draw.rounded_rectangle(
        [(icon_x - 2, icon_y - 2), (icon_x + icon_size + 2, icon_y + icon_size + 2)],
        radius=98,
        outline=(0, 229, 255, 180),
        width=2,
    )

    # Top Pill Tag
    pill_text = "GEOSPATIAL DEEP LEARNING  •  SATELLITE AI"
    pill_w = 440
    pill_h = 36
    pill_x = (width - pill_w) // 2
    pill_y = 620
    draw.rounded_rectangle([(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)], radius=18, fill="#0F2042", outline="#00E5FF", width=1)
    draw.ellipse([(pill_x + 18, pill_y + 13), (pill_x + 28, pill_y + 23)], fill="#00E676")
    draw.text((pill_x + 38, pill_y + 8), pill_text, font=font_bold_tag, fill="#00E5FF")

    # Main Title: "WATCHTOWER"
    title_text = "W A T C H T O W E R"
    title_bbox = draw.textbbox((0, 0), title_text, font=font_bold_huge)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_w) // 2, 680), title_text, font=font_bold_huge, fill="#FFFFFF")

    # Backronym Line 1: "Wildfire Attention & Transformer Calibration with"
    line1 = "Wildfire Attention & Transformer Calibration with"
    bbox1 = draw.textbbox((0, 0), line1, font=font_bold_mid)
    draw.text(((width - (bbox1[2] - bbox1[0])) // 2, 785), line1, font=font_bold_mid, fill="#38BDF8")

    # Backronym Line 2: "Hybrid Tversky Optimization for Woodland Emission Reporting"
    line2 = "Hybrid Tversky Optimization for Woodland Emission Reporting"
    bbox2 = draw.textbbox((0, 0), line2, font=font_bold_mid)
    draw.text(((width - (bbox2[2] - bbox2[0])) // 2, 825), line2, font=font_bold_mid, fill="#38BDF8")

    # Divider bar
    div_w = 600
    div_x = (width - div_w) // 2
    draw.line([(div_x, 885), (div_x + div_w, 885)], fill="#1E293B", width=2)
    draw.ellipse([(width // 2 - 4, 885 - 4), (width // 2 + 4, 885 + 4)], fill="#00E5FF")

    # Bottom Pill Badges Row
    badges = [
        ("🛰️ 17 Channels (S2+S5P)", "#E2E8F0", "#334155"),
        ("📊 Bayesian UQ (BALD)", "#38BDF8", "#0284C7"),
        ("🔥 ResNet-50 & ViT-S/16", "#F43F5E", "#E11D48"),
        ("🌲 GFW CO2e Accounting", "#4ADE80", "#16A34A"),
    ]
    total_badges_w = len(badges) * 270 + (len(badges) - 1) * 20
    start_bx = (width - total_badges_w) // 2
    by = 920

    for i, (btext, text_col, border_col) in enumerate(badges):
        bx = start_bx + i * 290
        draw.rounded_rectangle([(bx, by), (bx + 270, by + 46)], radius=12, fill="#0B132B", outline=border_col, width=1)
        tb = draw.textbbox((0, 0), btext, font=font_regular_pill)
        tw = tb[2] - tb[0]
        draw.text((bx + (270 - tw) // 2, by + 11), btext, font=font_regular_pill, fill=text_col)

    # Outer decorative frame
    margin = 32
    draw.rounded_rectangle([(margin, margin), (width - margin, height - margin)], radius=24, outline="#1E293B", width=2)
    # Corner tech ticks
    corner_len = 40
    for cx_coord, cy_coord, dx, dy in [
        (margin, margin, 1, 1),
        (width - margin, margin, -1, 1),
        (margin, height - margin, 1, -1),
        (width - margin, height - margin, -1, -1),
    ]:
        draw.line([(cx_coord, cy_coord), (cx_coord + dx * corner_len, cy_coord)], fill="#00E5FF", width=2)
        draw.line([(cx_coord, cy_coord), (cx_coord, cy_coord + dy * corner_len)], fill="#00E5FF", width=2)

    # Save Variant 1 in PNG and JPG formats (< 5 MB requirement)
    card_png_path = assets_dir / "watchtower_showcase_3x2.png"
    card_jpg_path = assets_dir / "watchtower_showcase_3x2.jpg"
    card.save(card_png_path, format="PNG", optimize=True)
    card.save(card_jpg_path, format="JPEG", quality=95, optimize=True)

    # Copy to outputs
    card.save(outputs_dir / "watchtower_showcase_3x2.png", format="PNG", optimize=True)
    card.save(outputs_dir / "watchtower_showcase_3x2.jpg", format="JPEG", quality=95, optimize=True)

    png_size_mb = os.path.getsize(card_png_path) / (1024 * 1024)
    jpg_size_mb = os.path.getsize(card_jpg_path) / (1024 * 1024)
    print(f"Variant 1 generated: {card_png_path} ({png_size_mb:.2f} MB), {card_jpg_path} ({jpg_size_mb:.2f} MB)")

    # =========================================================================
    # VARIANT 2: PURE ICON-FOCUSED 3:2 LOGO (OPTIMAL FOR PLATFORMS REQUIRING ICON LOGO)
    # =========================================================================
    icon_card = create_gradient_canvas(width, height)
    idraw = ImageDraw.Draw(icon_card)

    # Larger icon centered in 3:2 canvas
    large_size = 720
    large_resized = icon.resize((large_size, large_size), Image.Resampling.LANCZOS)
    lmask = Image.new("L", (large_size, large_size), 0)
    ImageDraw.Draw(lmask).rounded_rectangle([(0, 0), (large_size, large_size)], radius=150, fill=255)

    # Glow
    lglow_size = large_size + 90
    lglow = Image.new("RGBA", (lglow_size, lglow_size), (0, 0, 0, 0))
    ImageDraw.Draw(lglow).rounded_rectangle([(45, 45), (lglow_size - 45, lglow_size - 45)], radius=170, outline=(0, 229, 255, 140), width=8)
    lglow_blurred = lglow.filter(ImageFilter.GaussianBlur(24))

    lx = (width - large_size) // 2
    ly = (height - large_size) // 2 - 50

    icon_card.paste(lglow_blurred.convert("RGB"), (lx - 45, ly - 45), mask=lglow_blurred.split()[3])
    icon_card.paste(large_resized, (lx, ly), mask=lmask)

    idraw.rounded_rectangle(
        [(lx - 3, ly - 3), (lx + large_size + 3, ly + large_size + 3)],
        radius=153,
        outline=(0, 229, 255, 180),
        width=3,
    )

    # Centered Title below icon
    bt_text = "W A T C H T O W E R"
    bt_bbox = idraw.textbbox((0, 0), bt_text, font=font_bold_huge)
    bt_w = bt_bbox[2] - bt_bbox[0]
    idraw.text(((width - bt_w) // 2, ly + large_size + 40), bt_text, font=font_bold_huge, fill="#FFFFFF")

    # Subtitle
    sub_text = "Wildfire Attention & Transformer Calibration with Hybrid Tversky Optimization"
    sub_bbox = idraw.textbbox((0, 0), sub_text, font=font_bold_mid)
    sub_w = sub_bbox[2] - sub_bbox[0]
    idraw.text(((width - sub_w) // 2, ly + large_size + 130), sub_text, font=font_bold_mid, fill="#38BDF8")

    # Decorative frame
    idraw.rounded_rectangle([(margin, margin), (width - margin, height - margin)], radius=24, outline="#1E293B", width=2)
    for cx_coord, cy_coord, dx, dy in [
        (margin, margin, 1, 1),
        (width - margin, margin, -1, 1),
        (margin, height - margin, 1, -1),
        (width - margin, height - margin, -1, -1),
    ]:
        idraw.line([(cx_coord, cy_coord), (cx_coord + dx * corner_len, cy_coord)], fill="#00E5FF", width=2)
        idraw.line([(cx_coord, cy_coord), (cx_coord, cy_coord + dy * corner_len)], fill="#00E5FF", width=2)

    icon_png_path = assets_dir / "watchtower_icon_logo_3x2.png"
    icon_jpg_path = assets_dir / "watchtower_icon_logo_3x2.jpg"
    icon_card.save(icon_png_path, format="PNG", optimize=True)
    icon_card.save(icon_jpg_path, format="JPEG", quality=95, optimize=True)
    icon_card.save(outputs_dir / "watchtower_icon_logo_3x2.png", format="PNG", optimize=True)
    icon_card.save(outputs_dir / "watchtower_icon_logo_3x2.jpg", format="JPEG", quality=95, optimize=True)

    print("Successfully generated all 3:2 ratio logo and thumbnail cards!")


if __name__ == "__main__":
    main()

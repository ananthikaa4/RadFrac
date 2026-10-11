"""Generate the RADFRAC landing-screen backdrop.

Produces ``app/assets/home_bg.jpg`` -- an original, procedurally drawn medical
HUD: a glowing anterior thorax over a dark navy field, framed by instrument
panels, bar charts and a trace readout.

Why generate rather than ship a photo:
* No third-party licensing to track, and no stock watermark.
* The theme washes this to 93-99% white on the page and 58-97% in the hero band,
  so only large-scale composition survives. A procedural image can be tuned to
  put its structure exactly where the hero gradient is weakest (the right side),
  which a stock crop cannot.
* It is reproducible. Re-run this script after changing PALETTE and the asset
  regenerates deterministically (SEED is fixed).

Run::

    python -m scripts.make_home_backdrop
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OUT_PATH = Path(__file__).resolve().parent.parent / "app" / "assets" / "home_bg.jpg"

WIDTH, HEIGHT = 1600, 900
SUPERSAMPLE = 2                      # draw big, downscale -> smooth strokes
SEED = 20260927

PALETTE = {
    "deep":   (4, 10, 24),           # darkest corner of the field
    "mid":    (7, 22, 48),           # centre of the field
    "wash":   (18, 58, 112),         # radial glow behind the figure
    "anat":   (120, 226, 255),       # anatomy strokes
    "anat_2": (72, 190, 245),        # secondary anatomy
    "hud":    (60, 170, 230),        # panel chrome
    "hud_hi": (150, 240, 255),       # panel highlights / data
    "accent": (96, 165, 250),        # cool blue data
}


# --------------------------------------------------------------------------- #
# Field
# --------------------------------------------------------------------------- #
def build_field(w: int, h: int) -> np.ndarray:
    """Dark navy base: vertical gradient plus a radial wash behind the figure."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny, nx = yy / (h - 1), xx / (w - 1)

    deep = np.array(PALETTE["deep"], np.float32)
    mid = np.array(PALETTE["mid"], np.float32)
    wash = np.array(PALETTE["wash"], np.float32)

    # vertical gradient, darkest at the very top and bottom
    v = 1.0 - np.abs(ny - 0.42) * 1.75
    v = np.clip(v, 0.0, 1.0)[..., None]
    field = deep + (mid - deep) * v

    # radial wash centred on the thorax
    r = np.sqrt(((nx - 0.355) * 1.55) ** 2 + ((ny - 0.48) * 2.1) ** 2)
    glow = np.clip(1.0 - r, 0.0, 1.0) ** 2.2
    field += wash * glow[..., None] * 0.85

    # a second, cooler wash on the right where the hero gradient is thinnest
    r2 = np.sqrt(((nx - 0.84) * 1.9) ** 2 + ((ny - 0.5) * 2.6) ** 2)
    field += np.array(PALETTE["accent"], np.float32) * (
        np.clip(1.0 - r2, 0.0, 1.0) ** 2.6)[..., None] * 0.30

    return field


def add_grid(field: np.ndarray, step: int, alpha: float) -> None:
    """Faint instrument grid, in place."""
    line = np.array(PALETTE["hud"], np.float32) * alpha
    field[::step, :, :] += line
    field[:, ::step, :] += line


# --------------------------------------------------------------------------- #
# Drawing helpers (all operate on the additive glow layer)
# --------------------------------------------------------------------------- #
def bracket(d: ImageDraw.ImageDraw, box, size, colour, width) -> None:
    """Four corner brackets -- the HUD frame motif."""
    x0, y0, x1, y1 = box
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(cx, cy), (cx + sx * size, cy)], fill=colour, width=width)
        d.line([(cx, cy), (cx, cy + sy * size)], fill=colour, width=width)


def panel(d: ImageDraw.ImageDraw, box, colour, width, radius) -> None:
    d.rounded_rectangle(box, radius=radius, outline=colour, width=width)


def fake_text(d: ImageDraw.ImageDraw, x, y, widths, colour, lh, th) -> None:
    """Rows of short bars that read as label text once blurred and washed."""
    for i, bw in enumerate(widths):
        d.rectangle([x, y + i * lh, x + bw, y + i * lh + th], fill=colour)


def bars(d: ImageDraw.ImageDraw, box, values, colour, gap) -> None:
    x0, y0, x1, y1 = box
    n = len(values)
    bw = (x1 - x0 - gap * (n - 1)) / n
    for i, v in enumerate(values):
        bx = x0 + i * (bw + gap)
        by = y1 - (y1 - y0) * v
        d.rectangle([bx, by, bx + bw, y1], fill=colour)


def trace(d: ImageDraw.ImageDraw, box, pts, colour, width) -> None:
    """Polyline readout scaled into a box; ``pts`` are 0..1 values."""
    x0, y0, x1, y1 = box
    n = len(pts)
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    ys = [y1 - (y1 - y0) * v for v in pts]
    d.line(list(zip(xs, ys)), fill=colour, width=width, joint="curve")


def curve(d: ImageDraw.ImageDraw, pts, colour, width) -> None:
    if len(pts) > 1:
        d.line(pts, fill=colour, width=max(1, int(width)), joint="curve")


def sweep(cx, cy, rx, ry, a0, a1, n=56):
    """Points along an ellipse arc, degrees, 0 deg = +x, CW in image space."""
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


def dial(d: ImageDraw.ImageDraw, cx, cy, r, colour, width, ticks) -> None:
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=width)
    d.ellipse([cx - r * .62, cy - r * .62, cx + r * .62, cy + r * .62],
              outline=colour, width=max(1, width - 1))
    for i in range(ticks):
        a = 2 * math.pi * i / ticks
        d.line([(cx + math.cos(a) * r * .70, cy + math.sin(a) * r * .70),
                (cx + math.cos(a) * r * .94, cy + math.sin(a) * r * .94)],
               fill=colour, width=max(1, width - 1))


# --------------------------------------------------------------------------- #
# Anatomy
# --------------------------------------------------------------------------- #
def draw_thorax(d: ImageDraw.ImageDraw, cx: float, top: float, u: float) -> None:
    """Stylised anterior thorax, drawn from explicit curves.

    Everything is expressed in ``u`` (one vertebral-block unit) measured down from
    ``top``, the base of the neck. No arms: a chest plate reads more clearly as
    anatomy than a full skeleton at this scale, and the arm strokes previously
    closed into tube shapes.
    """
    anat, anat2 = PALETTE["anat"], PALETTE["anat_2"]
    w3 = max(1, int(u * 0.032))
    w2 = max(1, int(u * 0.023))

    # --- cervical spine, above the shoulder line ------------------------------
    for i in range(3):
        y = top - u * (0.72 - 0.24 * i)
        vw = u * (0.105 + 0.014 * i)
        d.rounded_rectangle([cx - vw, y - u * .062, cx + vw, y + u * .062],
                            radius=max(1, int(u * .03)), outline=anat2, width=1)

    # --- thoracic + lumbar spine ---------------------------------------------
    spine_top, spine_bot = top + u * 0.30, top + u * 4.70
    n_vert = 14
    for i in range(n_vert):
        y = spine_top + (spine_bot - spine_top) * i / (n_vert - 1)
        vw = u * (0.13 + 0.075 * i / n_vert)
        d.rounded_rectangle([cx - vw, y - u * .072, cx + vw, y + u * .072],
                            radius=max(1, int(u * .035)), outline=anat, width=1)
        d.line([(cx, y - u * .072), (cx, y + u * .072)], fill=anat2, width=1)

    # --- sternum: manubrium + body -------------------------------------------
    d.rounded_rectangle([cx - u * .22, top + u * .62, cx + u * .22, top + u * 1.05],
                        radius=max(1, int(u * .07)), outline=anat, width=w2)
    d.rounded_rectangle([cx - u * .13, top + u * 1.08, cx + u * .13, top + u * 3.05],
                        radius=max(1, int(u * .07)), outline=anat, width=w2)

    # --- ribs ----------------------------------------------------------------- #
    # Half-widths are tabulated rather than computed from a sine: the cage has to
    # taper hard at both ends (rib 1 is short, ribs 9-10 pull back in) or it reads
    # as a barrel instead of a thorax.
    # The top pair has to be nearly as wide as the widest pair, otherwise the
    # clavicles overhang a narrow cage and read as antennae.
    half_f = (0.98, 1.16, 1.30, 1.42, 1.50, 1.55, 1.52, 1.38)
    drop_f = (0.42, 0.56, 0.70, 0.84, 0.96, 1.06, 1.12, 1.16)
    tips: dict[tuple[int, int], tuple[float, float]] = {}
    for i, (hf, df) in enumerate(zip(half_f, drop_f)):
        half, drop = u * hf, u * df
        y0 = top + u * (0.62 + 0.30 * i)
        col = anat if i % 2 == 0 else anat2
        for s in (-1, 1):
            pts = []
            for a in np.linspace(0.0, 100.0, 40):
                rad = math.radians(a)
                pts.append((cx + s * half * math.sin(rad),
                            y0 + drop * (1.0 - math.cos(rad))))
            curve(d, pts, col, w3)
            tips[(i, s)] = pts[-1]

    # Costal cartilage. One stub per rib, angled back toward the sternum: on the
    # lower ribs these naturally form the costal margin, so no separate curve is
    # drawn for it (the earlier one doubled up with the rib tips and read as a
    # bucket under the cage).
    sternum_bot = top + u * 3.05
    for s in (-1, 1):
        for i in range(len(half_f)):
            tx, ty = tips[(i, s)]
            ay = min(ty - u * .14, sternum_bot)
            d.line([(tx, ty), (cx + s * u * .15, ay)], fill=anat2, width=w2)

    # --- lungs: closed outlines tucked inside the cage ------------------------ #
    for s in (-1, 1):
        body = (sweep(cx + s * u * .76, top + u * 1.95, u * .54, u * 1.10, 90, 270)
                if s < 0 else
                sweep(cx + s * u * .76, top + u * 1.95, u * .54, u * 1.10, -90, 90))
        curve(d, body + [body[0]], anat2, w2)

    # --- pelvis: iliac wings + sacrum ----------------------------------------- #
    for s in (-1, 1):
        curve(d, sweep(cx + s * u * .50, top + u * 4.95, u * .78, u * .70,
                       184 if s > 0 else 356, 305 if s > 0 else 235), anat, w3)
    curve(d, sweep(cx, top + u * 4.80, u * .22, u * .54, 250, 290), anat2, w2)

    # --- scan platform, clear of the pelvis ----------------------------------- #
    for rr, aw in ((1.85, w3), (1.38, w2), (0.92, w2)):
        d.ellipse([cx - u * rr, top + u * (5.90 - rr * .10),
                   cx + u * rr, top + u * (5.90 + rr * .28)],
                  outline=PALETTE["hud"], width=aw)

    # --- shoulder girdle ------------------------------------------------------ #
    # Clavicle plus humeral head only. The scapula sweep that used to be here
    # read as a pair of antennae above the shoulders, so it is gone.
    for s in (-1, 1):
        clav = []
        for t in np.linspace(0.0, 1.0, 40):
            clav.append((cx + s * u * (0.18 + 1.18 * t),
                         top + u * (0.34 - 0.20 * math.sin(math.pi * t) + 0.14 * t)))
        curve(d, clav, anat, w3)
        hx, hy = cx + s * u * 1.46, top + u * .56
        d.ellipse([hx - u * .18, hy - u * .18, hx + u * .18, hy + u * .18],
                  outline=anat, width=w2)
        # short humeral shaft so the joint is not left floating
        d.line([(hx + s * u * .06, hy + u * .18),
                (hx + s * u * .14, hy + u * .78)], fill=anat2, width=w2)


# --------------------------------------------------------------------------- #
# HUD furniture
# --------------------------------------------------------------------------- #
def draw_hud(d: ImageDraw.ImageDraw, w: int, h: int, rng: np.random.Generator) -> None:
    """Instrument furniture, arranged around a clear central channel.

    Layout rule: the figure occupies roughly x 22-50%, so nothing but thin chrome
    crosses that band. All data panels live in the right column (x 54-96%), which
    is exactly where the hero gradient thins out and lets the image through.
    """
    hud, hi, acc = PALETTE["hud"], PALETTE["hud_hi"], PALETTE["accent"]
    u = w / 100.0                                      # 1 u == 1% of the WIDTH
    v = h / 100.0                                      # 1 v == 1% of the HEIGHT
    lw = max(1, int(u * 0.16))
    rcol_l, rcol_r = u * 54.0, w - u * 4.2             # right data column

    # ---- frame -------------------------------------------------------------- #
    d.rounded_rectangle([u * 1.6, v * 2.2, w - u * 1.6, h - v * 2.2],
                        radius=int(u * 1.2), outline=hud, width=lw)
    bracket(d, (u * 3.0, v * 4.6, w - u * 3.0, h - v * 4.6), u * 3.2, hi,
            max(1, lw + 1))

    # top rail, broken either side of the figure's head
    for x0, x1 in ((u * 5.0, u * 27.0), (u * 45.0, w - u * 5.0)):
        d.line([(x0, v * 6.0), (x1, v * 6.0)], fill=hud, width=lw)
        n = max(1, int((x1 - x0) / (u * 1.8)))
        for i in range(n + 1):
            x = x0 + i * (x1 - x0) / n
            d.line([(x, v * 4.8), (x, v * 7.3)], fill=hud, width=1)

    # ---- left column -------------------------------------------------------- #
    d.line([(u * 4.6, v * 13.0), (u * 17.5, v * 13.0)], fill=hi, width=lw)
    fake_text(d, u * 4.6, v * 15.5, [u * 9, u * 6.5, u * 11.5, u * 5], hi,
              v * 5.0, max(1, int(v * .62)))

    dial(d, u * 7.6, v * 42.0, u * 3.2, hud, lw, 12)
    fake_text(d, u * 12.4, v * 39.8, [u * 6.5, u * 4.0], hud, v * 4.2,
              max(1, int(v * .56)))

    # gauge stack, low left, clear of the pelvis
    fake_text(d, u * 4.6, v * 59.0, [u * 8.5], hi, v * 4.0, max(1, int(v * .6)))
    for i, frac in enumerate((0.62, 0.84, 0.41, 0.73)):
        y = v * 66.0 + i * v * 5.6
        d.rounded_rectangle([u * 4.6, y, u * 4.6 + u * 13.0, y + v * 2.6],
                            radius=int(v * 1.3), outline=hud, width=1)
        d.rounded_rectangle([u * 4.6, y, u * 4.6 + u * 13.0 * frac, y + v * 2.6],
                            radius=int(v * 1.3), fill=acc)

    # ---- right column: three stacked panels, no overlap --------------------- #
    # 1. vitals
    box = (rcol_l, v * 11.0, rcol_r, v * 33.0)
    panel(d, box, hud, lw, int(u * .9))
    fake_text(d, box[0] + u * 1.6, box[1] + v * 3.0, [u * 5.5, u * 8.0, u * 4.5],
              hi, v * 4.6, max(1, int(v * .64)))
    for i, frac in enumerate((0.78, 0.55, 0.90)):
        yy = box[1] + v * 4.0 + i * v * 4.6
        d.rounded_rectangle([box[0] + u * 16.0, yy,
                             box[0] + u * 16.0 + u * 20.0 * frac, yy + v * 1.8],
                            radius=int(v * .9), fill=acc)
    d.line([(box[0] + u * 1.4, box[3] - v * 6.6), (box[2] - u * 1.4, box[3] - v * 6.6)],
           fill=hud, width=1)
    fake_text(d, box[0] + u * 1.6, box[3] - v * 5.2, [u * 22, u * 14], hud,
              v * 3.0, max(1, int(v * .5)))

    # 2. bars + sparkline
    box = (rcol_l, v * 37.0, rcol_r, v * 62.0)
    panel(d, box, hud, lw, int(u * .9))
    fake_text(d, box[0] + u * 1.6, box[1] + v * 2.6, [u * 8.5, u * 5.0], hi, v * 3.8,
              max(1, int(v * .6)))
    inner = (box[0] + u * 1.8, box[1] + v * 11.0, box[2] - u * 1.8, box[3] - v * 3.4)
    bars(d, inner, list(rng.uniform(0.22, 1.0, 24)), acc, u * .26)
    trace(d, inner, list(rng.uniform(0.38, 0.95, 24)), hi, lw)

    # 3. trace readout (QRS-like)
    box = (rcol_l, v * 66.0, rcol_r, v * 91.0)
    panel(d, box, hud, lw, int(u * .9))
    fake_text(d, box[0] + u * 1.6, box[1] + v * 2.6, [u * 11, u * 6], hi, v * 3.8,
              max(1, int(v * .6)))
    inner = (box[0] + u * 1.8, box[1] + v * 11.0, box[2] - u * 1.8, box[3] - v * 3.4)
    d.line([(inner[0], (inner[1] + inner[3]) / 2),
            (inner[2], (inner[1] + inner[3]) / 2)], fill=hud, width=1)
    beats = []
    for _ in range(7):
        beats += [.44, .42, .46, .40, .54, .30, .96, .16, .60, .50, .45, .46]
    trace(d, inner, beats, hi, lw)

    # ---- side rails --------------------------------------------------------- #
    for x in (u * 2.6, w - u * 2.6):
        d.line([(x, v * 34.0), (x, v * 66.0)], fill=hi, width=max(1, lw + 1))


# --------------------------------------------------------------------------- #
# Compose
# --------------------------------------------------------------------------- #
def render(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(SEED)
    s = SUPERSAMPLE
    w, h = width * s, height * s

    field = build_field(w, h)
    add_grid(field, step=int(26 * s), alpha=0.055)

    glow = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(glow)

    draw_thorax(d, cx=w * 0.355, top=h * 0.185, u=h * 0.115)
    draw_hud(d, w, h, rng)

    # bloom: wide + tight blur added on top of the crisp strokes
    wide = glow.filter(ImageFilter.GaussianBlur(radius=int(9 * s)))
    tight = glow.filter(ImageFilter.GaussianBlur(radius=int(2.2 * s)))

    out = (field
           + np.asarray(wide, np.float32) * 0.85
           + np.asarray(tight, np.float32) * 0.65
           + np.asarray(glow, np.float32) * 0.80)

    # scanlines
    out[::int(4 * s), :, :] *= 0.93

    # vignette so the washed-out corners stay quiet under the hero gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / (w - 1)) - .5) ** 2 + ((yy / (h - 1)) - .5) ** 2) / 0.707
    out *= (1.0 - 0.46 * np.clip(r, 0, 1) ** 2.1)[..., None]

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    return img.resize((width, height), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img = render(args.width, args.height)
    img.save(args.out, format="JPEG", quality=args.quality, optimize=True,
             progressive=True)
    kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out}  {img.size[0]}x{img.size[1]}  {kb:.0f} KB")


if __name__ == "__main__":
    main()

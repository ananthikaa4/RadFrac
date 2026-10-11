"""RADFRAC visual identity — light clinical theme.

Original design system. The direction was informed by the reference aesthetics
named in the brief - Linear's restraint and typographic hierarchy, Vercel/WorkOS
neutrals with a single accent, Hex's data-density discipline, and clinical
dashboards' preference for calm surfaces over saturated colour - but no reference
is reproduced. Nothing here is copied from a specific site.

Principles applied:
* Bright, white-paper surfaces. A radiology report is read on white; the UI now
  matches the artefact it produces.
* One accent hue (clinical cyan-blue) carries interaction; semantic red/green/amber
  are reserved strictly for findings and warnings, never decoration.
* Larger type throughout. The smallest text on screen is 13px, body copy is 17px,
  and every interactive control clears a 48px hit target.
* A photographic backdrop on the landing screen, washed to near-white so contrast
  ratios stay well inside AA. The image never sits directly under body text.
* Type scale and 4px spacing rhythm rather than ad-hoc sizes.

Background image
----------------
Drop a JPG/PNG into ``app/assets/`` (``home_bg.jpg`` is the canonical name) and it
is picked up automatically, inlined as a data URI and used behind the landing
screen. With no file present the theme falls back to a CSS-only light gradient,
so the app always renders correctly.
"""
from __future__ import annotations

import base64
import functools
import io
from pathlib import Path

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
COLORS = {
    "bg":          "#f4f7fb",   # app background (bright, faint blue cast)
    "bg_soft":     "#e9f0f8",
    "surface":     "#ffffff",   # cards
    "surface_2":   "#f1f6fb",   # raised / hover / inset
    "border":      "#cfdcea",
    "border_soft": "#e3ebf4",
    "text":        "#0c1b2a",   # near-black navy
    "text_muted":  "#42566d",
    "text_faint":  "#64788f",
    "accent":      "#0e7490",   # clinical cyan-blue
    "accent_dim":  "#0b5c73",
    "accent_soft": "#e0f2f7",
    "positive":    "#c81e1e",   # a FINDING (fracture) - attention, not "good"
    "negative":    "#15803d",   # no finding
    "warn":        "#b45309",
}

FONT_STACK = ('"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, '
              '"Helvetica Neue", Arial, sans-serif')
MONO_STACK = ('"JetBrains Mono", "SF Mono", ui-monospace, "Cascadia Mono", '
              'Consolas, monospace')

# --------------------------------------------------------------------------- #
# Background asset discovery
# --------------------------------------------------------------------------- #
ASSET_DIR = Path(__file__).resolve().parent / "assets"
BACKGROUND_FILENAME = "home_bg.jpg"

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_PREFERRED_STEMS = ("home_bg", "hero_bg", "background", "bg", "radfrac_bg",
                    "radfrac")
_MIME = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
         ".webp": "webp", ".bmp": "bmp"}


def find_background_file() -> Path | None:
    """First usable image in ``app/assets``, preferring conventional names."""
    if not ASSET_DIR.is_dir():
        return None
    files = [p for p in sorted(ASSET_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in _IMAGE_EXT]
    if not files:
        return None
    lowered = {p: p.stem.lower() for p in files}
    for stem in _PREFERRED_STEMS:                     # exact name wins
        for p in files:
            if lowered[p] == stem:
                return p
    for stem in _PREFERRED_STEMS:                     # then partial match
        for p in files:
            if stem in lowered[p]:
                return p
    return files[0]


@functools.lru_cache(maxsize=4)
def _encode_background(path: str, mtime: float, size: int) -> str:
    """Inline an image as a data URI, downscaling anything oversized.

    ``mtime``/``size`` are part of the cache key so replacing the file busts it.
    """
    src = Path(path)
    raw = src.read_bytes()
    mime = _MIME.get(src.suffix.lower(), "jpeg")

    if len(raw) > 250_000:                            # keep the stylesheet sane
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.thumbnail((1600, 1600), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=78, optimize=True)
            if buf.tell() < len(raw):
                raw, mime = buf.getvalue(), "jpeg"
        except Exception:                             # pragma: no cover
            pass                                      # oversized but still valid

    return f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")


def background_data_uri() -> str | None:
    """CSS-ready data URI for the landing backdrop, or ``None`` if absent."""
    found = find_background_file()
    if found is None:
        return None
    try:
        stat = found.stat()
        return _encode_background(str(found), stat.st_mtime, stat.st_size)
    except Exception:                                 # pragma: no cover
        return None


def has_background_image() -> bool:
    return find_background_file() is not None


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #
def css() -> str:
    c = COLORS
    uri = background_data_uri()

    if uri:
        # The data URI is declared once as a custom property and referenced twice,
        # so a large photo is not inlined into the stylesheet two times over.
        bg_var = f':root {{ --rf-bg-image: url("{uri}"); }}'
        # Whole page: the photograph, washed to near-white so it reads as texture.
        page_bg = f"""
{bg_var}
[data-testid="stAppViewContainer"] {{
    background-image:
        linear-gradient(180deg, rgba(255,255,255,.93) 0%,
                                rgba(255,255,255,.96) 40%,
                                rgba(255,255,255,.99) 100%),
        var(--rf-bg-image);
    background-size: cover, cover;
    background-position: center top, center top;
    background-repeat: no-repeat, no-repeat;
    background-attachment: fixed, fixed;
}}"""
        # Hero band: the same photograph allowed through on the right, where no
        # text sits, so the visual is legible without touching copy contrast.
        hero_bg = """
    background-image:
        linear-gradient(100deg, rgba(255,255,255,.975) 0%,
                                rgba(255,255,255,.945) 34%,
                                rgba(255,255,255,.80) 60%,
                                rgba(226,242,250,.58) 100%),
        var(--rf-bg-image);
    background-size: cover, cover;
    background-position: center, center right;
    background-repeat: no-repeat, no-repeat;"""
    else:
        page_bg = f"""
[data-testid="stAppViewContainer"] {{
    background-image:
        radial-gradient(1100px 520px at 78% -8%, {c['accent_soft']} 0%,
                        rgba(224,242,247,0) 70%),
        linear-gradient(180deg, #ffffff 0%, {c['bg']} 100%);
    background-attachment: fixed;
}}"""
        hero_bg = f"""
    background-image:
        radial-gradient(760px 340px at 88% 26%, {c['accent_soft']} 0%,
                        rgba(224,242,247,0) 72%),
        linear-gradient(100deg, #ffffff 0%, {c['surface_2']} 100%);"""

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ---------- reset Streamlit chrome so this stops looking like Streamlit ---- */
#MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

.block-container {{
    padding: 0 2rem 6rem 2rem !important;
    max-width: 1220px !important;
}}
html, body, [class*="css"] {{
    font-family: {FONT_STACK};
    color: {c['text']};
    -webkit-font-smoothing: antialiased;
}}
html, body {{ font-size: 17px; background: {c['bg']}; }}
[data-testid="stAppViewContainer"] {{ background-color: {c['bg']}; }}
{page_bg}
[data-testid="stAppViewContainer"] > .main {{ background: transparent; }}
p, li, span, div {{ font-size: inherit; }}

/* consistent vertical rhythm; columns stay top-aligned so side-by-side
   panels line up on their headings rather than on their last element */
[data-testid="stVerticalBlock"] {{ gap: .6rem; }}
[data-testid="stHorizontalBlock"] {{ align-items: flex-start; }}
[data-testid="column"] > [data-testid="stVerticalBlock"] {{ gap: .6rem; }}

/* ---------- app bar ------------------------------------------------------- */
.rf-appbar {{
    display: flex; align-items: center; justify-content: flex-start;
    gap: 14px;
    padding: 22px 4px 18px 4px;
    border-bottom: 1px solid {c['border']};
    margin-bottom: 34px;
}}
.rf-brand {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
.rf-logo {{
    font-size: 26px; font-weight: 800; letter-spacing: .14em;
    color: {c['text']}; line-height: 1;
}}
.rf-logo span {{ color: {c['accent']}; }}
.rf-brand-sub {{
    font-size: 14px; color: {c['text_muted']}; letter-spacing: .08em;
    text-transform: uppercase; font-weight: 600;
}}

/* ---------- hero ---------------------------------------------------------- */
.rf-hero {{
    border: 1px solid {c['border']};
    border-radius: 20px;
    padding: 48px 46px 50px 46px;
    margin-bottom: 8px;
    box-shadow: 0 18px 42px -26px rgba(12,27,42,.34),
                0 2px 6px -2px rgba(12,27,42,.08);
    {hero_bg}
}}
.rf-hero h1 {{
    font-size: 50px; line-height: 1.08; font-weight: 800;
    letter-spacing: -.028em; margin: 0 0 18px 0; color: {c['text']};
    max-width: 30ch;
}}
.rf-hero p {{
    font-size: 19px; color: {c['text_muted']}; margin: 0; max-width: 60ch;
    line-height: 1.62; font-weight: 400;
}}
.rf-eyebrow {{
    display: inline-block;
    font-size: 13px; letter-spacing: .16em; text-transform: uppercase;
    color: {c['accent_dim']}; font-weight: 700; margin-bottom: 20px;
    background: rgba(255,255,255,.9);
    border: 1px solid {c['border']};
    border-radius: 999px; padding: 7px 15px;
}}

/* ---------- step rail ----------------------------------------------------- */
.rf-steps {{
    display: flex; gap: 10px; margin: 34px 0 30px 0; flex-wrap: wrap;
    align-items: stretch;
}}
.rf-step {{
    display: flex; align-items: center; gap: 11px;
    font-size: 15px; font-weight: 600; letter-spacing: .01em;
    padding: 12px 20px; border-radius: 11px;
    border: 1px solid {c['border']}; color: {c['text_faint']};
    background: rgba(255,255,255,.82);
}}
.rf-step.active {{
    border-color: {c['accent']}; color: {c['accent_dim']};
    background: {c['accent_soft']};
    box-shadow: 0 0 0 3px rgba(14,116,144,.10);
}}
.rf-step.done {{ color: {c['negative']}; border-color: #bfe3cc; background: #f1faf4; }}
.rf-step-n {{
    width: 26px; height: 26px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 800; flex: 0 0 auto;
    border: 2px solid currentColor;
}}

/* ---------- surfaces ------------------------------------------------------ */
.rf-card {{
    background: {c['surface']}; border: 1px solid {c['border']};
    border-radius: 14px; padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(12,27,42,.05);
}}
.rf-section-title {{
    font-size: 14px; letter-spacing: .14em; text-transform: uppercase;
    color: {c['text_muted']}; font-weight: 700; margin: 0 0 14px 0;
}}
.rf-grid3 {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px; margin: 6px 0 4px 0; align-items: stretch;
}}
.rf-tile {{
    background: {c['surface']}; border: 1px solid {c['border']};
    border-radius: 14px; padding: 22px 22px 24px 22px;
    box-shadow: 0 1px 3px rgba(12,27,42,.05);
    display: flex; flex-direction: column; gap: 8px;
}}
.rf-tile-n {{
    font-family: {MONO_STACK}; font-size: 14px; font-weight: 600;
    color: {c['accent']}; letter-spacing: .1em;
}}
.rf-tile-t {{ font-size: 18px; font-weight: 700; color: {c['text']}; }}
.rf-tile-d {{ font-size: 15.5px; color: {c['text_muted']}; line-height: 1.58; }}

/* ---------- verdict ------------------------------------------------------- */
.rf-verdict {{
    border-radius: 18px; padding: 54px 34px 58px 34px; text-align: center;
    border: 1px solid {c['border']}; background: {c['surface']};
    position: relative; overflow: hidden;
    box-shadow: 0 14px 34px -24px rgba(12,27,42,.32),
                0 1px 3px rgba(12,27,42,.05);
}}
.rf-verdict::before {{
    content: ""; position: absolute; inset: 0 0 auto 0; height: 5px;
}}
.rf-verdict.pos::before {{ background: {c['positive']}; }}
.rf-verdict.neg::before {{ background: {c['negative']}; }}
.rf-verdict-label {{
    font-size: 13px; letter-spacing: .18em; text-transform: uppercase;
    color: {c['text_muted']}; margin-bottom: 16px; font-weight: 700;
}}
/* The verdict now carries the whole card on its own -- no percentage, no
   threshold caption, no probability meter -- so it is sized to fill the space
   those three elements used to occupy. */
.rf-verdict-head {{
    font-size: 46px; font-weight: 800; letter-spacing: -.022em;
    line-height: 1.12; margin: 0;
}}
.rf-verdict.pos .rf-verdict-head {{ color: {c['positive']}; }}
.rf-verdict.neg .rf-verdict-head {{ color: {c['negative']}; }}

/* ---------- model cards --------------------------------------------------- */
.rf-model {{
    background: {c['surface']}; border: 1px solid {c['border']};
    border-radius: 14px; padding: 24px 24px; height: 100%;
    box-shadow: 0 1px 3px rgba(12,27,42,.05);
}}
.rf-model-name {{ font-size: 19px; font-weight: 700; margin-bottom: 4px; }}
.rf-model-kind {{
    font-size: 14px; color: {c['text_muted']}; letter-spacing: .02em;
    margin-bottom: 18px;
}}
.rf-model-verdict {{
    font-size: 14px; font-weight: 800; letter-spacing: .12em;
    text-transform: uppercase;
}}
.rf-model-p {{
    font-family: {MONO_STACK}; font-size: 40px; font-weight: 600;
    letter-spacing: -.02em; margin: 8px 0 16px 0;
}}
.rf-kv {{
    display: flex; justify-content: space-between; gap: 12px;
    font-size: 14.5px; color: {c['text_muted']};
    padding: 9px 0; border-top: 1px solid {c['border_soft']};
}}
.rf-kv b {{ color: {c['text']}; font-weight: 600; font-family: {MONO_STACK}; }}

/* ---------- agreement ----------------------------------------------------- */
.rf-agree {{
    display: flex; align-items: flex-start; gap: 14px;
    border-radius: 13px; padding: 19px 21px; font-size: 16px;
    border: 1px solid {c['border']}; background: {c['surface']};
    box-shadow: 0 1px 3px rgba(12,27,42,.05);
}}
.rf-agree > span {{ font-size: 20px; line-height: 1.2; }}
.rf-agree.ok {{ border-left: 5px solid {c['negative']}; }}
.rf-agree.warn {{ border-left: 5px solid {c['warn']}; }}
.rf-agree-t {{ font-weight: 800; letter-spacing: .06em; margin-bottom: 5px;
               font-size: 16px; }}
.rf-agree-d {{ color: {c['text_muted']}; font-size: 15.5px; line-height: 1.6; }}

/* ---------- disclaimer ---------------------------------------------------- */
.rf-disclaim {{
    font-size: 15.5px; color: {c['text_muted']}; line-height: 1.62;
    border: 1px solid #f0dcb4; border-radius: 13px;
    padding: 18px 20px; background: #fffaf0;
}}
.rf-disclaim b {{ color: {c['warn']}; font-weight: 700; }}

/* ---------- viewer -------------------------------------------------------- */
.rf-viewer {{
    background: {c['surface_2']}; border: 1px solid {c['border']};
    border-radius: 14px; padding: 16px; text-align: center;
    box-shadow: inset 0 1px 3px rgba(12,27,42,.05);
}}
.rf-viewer-bar {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 12px;
    font-size: 13px; color: {c['text_muted']}; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: 12px; padding: 0 4px;
    font-weight: 600;
}}

/* ---------- buttons ------------------------------------------------------- */
.stButton > button {{
    background: {c['accent']} !important; color: #ffffff !important;
    border: 1px solid {c['accent']} !important;
    border-radius: 11px; font-weight: 700; font-size: 16.5px;
    padding: 13px 24px; width: 100%; min-height: 52px;
    letter-spacing: .01em; line-height: 1.25;
    box-shadow: 0 2px 6px -1px rgba(14,116,144,.38);
    transition: background .15s ease, transform .06s ease, box-shadow .15s ease;
}}
.stButton > button p {{ font-size: 16.5px !important; font-weight: 700 !important;
                        color: #ffffff !important; margin: 0 !important; }}
.stButton > button:hover {{
    background: {c['accent_dim']} !important; border-color: {c['accent_dim']} !important;
    box-shadow: 0 4px 12px -2px rgba(14,116,144,.45);
}}
.stButton > button:focus-visible {{
    outline: 3px solid rgba(14,116,144,.35) !important; outline-offset: 2px;
}}
.stButton > button:active {{ transform: translateY(1px); }}
.stButton > button:disabled,
.stButton > button:disabled:hover {{
    background: {c['surface_2']} !important; color: {c['text_faint']} !important;
    border-color: {c['border']} !important; box-shadow: none;
}}
.stButton > button:disabled p {{ color: {c['text_faint']} !important; }}

.stButton > button[kind="secondary"] {{
    background: {c['surface']} !important; color: {c['accent_dim']} !important;
    border: 2px solid {c['accent']} !important;
    box-shadow: 0 1px 3px rgba(12,27,42,.06);
}}
.stButton > button[kind="secondary"] p {{ color: {c['accent_dim']} !important; }}
.stButton > button[kind="secondary"]:hover {{
    background: {c['accent_soft']} !important; color: {c['accent_dim']} !important;
}}

.stDownloadButton > button {{
    background: {c['surface']} !important; color: {c['accent_dim']} !important;
    border: 2px solid {c['accent']} !important; border-radius: 11px;
    font-weight: 700; font-size: 16.5px; padding: 13px 24px; width: 100%;
    min-height: 52px;
}}
.stDownloadButton > button p {{ font-size: 16.5px !important; font-weight: 700 !important;
                               color: {c['accent_dim']} !important; margin: 0 !important; }}
.stDownloadButton > button:hover {{ background: {c['accent_soft']} !important; }}

/* ---------- file uploader ------------------------------------------------- */
[data-testid="stFileUploader"] {{
    background: {c['surface']};
    border: 2.5px dashed {c['accent']};
    border-radius: 16px; padding: 30px 26px;
    box-shadow: 0 1px 3px rgba(12,27,42,.05);
    transition: border-color .18s ease, background .18s ease;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {c['accent_dim']}; background: {c['accent_soft']};
}}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {{
    background: transparent !important; border: none !important;
    padding: 0 !important;
}}
[data-testid="stFileUploader"] label {{
    font-size: 16px !important; color: {c['text']} !important;
    font-weight: 700 !important; text-transform: none !important;
    letter-spacing: .01em !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {{
    font-size: 16.5px !important; color: {c['text']} !important; font-weight: 600;
}}
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploader"] small {{
    font-size: 13.5px !important; color: {c['text_muted']} !important;
}}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {{
    background: {c['accent']} !important; color: #ffffff !important;
    border: 1px solid {c['accent']} !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 16px !important;
    min-height: 48px !important; padding: 11px 22px !important;
    box-shadow: 0 2px 6px -1px rgba(14,116,144,.38);
}}
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploader"] button p {{
    color: #ffffff !important; font-size: 16px !important; font-weight: 700 !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {{
    background: {c['accent_dim']} !important; border-color: {c['accent_dim']} !important;
}}
[data-testid="stFileUploader"] svg {{ fill: {c['accent']} !important;
                                      color: {c['accent']} !important; }}

/* ---------- inputs -------------------------------------------------------- */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {{
    background: {c['surface']} !important;
    border: 1.5px solid {c['border']} !important;
    border-radius: 10px !important; color: {c['text']} !important;
    font-size: 16.5px !important; min-height: 50px;
}}
.stTextInput > div > div > input::placeholder,
.stNumberInput > div > div > input::placeholder {{
    color: {c['text_faint']} !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea textarea:focus {{
    border-color: {c['accent']} !important;
    box-shadow: 0 0 0 3px rgba(14,116,144,.16) !important;
}}
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stTextArea label {{
    font-size: 14px !important; color: {c['text_muted']} !important;
    letter-spacing: .1em !important; text-transform: uppercase;
    font-weight: 700 !important;
}}
.stNumberInput button {{
    background: {c['surface_2']} !important; color: {c['text']} !important;
    border-color: {c['border']} !important;
}}
[data-baseweb="popover"] li {{ font-size: 16px !important; }}
[data-baseweb="select"] svg {{ fill: {c['text_muted']} !important; }}

/* ---------- expander / tables / misc ------------------------------------- */
div[data-testid="stExpander"] {{
    border: 1px solid {c['border']} !important;
    border-radius: 13px !important; background: {c['surface']} !important;
    box-shadow: 0 1px 3px rgba(12,27,42,.05);
}}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {{
    font-size: 16px !important; color: {c['text']} !important;
    font-weight: 600 !important;
}}
div[data-testid="stExpander"] svg {{ fill: {c['text_muted']} !important; }}

[data-testid="stImage"] img {{
    border-radius: 10px; border: 1px solid {c['border']}; background: #fff;
}}
hr {{ border: none; border-top: 1px solid {c['border']}; margin: 34px 0; }}

.stAlert {{ border-radius: 12px; font-size: 16px; }}
.stAlert p {{ font-size: 16px !important; }}

[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    font-size: 14.5px !important; color: {c['text_muted']} !important;
    line-height: 1.55;
}}
[data-testid="stMetricValue"] {{ font-family: {MONO_STACK}; }}

table {{ font-size: 15px !important; background: {c['surface']} !important; }}
thead th {{
    background: {c['surface_2']} !important; color: {c['text']} !important;
    font-weight: 700 !important; font-size: 14px !important;
}}
tbody td {{ color: {c['text_muted']} !important; }}

[data-testid="stSlider"] [role="slider"] {{ background: {c['accent']} !important; }}
[data-testid="stSlider"] {{ padding-bottom: 4px; }}
[data-testid="stSpinner"] p {{ font-size: 16px !important; }}

/* ---------- motion -------------------------------------------------------- */
@keyframes rfIn {{ from {{ opacity: 0; transform: translateY(8px); }}
                   to {{ opacity: 1; transform: none; }} }}
.rf-anim {{ animation: rfIn .34s cubic-bezier(.2,.7,.3,1) both; }}
@keyframes rfPulse {{ 0%,100% {{ opacity: .4; }} 50% {{ opacity: 1; }} }}
.rf-run {{ animation: rfPulse 1.25s ease-in-out infinite; }}
@media (prefers-reduced-motion: reduce) {{
    .rf-anim, .rf-run {{ animation: none !important; }}
}}

/* ---------- responsive ---------------------------------------------------- */
@media (max-width: 1100px) {{
    .rf-hero h1 {{ font-size: 42px; }}
    .rf-hero {{ padding: 40px 34px; }}
    .block-container {{ max-width: 100% !important; }}
}}
@media (max-width: 860px) {{
    .rf-grid3 {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 720px) {{
    html, body {{ font-size: 16px; }}
    .rf-hero {{ padding: 30px 22px; border-radius: 16px; }}
    .rf-hero h1 {{ font-size: 31px; max-width: none; }}
    .rf-hero p {{ font-size: 16.5px; }}
    .rf-verdict {{ padding: 30px 20px; }}
    .rf-verdict-head {{ font-size: 29px; }}
    .rf-card {{ padding: 18px 17px; }}
    .rf-appbar {{ margin-bottom: 24px; }}
    .rf-logo {{ font-size: 21px; }}
    .rf-brand-sub {{ display: none; }}
    .rf-steps {{ gap: 7px; }}
    .rf-step {{ font-size: 13.5px; padding: 10px 13px; }}
    .rf-model-p {{ font-size: 32px; }}
    .block-container {{ padding: 0 1rem 4rem 1rem !important; }}
}}
</style>
"""

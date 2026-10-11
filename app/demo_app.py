"""RADFRAC - fracture screening (two-model system).

Information architecture, replacing the old sidebar-driven layout:

    app bar  ->  step rail  ->  [1] upload  ->  [2] case details
             ->  [3] analysis  ->  result / models / agreement
             ->  viewer + explainability  ->  PDF report

Design decisions worth stating:
* No sidebar. Controls belong to the step they serve, so the first screen is a
  single obvious action.
* No model picker. Both active models always run and the frozen ensemble decides;
  asking a nurse to choose an architecture is not a real decision.
* No raw metrics JSON in the main flow - it lives behind "Technical details".
* The archived custom CNN is never referenced.

Launch::

    streamlit run app/demo_app.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.theme import BACKGROUND_FILENAME, COLORS, css, has_background_image  # noqa: E402
from src import config  # noqa: E402
from src.dataset_status import WARNING_BODY, WARNING_TITLE  # noqa: E402
from src.inference import ModelUnavailableError  # noqa: E402
from src.service import InvalidInputError, ScreeningService  # noqa: E402

st.set_page_config(page_title="RADFRAC — Fracture Screening",
                   page_icon="◎", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(css(), unsafe_allow_html=True)

# Body-region capture is deliberately omitted. It never fed either model -- both
# run on the image alone -- so it was a required field with no effect on the
# result. The `cases` table and the /cases endpoint still accept the column for
# other callers; this UI simply leaves it unset.


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    d = {
        "stage": "upload", "image_bytes": None, "filename": None,
        "result": None, "patient": {}, "pdf_bytes": None, "pdf_name": None,
        "zoom": 100,
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def _reset() -> None:
    for k in ("image_bytes", "filename", "result", "pdf_bytes", "pdf_name"):
        st.session_state[k] = None
    st.session_state.stage = "upload"
    st.session_state.zoom = 100


def _fit_width() -> dict:
    """st.image container-width kwarg, resolved across Streamlit versions."""
    import inspect
    p = inspect.signature(st.image).parameters
    if "use_container_width" in p:
        return {"use_container_width": True}
    if "use_column_width" in p:
        return {"use_column_width": True}
    return {"width": "stretch"}


FITW = _fit_width()


@st.cache_resource(show_spinner=False)
def get_service() -> ScreeningService:
    return ScreeningService.instance()


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
def app_bar() -> None:
    st.markdown(
        """
        <div class="rf-appbar">
          <div class="rf-brand">
            <div class="rf-logo">RAD<span>FRAC</span></div>
            <div class="rf-brand-sub">Point-of-care decision support</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def step_rail(active: str) -> None:
    steps = [("upload", "Upload"), ("details", "Case details"),
             ("analyze", "Analysis"), ("result", "Result")]
    order = [s[0] for s in steps]
    ai = order.index(active) if active in order else 0
    html = '<div class="rf-steps">'
    for i, (key, label) in enumerate(steps):
        cls = "active" if i == ai else ("done" if i < ai else "")
        mark = "✓" if i < ai else str(i + 1)
        html += (f'<div class="rf-step {cls}"><span class="rf-step-n">{mark}</span>'
                 f'{label}</div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="rf-hero rf-anim">
          <div class="rf-eyebrow">Decision support · not a diagnosis</div>
          <h1>Fracture screening</h1>
          <p>RADFRAC runs a classical computer-vision classifier and a pretrained
          convolutional network on every image, then combines their calibrated
          probabilities. Results require review by a qualified clinician.</p>
        </div>
        """, unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(
        '<div class="rf-disclaim"><b>Screening result — not a medical '
        'diagnosis.</b> RADFRAC supports triage where specialist review is not '
        'immediately available. A negative screen does not exclude fracture. '
        'Final interpretation rests with a qualified healthcare professional.</div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Stage 1 - upload
# --------------------------------------------------------------------------- #
def stage_upload() -> None:
    if not has_background_image():
        st.info(
            f"**Landing backdrop not installed.** Save the RADFRAC background "
            f"image as `app/assets/{BACKGROUND_FILENAME}` and reload — this notice "
            f"disappears on its own. A light gradient is used until then.",
            icon="🖼️")
    hero()
    step_rail("upload")

    st.markdown('<div class="rf-section-title">Step 1 · Upload radiograph</div>',
                unsafe_allow_html=True)
    up = st.file_uploader("Drag and drop a radiograph, or browse your files"
                         "  ·  PNG · JPG · JPEG · BMP",
                         type=["png", "jpg", "jpeg", "bmp"],
                         label_visibility="visible")
    if up is not None:
        st.session_state.image_bytes = up.getvalue()
        st.session_state.filename = up.name
        st.session_state.stage = "details"
        st.rerun()

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="rf-section-title">What happens next</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rf-grid3">
          <div class="rf-tile">
            <div class="rf-tile-n">01</div>
            <div class="rf-tile-t">Case details</div>
            <div class="rf-tile-d">Add the patient name and age, plus an optional
            case reference, so the result can be traced and reported.</div>
          </div>
          <div class="rf-tile">
            <div class="rf-tile-n">02</div>
            <div class="rf-tile-t">Automated analysis</div>
            <div class="rf-tile-d">The SVM and MobileNetV2 models score the image and
            their calibrated probabilities are combined into one result; XGBoost,
            LightGBM and CatBoost add a gradient-boosting second opinion.</div>
          </div>
          <div class="rf-tile">
            <div class="rf-tile-n">03</div>
            <div class="rf-tile-t">Result and report</div>
            <div class="rf-tile-d">Review the verdict, the highlighted regions and
            model agreement, then export a referral PDF.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    disclaimer()


# --------------------------------------------------------------------------- #
# Stage 2 - case details
# --------------------------------------------------------------------------- #
def stage_details(svc: ScreeningService) -> None:
    step_rail("details")
    img = Image.open(__import__("io").BytesIO(st.session_state.image_bytes))
    w, h = img.size

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="rf-section-title">Submitted image</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="rf-viewer">', unsafe_allow_html=True)
        st.markdown(f'<div class="rf-viewer-bar"><span>{st.session_state.filename}'
                    f'</span><span>{w} × {h} px</span></div>',
                    unsafe_allow_html=True)
        st.image(img, **FITW)
        st.markdown("</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Replace image", type="secondary"):
            _reset(); st.rerun()
        if c2.button("Remove", type="secondary"):
            _reset(); st.rerun()

    with right:
        st.markdown('<div class="rf-section-title">Case information</div>',
                    unsafe_allow_html=True)
        p = st.session_state.patient
        name = st.text_input("Patient name", value=p.get("name", ""),
                             placeholder="Full name", max_chars=80)
        a, b = st.columns(2)
        age = a.number_input("Age", min_value=0, max_value=120,
                            value=int(p.get("age", 0)) or 0, step=1)
        ref = b.text_input("Case reference (optional)", value=p.get("ref", ""),
                           placeholder="e.g. PHC-2026-0142", max_chars=60)

        problems = []
        if not name.strip():
            problems.append("Patient name is required.")
        if age <= 0:
            problems.append("Age must be greater than 0.")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Analyze X-ray", disabled=bool(problems)):
            st.session_state.patient = {"name": name.strip(), "age": int(age),
                                        "ref": ref.strip()}
            st.session_state.stage = "analyze"
            st.rerun()
        for msg in problems:
            st.caption(f"· {msg}")


# --------------------------------------------------------------------------- #
# Stage 3 - analysis
# --------------------------------------------------------------------------- #
def stage_analyze(svc: ScreeningService) -> None:
    step_rail("analyze")
    st.markdown('<div class="rf-section-title">Analyzing radiograph</div>',
                unsafe_allow_html=True)
    box = st.empty()
    lines = ["Image preprocessing & validity checks",
             "HOG + Canny + SVM",
             "MobileNetV2 transfer learning"]
    # Gradient-boosting models run on hybrid HOG + MobileNetV2 features (PCA-
    # reduced); listed only when they have been trained and loaded.
    from src.model_boosting import BOOSTER_KEYS, DISPLAY_NAMES as BOOST_NAMES
    boosters = [k for k in BOOSTER_KEYS if k in svc.advanced_models]
    if boosters:
        lines.append("Hybrid feature extraction (HOG + MobileNetV2 → PCA)")
        lines += [f"{BOOST_NAMES[k]} gradient boosting" for k in boosters]
    lines += ["Combining calibrated probabilities",
              "Generating screening result"]
    n_pre = len(lines) - 2      # steps shown as running before the screen call

    def render(done: int, running: int | None) -> None:
        html = '<div class="rf-card">'
        for i, txt in enumerate(lines):
            if i < done:
                html += (f'<div style="font-size:16.5px;padding:10px 0;font-weight:600;'
                         f'color:{COLORS["negative"]}">✓ &nbsp;{txt}</div>')
            elif i == running:
                html += (f'<div class="rf-run" style="font-size:16.5px;padding:10px 0;'
                         f'font-weight:600;color:{COLORS["accent"]}">→ &nbsp;{txt}</div>')
            else:
                html += (f'<div style="font-size:16.5px;padding:10px 0;color:'
                         f'{COLORS["text_faint"]}">· &nbsp;{txt}</div>')
        box.markdown(html + "</div>", unsafe_allow_html=True)

    render(0, 0)
    arr = np.asarray(
        Image.open(__import__("io").BytesIO(st.session_state.image_bytes)
                   ).convert("RGB"), dtype="uint8")
    time.sleep(0.25)
    render(1, 1)

    try:
        from src import db
        p = st.session_state.patient
        case_id = db.create_case(
            patient_ref=(p.get("ref") or p.get("name") or None),
            notes=f"name={p.get('name')}; age={p.get('age')}",
        )
        for i in range(2, n_pre + 1):
            render(i, i)
            time.sleep(0.15)
        result = svc.screen(arr, case_id=case_id,
                            original_filename=st.session_state.filename)
        render(len(lines) - 1, len(lines) - 1)
        time.sleep(0.15)
        render(len(lines), None)
    except InvalidInputError as exc:
        box.empty()
        st.error(f"**Image rejected.** {exc}", icon="🚫")
        for e in (exc.validity.get("errors") or []):
            st.caption(f"· {e}")
        st.info("Upload a genuine radiograph image to continue.")
        if st.button("Start over", type="secondary"):
            _reset(); st.rerun()
        return
    except Exception as exc:
        box.empty()
        st.error(f"Analysis failed: {exc}")
        if st.button("Start over", type="secondary"):
            _reset(); st.rerun()
        return

    st.session_state.result = result
    st.session_state.stage = "result"
    st.rerun()


# --------------------------------------------------------------------------- #
# Stage 4 - result
# --------------------------------------------------------------------------- #
def stage_result(svc: ScreeningService) -> None:
    r = st.session_state.result
    step_rail("result")

    pos = r.positive
    head = "FRACTURE INDICATED" if pos else "NO FRACTURE INDICATED"

    # The verdict is stated on its own. The ensemble probability, the decision
    # threshold and the probability meter used to sit here; a percentage invites
    # the reader to second-guess the threshold at the point of care, which is not
    # a judgement the screen should be asking for. Both numbers are still
    # recorded on the prediction row and printed in the referral PDF, and the
    # per-model probabilities remain under "Model analysis" below.
    st.markdown(
        f"""
        <div class="rf-verdict {'pos' if pos else 'neg'} rf-anim">
          <div class="rf-verdict-label">RADFRAC ensemble analysis</div>
          <div class="rf-verdict-head">{head}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    disclaimer()

    # ---- patient strip ----
    p = st.session_state.patient
    st.markdown('<div class="rf-section-title" style="margin-top:30px">Case</div>',
                unsafe_allow_html=True)
    fields = [
        ("Patient", p.get("name", "—")), ("Age", str(p.get("age", "—"))),
        ("Case reference", p.get("ref") or "—"),
        ("Date", datetime.now().strftime("%d %b %Y")),
    ]
    cols = st.columns(len(fields))
    for col, (k, v) in zip(cols, fields):
        col.markdown(
            f'<div class="rf-card" style="padding:18px 20px;margin:0;height:100%">'
            f'<div style="font-size:13px;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{COLORS["text_muted"]};margin-bottom:7px;font-weight:700">{k}</div>'
            f'<div style="font-size:19px;font-weight:700;color:{COLORS["text"]}">'
            f'{v}</div></div>',
            unsafe_allow_html=True)

    # ---- model breakdown ----
    st.markdown('<div class="rf-section-title" style="margin-top:34px">'
                'Model analysis</div>', unsafe_allow_html=True)
    cat = svc.model_catalogue()
    auc_of = {m["key"]: (m.get("test_metrics") or {}).get("roc_auc")
              for m in cat["active_models"]}
    kind_of = {"classical": "Classical computer vision",
               "transfer": "Deep transfer learning"}

    mcols = st.columns(2, gap="medium")
    for col, key in zip(mcols, ["classical", "transfer"]):
        info = r.per_model.get(key)
        if not info:
            continue
        mpos = info["label"] == config.POSITIVE_CLASS
        mcol = COLORS["positive"] if mpos else COLORS["negative"]
        auc = auc_of.get(key)
        col.markdown(
            f"""
            <div class="rf-model rf-anim">
              <div class="rf-model-name">{info['display_name']}</div>
              <div class="rf-model-kind">{kind_of.get(key, '')}</div>
              <div class="rf-model-verdict" style="color:{mcol}">
                  {'FRACTURED' if mpos else 'NOT FRACTURED'}</div>
              <div class="rf-model-p" style="color:{mcol}">
                  {info['probability'] * 100:.1f}%</div>
              <div class="rf-kv"><span>Decision threshold</span>
                  <b>{info['threshold']:.3f}</b></div>
              <div class="rf-kv"><span>Test ROC-AUC</span>
                  <b>{'—' if auc is None else f'{auc:.3f}'}</b></div>
            </div>
            """, unsafe_allow_html=True)

    # ---- agreement ----
    dis = r.disagreement or {}
    labels = dis.get("labels", {})
    split = len(set(labels.values())) > 1
    st.markdown('<div class="rf-section-title" style="margin-top:28px">'
                'Model agreement</div>', unsafe_allow_html=True)
    if split:
        detail = " · ".join(
            f"{r.per_model[k]['display_name']}: "
            f"{'FRACTURED' if v == config.POSITIVE_CLASS else 'NOT FRACTURED'}"
            for k, v in labels.items() if k in r.per_model)
        st.markdown(
            f'<div class="rf-agree warn"><span>⚠</span><div>'
            f'<div class="rf-agree-t" style="color:{COLORS["warn"]}">'
            f'MODELS DISAGREE</div><div class="rf-agree-d">{detail}. The ensemble '
            f'used its validated probability combination. Manual review is '
            f'recommended.</div></div></div>', unsafe_allow_html=True)
    else:
        spread = dis.get("spread")
        extra = (f" Probability spread {spread:.2f}." if isinstance(spread, float) else "")
        st.markdown(
            f'<div class="rf-agree ok"><span>✓</span><div>'
            f'<div class="rf-agree-t" style="color:{COLORS["negative"]}">'
            f'BOTH MODELS AGREE</div><div class="rf-agree-d">Two independent '
            f'approaches reached the same verdict.{extra}</div></div></div>',
            unsafe_allow_html=True)

    advanced_panel(r, cat)

    if r.recommend_manual_review:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.warning("Flagged for manual review — see agreement and image-quality notes.",
                   icon="⚠️")

    # ---- viewer + explainability ----
    st.markdown('<div class="rf-section-title" style="margin-top:34px">'
                'Radiograph viewer</div>', unsafe_allow_html=True)
    vleft, vright = st.columns([1.25, 1], gap="large")
    img = Image.open(__import__("io").BytesIO(st.session_state.image_bytes)).convert("RGB")
    with vleft:
        zoom = st.slider("Zoom", 50, 250, st.session_state.zoom, 10)
        zc1, zc2 = st.columns(2, gap="small")
        if zc1.button("Fit to width", type="secondary"):
            zoom = 100; st.session_state.zoom = 100
        if zc2.button("Reset zoom", type="secondary"):
            zoom = 100; st.session_state.zoom = 100
        st.session_state.zoom = zoom
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="rf-viewer">', unsafe_allow_html=True)
        st.markdown(f'<div class="rf-viewer-bar"><span>{st.session_state.filename}'
                    f'</span><span>{img.size[0]} × {img.size[1]} px · {zoom}%</span>'
                    f'</div>', unsafe_allow_html=True)
        # Scale by width only so the aspect ratio is never distorted.
        base = 560
        st.image(img, width=int(base * zoom / 100))
        st.markdown("</div>", unsafe_allow_html=True)

    with vright:
        st.markdown(
            f'<div style="font-size:14px;letter-spacing:.14em;text-transform:uppercase;'
            f'color:{COLORS["text_muted"]};margin-bottom:10px;font-weight:700">'
            f'Explainability</div>',
            unsafe_allow_html=True)
        if r.gradcam_path and Path(r.gradcam_path).exists():
            st.markdown('<div style="font-size:16px;font-weight:700;'
                        'margin-bottom:7px">MobileNetV2 — LayerCAM</div>',
                        unsafe_allow_html=True)
            st.image(str(r.gradcam_path), **FITW)
            st.caption("Regions that most influenced the MobileNetV2 score. "
                       "This explains the neural model only — not the SVM.")
        if r.classical_explain_path and Path(r.classical_explain_path).exists():
            st.markdown('<div style="font-size:16px;font-weight:700;'
                        'margin:18px 0 7px 0">HOG + Canny + SVM — input features</div>',
                        unsafe_allow_html=True)
            st.image(str(r.classical_explain_path), **FITW)
            st.caption("The gradient and edge features the SVM actually receives.")

    # ---- note ----
    with st.expander("Screening note"):
        st.write(r.screening_note)

    # ---- report ----
    st.markdown('<div class="rf-section-title" style="margin-top:34px">'
                'Referral report</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 1], gap="medium")
    with rc1:
        if st.button("Generate PDF report"):
            with st.spinner("Building report…"):
                try:
                    from src import db
                    from src.referral_report import build_referral_pdf
                    payload = r.to_dict()
                    payload["patient"] = st.session_state.patient
                    pdf = build_referral_pdf(payload)
                    if r.prediction_id:
                        db.update_prediction(r.prediction_id,
                                             {"report_path": str(pdf)})
                    st.session_state.pdf_bytes = Path(pdf).read_bytes()
                    st.session_state.pdf_name = (
                        f"RADFRAC_{(st.session_state.patient.get('name') or 'case')
                                   .replace(' ', '_')}_{r.prediction_id}.pdf")
                except Exception as exc:
                    st.error(f"Could not generate the report: {exc}")
    with rc2:
        if st.session_state.pdf_bytes:
            st.download_button("Download PDF", data=st.session_state.pdf_bytes,
                               file_name=st.session_state.pdf_name,
                               mime="application/pdf")
    if st.session_state.pdf_bytes:
        st.caption("Report includes patient details, the radiograph, both model "
                   "results, agreement and explainability.")

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Screen another X-ray", type="secondary"):
        _reset(); st.rerun()

    with st.expander("Technical details — models, performance, method"):
        technical_panel()


def advanced_panel(r, cat: dict) -> None:
    """Gradient-boosting results: XGBoost, LightGBM, CatBoost and their committee.

    Shown as a second opinion next to the ensemble. The headline verdict above
    is still the validated two-model ensemble unless configured otherwise.
    """
    adv = (getattr(r, "advanced", None) or {}).get("models") or {}
    if not adv:
        return
    st.markdown('<div class="rf-section-title" style="margin-top:34px">'
                'Advanced models · gradient boosting</div>', unsafe_allow_html=True)
    st.caption("Trained on hybrid features — HOG + Canny edges and MobileNetV2 "
               "embeddings, reduced with PCA — and tuned with Optuna. Each model "
               "is judged at its own validated threshold.")

    auc_of = {m["key"]: (m.get("test_metrics") or {}).get("roc_auc")
              for m in cat.get("advanced_models", [])}
    keys = [k for k in ("xgboost", "lightgbm", "catboost", "boost_committee") if k in adv]
    cols = st.columns(len(keys), gap="small")
    for col, key in zip(cols, keys):
        info = adv[key]
        mpos = info["label"] == config.POSITIVE_CLASS
        mcol = COLORS["positive"] if mpos else COLORS["negative"]
        auc = auc_of.get(key)
        col.markdown(
            f"""
            <div class="rf-model rf-anim">
              <div class="rf-model-name">{info['display_name']}</div>
              <div class="rf-model-kind">{info.get('description', '')}</div>
              <div class="rf-model-verdict" style="color:{mcol}">
                  {'FRACTURED' if mpos else 'NOT FRACTURED'}</div>
              <div class="rf-model-p" style="color:{mcol}">
                  {info['probability'] * 100:.1f}%</div>
              <div class="rf-kv"><span>Threshold</span>
                  <b>{info['threshold']:.3f}</b></div>
              <div class="rf-kv"><span>Test ROC-AUC</span>
                  <b>{'—' if auc is None else f'{auc:.3f}'}</b></div>
            </div>
            """, unsafe_allow_html=True)

    agreement = r.advanced.get("agreement") or {}
    same = r.advanced.get("agrees_with_primary")
    boost_labels = set((agreement.get("labels") or {}).values())
    if len(boost_labels) > 1:
        st.markdown(
            f'<div class="rf-agree warn"><span>⚠</span><div>'
            f'<div class="rf-agree-t" style="color:{COLORS["warn"]}">BOOSTERS SPLIT</div>'
            f'<div class="rf-agree-d">The gradient-boosting models do not agree '
            f'with each other on this image.</div></div></div>', unsafe_allow_html=True)
    elif same:
        st.markdown(
            f'<div class="rf-agree ok"><span>✓</span><div>'
            f'<div class="rf-agree-t" style="color:{COLORS["negative"]}">'
            f'BOOSTERS CONFIRM THE RESULT</div><div class="rf-agree-d">All three '
            f'gradient-boosting models reach the same verdict as the main result.'
            f'</div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="rf-agree warn"><span>⚠</span><div>'
            f'<div class="rf-agree-t" style="color:{COLORS["warn"]}">'
            f'BOOSTERS DIFFER FROM THE MAIN RESULT</div><div class="rf-agree-d">'
            f'The gradient-boosting models agree with each other but not with the '
            f'ensemble verdict. Manual review is recommended.</div></div></div>',
            unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Technical panel (kept OUT of the main flow)
# --------------------------------------------------------------------------- #
def technical_panel() -> None:
    try:
        svc = get_service()
    except ModelUnavailableError:
        st.caption("Models unavailable.")
        return
    cat = svc.model_catalogue()

    st.markdown("**Active models**")
    rows = []
    for m in cat["active_models"]:
        tm = m.get("test_metrics") or {}
        rows.append({
            "Model": m["name"],
            "Accuracy": f"{tm.get('accuracy'):.3f}" if tm.get("accuracy") else "—",
            "Precision": f"{tm.get('precision'):.3f}" if tm.get("precision") else "—",
            "Recall": f"{tm.get('recall'):.3f}" if tm.get("recall") else "—",
            "F1": f"{tm.get('f1'):.3f}" if tm.get("f1") else "—",
            "ROC-AUC": f"{tm.get('roc_auc'):.3f}" if tm.get("roc_auc") else "—",
            "Threshold": f"{m['threshold']:.3f}",
            "Explainability": m["explainability"],
        })
    ens = cat.get("ensemble")
    if ens:
        tm = ens.get("test_metrics") or {}
        rows.append({
            "Model": ens["name"],
            "Accuracy": f"{tm.get('accuracy'):.3f}" if tm.get("accuracy") else "—",
            "Precision": f"{tm.get('precision'):.3f}" if tm.get("precision") else "—",
            "Recall": f"{tm.get('recall'):.3f}" if tm.get("recall") else "—",
            "F1": f"{tm.get('f1'):.3f}" if tm.get("f1") else "—",
            "ROC-AUC": f"{tm.get('roc_auc'):.3f}" if tm.get("roc_auc") else "—",
            "Threshold": f"{ens['threshold']:.3f}",
            "Explainability": "inherits member explanations",
        })
    for m in cat.get("advanced_models", []):
        tm = m.get("test_metrics") or {}
        rows.append({
            "Model": m["name"],
            "Accuracy": f"{tm.get('accuracy'):.3f}" if tm.get("accuracy") else "—",
            "Precision": f"{tm.get('precision'):.3f}" if tm.get("precision") else "—",
            "Recall": f"{tm.get('recall'):.3f}" if tm.get("recall") else "—",
            "F1": f"{tm.get('f1'):.3f}" if tm.get("f1") else "—",
            "ROC-AUC": f"{tm.get('roc_auc'):.3f}" if tm.get("roc_auc") else "—",
            "Threshold": f"{m['threshold']:.3f}" if m.get("threshold") else "—",
            "Explainability": m["explainability"],
        })
    st.table(rows)
    if cat.get("advanced_models"):
        st.caption("Gradient-boosting models: StandardScaler → PCA → booster on "
                   "hybrid HOG + MobileNetV2 features; PCA size and all "
                   "hyperparameters chosen by Optuna (TPE) on the validation split, "
                   "trials logged to MLflow.")

    if ens:
        w = ens["weights"]
        st.caption(
            f"Ensemble: {ens['method']} · weights SVM {w.get('classical', 0):.2f} / "
            f"MobileNetV2 {w.get('transfer', 0):.2f} · threshold {ens['threshold']:.3f}. "
            f"Calibration (Platt), weights and threshold were fitted on the "
            f"validation split only; the test split was scored once."
        )
    if cat.get("verdict"):
        st.caption(f"Ensemble vs best single model: {cat['verdict']}")
    st.caption(
        "Evaluation uses a group-aware, leakage-free split: near-duplicate and "
        "augmented copies of the same underlying film are kept within one split. "
        "Preprocessing centre-crops to 72% before resizing to 128×128 to remove "
        "label-correlated border artefacts."
    )
    st.caption("A custom CNN was evaluated during development and is archived; "
               "it is not part of the active system.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    app_bar()

    try:
        svc = get_service()
    except ModelUnavailableError as exc:
        st.error(f"No trained models available. {exc}")
        st.code("python -m src.train --model transfer\n"
                "python -m src.train --model classical\n"
                "python -m scripts.fit_ensemble", language="bash")
        return

    if svc.synthetic:
        st.error(f"**{WARNING_TITLE}** — {WARNING_BODY}", icon="⚠️")

    stage = st.session_state.stage
    if stage == "upload" or st.session_state.image_bytes is None:
        stage_upload()
    elif stage == "details":
        stage_details(svc)
    elif stage == "analyze":
        stage_analyze(svc)
    elif stage == "result" and st.session_state.result is not None:
        stage_result(svc)
    else:
        stage_upload()


if __name__ == "__main__":
    main()

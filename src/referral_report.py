"""Designed PDF referral report for a RADFRAC screening.

This document often travels with the patient to the next facility, so it is
treated as a real deliverable rather than a debug dump. Layout intent:

* A dark branded header band carrying the RADFRAC identity, so the page is
  recognisable at a glance and matches the web application.
* Patient details first - a receiving clinician needs to know *who* before *what*.
* One unambiguous result block, then the two member models side by side, then
  agreement, then the images.
* Explicit framing on every page: screening support, not a diagnosis.
* Page numbers and a footer rule, so it reads as a report and not a printout.

Only the two active models appear. The archived CNN is never referenced.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import config

DISCLAIMER = (
    "This system provides AI-assisted screening support and is not a medical "
    "diagnosis. Results should be reviewed by a qualified healthcare "
    "professional. A negative screen does not exclude fracture. RADFRAC has not "
    "been clinically validated and must not be used as the sole basis for any "
    "care decision."
)

# Identity shared with the web app
INK = "#0a0e13"
SURFACE = "#f4f7fa"
BORDER = "#d3dce6"
MUTED = "#5d6b7e"
ACCENT = "#0f9e8e"
POSITIVE = "#c2410c"      # a finding - attention
NEGATIVE = "#15803d"
WARN = "#b45309"


def _fmt_pct(v) -> str:
    return "—" if v is None else f"{float(v) * 100:.1f}%"


def _member_rows(result: dict) -> list[tuple[str, dict]]:
    """Active members in a stable display order, archived keys excluded."""
    from .ensemble import MEMBER_LABELS, MEMBERS

    out = []
    per = result.get("per_model") or {}
    for key in MEMBERS:
        if key in per:
            info = dict(per[key])
            info.setdefault("display_name", MEMBER_LABELS.get(key, key))
            out.append((key, info))
    # Gradient-boosting second opinion, when present.
    adv = (result.get("advanced") or {}).get("models") or {}
    for key, info in adv.items():
        info = dict(info)
        info["display_name"] = f"{info.get('display_name', key)} (gradient boosting)"
        out.append((key, info))
    return out


def build_referral_pdf(result: dict, out_path: str | Path | None = None) -> Path:
    """Render a screening result into a designed A4 PDF. Returns the path."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image as RLImage,
        KeepTogether,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    pred_id = result.get("prediction_id") or result.get("id") or "unsaved"
    patient = result.get("patient") or {}

    if out_path is None:
        out_dir = config.CASE_STORE_DIR / str(pred_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "radfrac_report.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=9, leading=13, textColor=colors.HexColor(INK))
    small = ParagraphStyle("s", parent=body, fontSize=7.4, leading=10.2,
                           textColor=colors.HexColor(MUTED))
    label = ParagraphStyle("l", parent=body, fontSize=6.8, leading=9,
                           textColor=colors.HexColor(MUTED))
    value = ParagraphStyle("v", parent=body, fontSize=10.5, leading=13.5,
                           fontName="Helvetica-Bold")
    sect = ParagraphStyle("sec", parent=body, fontSize=7.6, leading=10,
                          fontName="Helvetica-Bold",
                          textColor=colors.HexColor(MUTED))

    # ---------------- page furniture -------------------------------------- #
    def decorate(canvas, doc):
        canvas.saveState()
        W, H = A4
        # header band
        canvas.setFillColor(colors.HexColor(INK))
        canvas.rect(0, H - 30 * mm, W, 30 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 17)
        canvas.drawString(16 * mm, H - 16 * mm, "RAD")
        tw = canvas.stringWidth("RAD", "Helvetica-Bold", 17)
        canvas.setFillColor(colors.HexColor("#2dd4bf"))
        canvas.drawString(16 * mm + tw, H - 16 * mm, "FRAC")
        canvas.setFillColor(colors.HexColor("#9fb0c3"))
        canvas.setFont("Helvetica", 7.6)
        canvas.drawString(16 * mm, H - 21.5 * mm,
                          "AI-ASSISTED FRACTURE SCREENING  ·  SCREENING SUPPORT, NOT A DIAGNOSIS")
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(colors.HexColor("#9fb0c3"))
        canvas.drawRightString(W - 16 * mm, H - 16 * mm,
                               datetime.now().strftime("%d %b %Y  %H:%M"))
        canvas.drawRightString(W - 16 * mm, H - 21.5 * mm, f"CASE {pred_id}")
        # footer
        canvas.setStrokeColor(colors.HexColor(BORDER))
        canvas.setLineWidth(.5)
        canvas.line(16 * mm, 14 * mm, W - 16 * mm, 14 * mm)
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(16 * mm, 10 * mm,
                          "RADFRAC · AI-assisted screening · not a medical diagnosis")
        canvas.drawRightString(W - 16 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=36 * mm, bottomMargin=18 * mm,
        title=f"RADFRAC screening report {pred_id}",
        author="RADFRAC",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])

    flow = []
    W = doc.width

    def band(text, bg, fg, bold=True):
        return Table([[Paragraph(text, ParagraphStyle(
            "x", parent=body, fontSize=8, leading=11,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            textColor=colors.HexColor(fg)))]], colWidths=[W],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))

    # ---------------- synthetic-data guard -------------------------------- #
    if result.get("synthetic_data"):
        flow.append(band(
            "WARNING — SYNTHETIC TRAINING DATA. This model was trained on "
            "artificially generated images. The output below has no clinical "
            "validity.", "#fdeaea", "#b03030"))
        flow.append(Spacer(1, 5 * mm))

    # ---------------- patient --------------------------------------------- #
    flow.append(Paragraph("PATIENT INFORMATION", sect))
    flow.append(Spacer(1, 2.2 * mm))

    def field(lbl, val):
        return [Paragraph(lbl.upper(), label), Paragraph(str(val or "—"), value)]

    # Body region used to occupy the first half of row 2. The UI no longer
    # collects it (it never fed the models), so the row would have printed a
    # permanent em dash; the case reference is spanned across instead.
    pt = Table(
        [field("Name", patient.get("name")) + field("Age", patient.get("age")),
         field("Case reference", patient.get("ref") or pred_id) + ["", ""]],
        colWidths=[W * 0.28, W * 0.22, W * 0.28, W * 0.22],
        style=TableStyle([
            ("SPAN", (1, 1), (3, 1)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(SURFACE)),
            ("BOX", (0, 0), (-1, -1), .6, colors.HexColor(BORDER)),
            ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor(BORDER)),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
    flow.append(pt)
    flow.append(Spacer(1, 6 * mm))

    # ---------------- headline result ------------------------------------- #
    positive = result.get("label") == config.POSITIVE_CLASS
    accent = POSITIVE if positive else NEGATIVE
    verdict = "FRACTURE INDICATED" if positive else "NO FRACTURE INDICATED"
    prob = result.get("probability")

    flow.append(Paragraph("RADFRAC ENSEMBLE RESULT", sect))
    flow.append(Spacer(1, 2.2 * mm))
    res_tbl = Table([[
        Paragraph(f"<font size=15><b>{verdict}</b></font><br/>"
                  f"<font size=7.4 color='{MUTED}'>Two-model ensemble · "
                  f"decision threshold {float(result.get('threshold', 0)):.3f}</font>",
                  ParagraphStyle("r", parent=body, textColor=colors.HexColor(accent),
                                 leading=19)),
        Paragraph(f"<font size=24><b>{_fmt_pct(prob)}</b></font><br/>"
                  f"<font size=7 color='{MUTED}'>ENSEMBLE PROBABILITY</font>",
                  ParagraphStyle("p", parent=body, alignment=TA_CENTER,
                                 textColor=colors.HexColor(INK), leading=27)),
    ]], colWidths=[W * 0.62, W * 0.38],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(SURFACE)),
            ("BOX", (0, 0), (-1, -1), .6, colors.HexColor(BORDER)),
            ("LINEBEFORE", (0, 0), (0, 0), 2.4, colors.HexColor(accent)),
            ("LEFTPADDING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ]))
    flow.append(res_tbl)
    flow.append(Spacer(1, 6 * mm))

    # ---------------- member models --------------------------------------- #
    members = _member_rows(result)
    if members:
        flow.append(Paragraph("MODEL RESULTS", sect))
        flow.append(Spacer(1, 2.2 * mm))
        head = ["Model", "Prediction", "Probability", "Threshold"]
        rows = [head]
        for _k, info in members:
            mpos = info.get("label") == config.POSITIVE_CLASS
            rows.append([
                info.get("display_name", "—"),
                "FRACTURED" if mpos else "NOT FRACTURED",
                _fmt_pct(info.get("probability")),
                f"{float(info.get('threshold', 0)):.3f}",
            ])
        mt = Table(rows, colWidths=[W * 0.40, W * 0.24, W * 0.18, W * 0.18])
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor(BORDER)),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        for i, (_k, info) in enumerate(members, start=1):
            mpos = info.get("label") == config.POSITIVE_CLASS
            style.append(("TEXTCOLOR", (1, i), (1, i),
                          colors.HexColor(POSITIVE if mpos else NEGATIVE)))
            style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafcfe")))
        mt.setStyle(TableStyle(style))
        flow.append(mt)
        flow.append(Spacer(1, 5 * mm))

        # agreement
        labels = {k: info.get("label") for k, info in members}
        agree = len(set(labels.values())) == 1
        if agree:
            flow.append(band("BOTH MODELS AGREE — two independent approaches "
                             "reached the same verdict.", "#eefaf1", NEGATIVE))
        else:
            detail = " · ".join(
                f"{info.get('display_name')}: "
                f"{'FRACTURED' if info.get('label') == config.POSITIVE_CLASS else 'NOT FRACTURED'}"
                for _k, info in members)
            flow.append(band(f"MODELS DISAGREE — {detail}. The ensemble used its "
                             f"validated probability combination; manual review is "
                             f"recommended.", "#fff7e6", WARN))
        flow.append(Spacer(1, 6 * mm))

    # ---------------- imaging --------------------------------------------- #
    img_path = result.get("image_path")
    gc_path = result.get("gradcam_path")
    cl_path = result.get("classical_explain_path")

    if img_path and Path(img_path).exists():
        flow.append(Paragraph("RADIOGRAPH", sect))
        flow.append(Spacer(1, 2.2 * mm))
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                iw, ih = im.size
            disp_w = W * 0.56
            disp_h = disp_w * ih / max(iw, 1)      # preserve aspect ratio
            max_h = 78 * mm
            if disp_h > max_h:
                disp_h, disp_w = max_h, max_h * iw / max(ih, 1)
            flow.append(KeepTogether([
                RLImage(str(img_path), width=disp_w, height=disp_h),
                Spacer(1, 1.5 * mm),
                Paragraph(f"Submitted image · {iw} × {ih} px", small),
            ]))
        except Exception:
            pass
        flow.append(Spacer(1, 5 * mm))

    expl = [(gc_path, "MobileNetV2 — LayerCAM attention",
             "Highlights regions that most influenced the MobileNetV2 score. "
             "Explains the neural model only, not the SVM."),
            (cl_path, "HOG + Canny + SVM — input features",
             "The gradient-orientation and edge features supplied to the SVM.")]
    shown = [(p, t, c) for p, t, c in expl if p and Path(p).exists()]
    if shown:
        flow.append(Paragraph("EXPLAINABILITY", sect))
        flow.append(Spacer(1, 2.2 * mm))
        for p, title, cap in shown:
            try:
                from PIL import Image as PILImage
                with PILImage.open(p) as im:
                    iw, ih = im.size
                dw = W * 0.80
                dh = dw * ih / max(iw, 1)
                if dh > 58 * mm:
                    dh, dw = 58 * mm, 58 * mm * iw / max(ih, 1)
                flow.append(KeepTogether([
                    Paragraph(f"<b>{title}</b>", body),
                    Spacer(1, 1.4 * mm),
                    RLImage(str(p), width=dw, height=dh),
                    Spacer(1, 1.2 * mm),
                    Paragraph(cap, small),
                    Spacer(1, 4 * mm),
                ]))
            except Exception:
                continue

    # ---------------- technical ------------------------------------------- #
    ens = result.get("ensemble") or {}
    unc = result.get("uncertainty") or {}
    weights = ens.get("weights") or {}
    tech = [
        ["Preprocessing", "RGB · centre crop 72% · resize 128×128 · scale [0,1]"],
        ["Active models", "HOG+Canny+SVM · MobileNetV2 transfer learning"],
        ["Ensemble method", f"{ens.get('method') or '—'} "
                            f"(SVM {weights.get('classical', 0):.2f} / "
                            f"MobileNetV2 {weights.get('transfer', 0):.2f})"],
        ["Decision threshold", f"{float(result.get('threshold', 0)):.3f} "
                               f"(fitted on validation data)"],
        ["Prediction stability",
         "n/a" if not unc.get("supported")
         else f"MC-dropout sd {unc.get('std'):.3f} over {unc.get('passes')} passes"],
        ["Manual review", "YES" if result.get("recommend_manual_review") else "No"],
    ]
    flow.append(Paragraph("TECHNICAL INFORMATION", sect))
    flow.append(Spacer(1, 2.2 * mm))
    tt = Table([[Paragraph(k, label), Paragraph(v, body)] for k, v in tech],
               colWidths=[W * 0.30, W * 0.70],
               style=TableStyle([
                   ("GRID", (0, 0), (-1, -1), .4, colors.HexColor(BORDER)),
                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                   ("LEFTPADDING", (0, 0), (-1, -1), 7),
                   ("TOPPADDING", (0, 0), (-1, -1), 5),
                   ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
               ]))
    flow.append(tt)

    validity = result.get("validity") or {}
    if validity.get("warnings"):
        flow.append(Spacer(1, 4 * mm))
        flow.append(Paragraph("IMAGE-QUALITY NOTES", sect))
        for w_ in validity["warnings"]:
            flow.append(Paragraph(f"• {w_}", small))

    if result.get("screening_note"):
        flow.append(Spacer(1, 4 * mm))
        flow.append(Paragraph("SCREENING NOTE", sect))
        flow.append(Spacer(1, 1.6 * mm))
        flow.append(Paragraph(str(result["screening_note"]), body))

    # ---------------- disclaimer ------------------------------------------ #
    flow.append(Spacer(1, 6 * mm))
    flow.append(Table([[Paragraph(f"<b>DISCLAIMER</b><br/>{DISCLAIMER}", small)]],
                      colWidths=[W],
                      style=TableStyle([
                          ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9ec")),
                          ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#e6d3a8")),
                          ("LEFTPADDING", (0, 0), (-1, -1), 9),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                          ("TOPPADDING", (0, 0), (-1, -1), 7),
                          ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                      ])))

    doc.build(flow)
    return out_path

"""RadFrac HTTP API (6.2) - FastAPI wrapper over the service layer.

The API is a thin transport shell: all orchestration, validation and
persistence live in ``src.service`` so the Streamlit UI, this API, and any
future frontend behave identically (6.4).

Endpoints
---------
``GET  /health``                    liveness + which models are loaded
``GET  /models``                   available models (incl. boosters) + test-set metrics
``POST /predict``                  single image -> full screening result
``POST /predict/batch``            several images (ER queue use case)
``GET  /predict/{id}``             fetch a stored prediction
``GET  /predict/{id}/gradcam``     Grad-CAM heatmap PNG
``GET  /predict/{id}/report``      generate/fetch the referral PDF
``GET  /predictions``              recent predictions
``GET  /triage``                   predictions re-sorted by urgency

Run::

    python -m uvicorn api.main:app --reload --port 8000

Security note: this ships with **no authentication**, which is fine for
localhost / single-machine clinic use but must not be exposed to a network
without an auth layer and TLS in front of it.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when uvicorn loads this module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src import config, db  # noqa: E402
from src.inference import ModelUnavailableError  # noqa: E402
from src.service import InvalidInputError, ScreeningService  # noqa: E402

app = FastAPI(
    title="RadFrac Screening API",
    version="1.0.0",
    description=(
        "AI-assisted fracture screening and triage. Outputs are screening aids "
        "for a qualified clinician and are never diagnoses."
    ),
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB guard against oversized uploads


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class CaseCreate(BaseModel):
    patient_ref: str | None = Field(None, max_length=120)
    body_region: str | None = Field(None, max_length=80)
    notes: str | None = Field(None, max_length=2000)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _service() -> ScreeningService:
    try:
        return ScreeningService.instance()
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


def _read_image(upload: UploadFile) -> np.ndarray:
    """Read and decode an upload synchronously.

    Deliberately sync: the prediction handlers are plain ``def`` so FastAPI
    runs them in its threadpool instead of blocking the event loop on
    TensorFlow inference. ``UploadFile.file`` is a spooled file object, so it
    can be read directly here.

    Reads one byte past the limit so an oversized body is rejected without
    buffering all of it (the previous version read the whole body into RAM
    before checking the size, which made the guard useless).
    """
    try:
        upload.file.seek(0)
        raw = upload.file.read(MAX_UPLOAD_BYTES + 1)
    except OSError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "could not read uploaded file") from exc
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "uploaded file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"), dtype="uint8")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"could not decode image: {exc}") from exc


@app.on_event("startup")
def _startup() -> None:
    config.ensure_runtime_dirs()
    db.init_db()
    try:
        svc = ScreeningService.instance()
        print(f"[radfrac-api] models loaded: {svc.loaded_models}")
        if svc.synthetic:
            print("[radfrac-api] WARNING: models were trained on SYNTHETIC data")
    except ModelUnavailableError as exc:
        # Start anyway so /health can report the problem clearly.
        print(f"[radfrac-api] startup warning: {exc}")


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    # Catch *everything*: a corrupt .keras file or a TensorFlow import failure
    # raises something other than ModelUnavailableError, and a 500 here makes
    # an ALB/ECS health check kill the task instead of reporting "degraded".
    detail = None
    try:
        svc = ScreeningService.instance()
        models = svc.loaded_models
        advanced = svc.advanced_models
        synthetic = svc.synthetic
    except Exception as exc:  # noqa: BLE001 - health must never raise
        models, advanced, synthetic = [], [], None
        detail = f"service unavailable: {type(exc).__name__}"
        print(f"[radfrac-api] /health degraded: {exc!r}")
    return {
        "status": "ok" if models else "degraded",
        "models_loaded": models,
        "advanced_models": advanced,
        "synthetic_data": synthetic,
        "detail": (detail if detail
                   else None if models
                   else "no trained models found - run src.train"),
    }


@app.get("/models")
def models() -> dict:
    """Active models + the frozen ensemble. The archived CNN is never listed."""
    svc = _service()
    try:
        return svc.model_catalogue()
    except ModelUnavailableError as exc:
        # No models loaded is a 503 (service not ready), not a 500.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.get("/ensemble")
def ensemble() -> dict:
    svc = _service()
    cfg = svc.ensemble_config
    if not cfg or not svc.ensemble_available():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "ensemble not fitted; run scripts.fit_ensemble")
    return {
        "method": cfg.method, "members": cfg.members, "weights": cfg.weights,
        "threshold": cfg.threshold, "threshold_source": cfg.threshold_source,
        "selected_on": cfg.selected_on, "notes": cfg.notes,
    }


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="X-ray image (png/jpg/bmp)"),
    model: str | None = Form(None, description="ensemble | classical | transfer | xgboost | lightgbm | catboost | boost_committee"),
    case_id: str | None = Form(None),
    cross_check: bool = Form(True),
    with_uncertainty: bool = Form(True),
) -> JSONResponse:
    # Plain ``def``, not ``async def``: screening is seconds of blocking
    # TensorFlow work (inference + Grad-CAM + 40 MC-dropout passes). On an
    # async handler that runs on the event loop and stalls every other
    # request, including /health. FastAPI offloads sync handlers to a
    # threadpool, which is what this workload needs.
    svc = _service()
    arr = _read_image(file)
    allowed = (list(svc.loaded_models) + svc.advanced_models
               + ([config.ENSEMBLE_KEY] if svc.ensemble_available() else []))
    if model and model not in allowed:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown model '{model}'; available: {allowed}")
    # Validate the case up front: an unknown case_id would otherwise trip the
    # predictions.case_id foreign key inside _persist and surface as a 500
    # *after* the full model run had already been paid for.
    if case_id and not db.get_case(case_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"unknown case_id '{case_id}'")
    try:
        result = svc.screen(
            arr, model_key=model, cross_check=cross_check,
            with_uncertainty=with_uncertainty, case_id=case_id,
            original_filename=file.filename,
        )
    except InvalidInputError as exc:
        # 422: the upload decoded fine but is not a usable diagnostic image.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "validity": exc.validity,
                     "rejected": True},
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return JSONResponse(content=result.to_dict())


@app.post("/predict/batch")
def predict_batch(
    files: list[UploadFile] = File(...),
    model: str | None = Form(None),
    with_uncertainty: bool = Form(False),
) -> dict:
    """Screen several images. Returns results plus an urgency-ordered queue.

    This supports the ER worklist use case: it re-orders an existing queue by
    urgency. It does not replace sequential human review of each case.
    """
    svc = _service()
    results, errors = [], []
    for f in files:
        try:
            arr = _read_image(f)
            res = svc.screen(arr, model_key=model, with_uncertainty=with_uncertainty,
                             original_filename=f.filename)
            payload = res.to_dict()
            payload["filename"] = f.filename
            results.append(payload)
        except InvalidInputError as exc:
            errors.append({"filename": f.filename, "rejected": True,
                           "detail": str(exc), "validity": exc.validity})
        except HTTPException as exc:
            errors.append({"filename": f.filename, "detail": exc.detail})
        except Exception as exc:  # noqa: BLE001
            # One bad image must not discard the whole batch (and every
            # already-computed result) with a 500.
            print(f"[radfrac-api] batch item failed ({f.filename}): {exc!r}")
            errors.append({"filename": f.filename,
                           "detail": f"screening failed: {type(exc).__name__}"})

    def urgency(r: dict) -> tuple:
        prob = r.get("probability")
        prob = prob if isinstance(prob, (int, float)) else 0.0
        return (r["label"] != config.POSITIVE_CLASS,
                not r.get("recommend_manual_review"),
                -prob)

    return {
        "count": len(results),
        "rejected": errors,
        "results": results,
        "triage_order": [r["prediction_id"] for r in sorted(results, key=urgency)
                         if r.get("prediction_id")],
    }


@app.get("/predict/{prediction_id}")
def get_prediction(prediction_id: str) -> dict:
    rec = db.get_prediction(prediction_id)
    if not rec:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction not found")
    return rec


@app.get("/predict/{prediction_id}/gradcam")
def get_gradcam(prediction_id: str):
    rec = db.get_prediction(prediction_id)
    if not rec:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction not found")
    path = rec.get("gradcam_path")
    if not path or not Path(path).exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "no Grad-CAM available for this prediction (classical model, or "
            "explainability was skipped)")
    return FileResponse(path, media_type="image/png",
                        filename=f"gradcam_{prediction_id}.png")


@app.get("/predict/{prediction_id}/report")
def get_report(prediction_id: str):
    rec = db.get_prediction(prediction_id)
    if not rec:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction not found")

    existing = rec.get("report_path")
    if existing and Path(existing).exists():
        return FileResponse(existing, media_type="application/pdf",
                            filename=f"radfrac_referral_{prediction_id}.pdf")

    from src.referral_report import build_referral_pdf

    payload = dict(rec)
    payload["prediction_id"] = prediction_id
    payload["recommend_manual_review"] = bool(
        rec.get("disagreement_flag") or rec.get("uncertainty_flag")
        or (rec.get("validity") or {}).get("warnings")
    )
    # Rehydrate the uncertainty block the report expects.
    if rec.get("uncertainty_std") is not None:
        payload["uncertainty"] = {
            "supported": True, "std": rec["uncertainty_std"],
            "passes": config.MC_DROPOUT_PASSES,
            "flagged": bool(rec.get("uncertainty_flag")),
        }
    try:
        pdf = build_referral_pdf(payload)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"report generation failed: {exc}") from exc
    db.update_prediction(prediction_id, {"report_path": str(pdf)})
    return FileResponse(pdf, media_type="application/pdf",
                        filename=f"radfrac_referral_{prediction_id}.pdf")


# --------------------------------------------------------------------------- #
# History / triage
# --------------------------------------------------------------------------- #
@app.get("/predictions")
def list_predictions(limit: int = 50, case_id: str | None = None) -> dict:
    limit = max(1, min(limit, 500))
    return {"predictions": db.list_predictions(limit=limit, case_id=case_id)}


@app.get("/triage")
def triage(limit: int = 50) -> dict:
    limit = max(1, min(limit, 500))
    return {
        "note": ("Worklist re-ordered by urgency. This re-sorts the queue; it "
                 "does not replace sequential review by a clinician."),
        "queue": db.triage_queue(limit=limit),
    }


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
@app.post("/cases", status_code=status.HTTP_201_CREATED)
def create_case(payload: CaseCreate) -> dict:
    case_id = db.create_case(payload.patient_ref, payload.body_region, payload.notes)
    return {"case_id": case_id}


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "case not found")
    case["predictions"] = db.list_predictions(case_id=case_id)
    return case

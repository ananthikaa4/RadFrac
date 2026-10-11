"""Service layer (6.1) - the orchestration boundary of the application.

Sits between any frontend (FastAPI, Streamlit, CLI) and the model modules. It
owns everything that must happen identically regardless of caller:

1. Load every available model **once** into a process-wide singleton.
2. Gate the input through the validity checks (4.8) before spending compute.
3. Run the primary model, plus the other models for cross-checking (4.7).
4. Compute Grad-CAM attention and its coherence.
5. Estimate Monte Carlo Dropout uncertainty (4.1).
6. Compose the natural-language screening note (4.9).
7. Persist the case + prediction and stored artifacts to SQLite (6.3).

Because validation and orchestration live here rather than in a frontend, any
future UI gets the same behaviour for free (6.4).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config, db
from .dataset_status import is_synthetic_data
from .disagreement import assess as assess_disagreement
from .inference import FracturePredictor, ModelUnavailableError, available_models
from .screening_note import describe_attention, generate_note
from .thresholds import get_threshold
from .uncertainty import mc_dropout_predict
from .validity import check_gradcam_coherence, check_image


class InvalidInputError(ValueError):
    """Raised when the input image fails a hard validity check."""

    def __init__(self, message: str, validity: dict):
        super().__init__(message)
        self.validity = validity


def _finite_only(d: dict) -> dict:
    """Replace non-finite floats with None.

    NaN/Inf cannot be serialised by Starlette's JSONResponse (it renders with
    ``allow_nan=False``), so a NaN anywhere in the payload turns a successful
    screening into an opaque HTTP 500. ``uncertainty`` is the one block that
    legitimately carries NaN on its unsupported path.
    """
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in d.items()}


@dataclass
class ScreeningResult:
    """Everything the application knows about one screened image."""
    prediction_id: str | None
    model_key: str
    probability: float
    threshold: float
    label: str
    positive: bool
    uncertainty: dict = field(default_factory=dict)
    disagreement: dict = field(default_factory=dict)
    per_model: dict = field(default_factory=dict)
    attention: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)
    screening_note: str = ""
    ensemble: dict = field(default_factory=dict)
    synthetic_data: bool = False
    case_id: str | None = None
    image_path: str | None = None
    gradcam_path: str | None = None
    classical_explain_path: str | None = None
    report_path: str | None = None
    recommend_manual_review: bool = False
    # Gradient-boosting models (XGBoost / LightGBM / CatBoost / committee).
    # Reported alongside the two-model ensemble; they never change its verdict
    # unless one of them is explicitly requested as the primary model.
    advanced: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


class ScreeningService:
    """Process-wide singleton holding loaded models and serving screenings."""

    _instance: "ScreeningService | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        config.ensure_output_dirs()
        config.ensure_runtime_dirs()
        db.init_db()
        self._predictors: dict[str, FracturePredictor] = {}
        self._mc_models: dict[str, Any] = {}
        self._load_lock = threading.Lock()
        self.synthetic = bool(is_synthetic_data())
        self._boost = None              # BoostingSuite, loaded in _load_all
        self._embedding_model = None    # MobileNetV2 GAP sub-model for boosters
        self._load_all()

    # -- singleton -------------------------------------------------------- #
    @classmethod
    def instance(cls) -> "ScreeningService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (used by tests after repointing config)."""
        with cls._lock:
            cls._instance = None

    # -- model loading ---------------------------------------------------- #
    def _load_all(self) -> None:
        for key in available_models():
            try:
                self._predictors[key] = FracturePredictor(key)
            except ModelUnavailableError:
                continue
        self._load_boosting()

    def _load_boosting(self) -> None:
        """Load the gradient-boosting suite if it has been trained.

        Optional by design: a missing or broken booster artifact leaves the
        original two-model system fully working.
        """
        try:
            from .model_boosting import BoostingSuite
            suite = BoostingSuite.load()
        except Exception as exc:
            print(f"[service] boosting models unavailable: {exc}")
            return
        if suite is None:
            return
        if suite.needs_deep:
            if "transfer" not in self._predictors:
                print("[service] boosters need the transfer model for deep features; skipped")
                return
            from .features_hybrid import build_embedding_model
            self._embedding_model = build_embedding_model(self._predictors["transfer"]._model)
        self._boost = suite

    @property
    def advanced_models(self) -> list[str]:
        return list(self._boost.keys) if self._boost else []

    def _screen_advanced(self, img_uint8: np.ndarray) -> dict:
        if not self._boost:
            return {}
        from .features_hybrid import hybrid_vector
        x = hybrid_vector(img_uint8, self._embedding_model,
                          use_handcrafted=self._boost.features.get("use_handcrafted", True),
                          use_deep=self._boost.needs_deep)
        return self._boost.predict_vector(x)

    @property
    def loaded_models(self) -> list[str]:
        return sorted(self._predictors)

    def _predictor(self, model_key: str) -> FracturePredictor:
        if model_key not in self._predictors:
            raise ModelUnavailableError(
                f"Model '{model_key}' is not loaded. Available: "
                f"{self.loaded_models or 'none - train a model first'}"
            )
        return self._predictors[model_key]

    def _mc_model_for(self, model_key: str):
        """Lazily build and cache the MC-dropout clone for a deep model."""
        if model_key in self._mc_models:
            return self._mc_models[model_key]
        if model_key not in ("cnn", "transfer"):
            self._mc_models[model_key] = None
            return None
        from .uncertainty import build_mc_model, model_has_dropout

        base = self._predictor(model_key)._model
        with self._load_lock:
            mc = build_mc_model(base) if model_has_dropout(base) else None
            self._mc_models[model_key] = mc
        return mc

    @property
    def ensemble_config(self):
        """The frozen two-model ensemble configuration, if it has been fitted."""
        from .ensemble import EnsembleConfig
        if getattr(self, "_ens_cfg", None) is None:
            self._ens_cfg = EnsembleConfig.load()
        return self._ens_cfg

    def ensemble_available(self) -> bool:
        cfg = self.ensemble_config
        return bool(cfg) and all(m in self._predictors for m in cfg.members)

    def default_model(self) -> str:
        """The ensemble when it is usable, else the best available single model."""
        if config.DEFAULT_SERVICE_MODEL == config.ENSEMBLE_KEY and self.ensemble_available():
            return config.ENSEMBLE_KEY
        if config.DEFAULT_SERVICE_MODEL in self._predictors:
            return config.DEFAULT_SERVICE_MODEL
        if self.ensemble_available():
            return config.ENSEMBLE_KEY
        from .inference import load_comparison_metrics
        metrics = load_comparison_metrics()
        if metrics and metrics.get("best_model") in self._predictors:
            return metrics["best_model"]
        if not self._predictors:
            raise ModelUnavailableError(
                "No trained models available. Run 'python -m src.train --model all'."
            )
        return self.loaded_models[0]

    # -- core screening --------------------------------------------------- #
    def screen(
        self,
        img_uint8: np.ndarray,
        model_key: str | None = None,
        *,
        cross_check: bool = True,
        with_uncertainty: bool = True,
        with_gradcam: bool = True,
        persist: bool = True,
        case_id: str | None = None,
        original_filename: str | None = None,
        store_image: bool = True,
    ) -> ScreeningResult:
        """Screen one image end-to-end."""
        model_key = model_key or self.default_model()
        is_advanced = model_key in self.advanced_models
        # "ensemble" and the boosters have no FracturePredictor of their own.
        predictor = (None if (model_key == config.ENSEMBLE_KEY or is_advanced)
                     else self._predictor(model_key))

        # 1. Validity gate BEFORE inference (4.8 / 6.4).
        validity = check_image(img_uint8)
        if validity.should_block:
            raise InvalidInputError(validity.summary(), validity.to_dict())

        # 2. Run BOTH active members, then either report one of them or the
        #    frozen ensemble blend. Members always run so the UI can show the
        #    two-model breakdown and the agreement signal.
        from .ensemble import MEMBERS

        member_raw: dict[str, float] = {}
        for mk in MEMBERS:
            if mk in self._predictors:
                try:
                    member_raw[mk] = self._predictors[mk].predict_array(img_uint8).probability
                except Exception as exc:
                    # Log it: swallowing this silently made a broken model look
                    # like a missing ensemble, which is undiagnosable in the field.
                    print(f"[service] member '{mk}' inference failed: {exc!r}")
                    continue

        # Gradient-boosting models always run when available (reported in their
        # own panel); failures degrade to an empty block, never a failed screen.
        advanced: dict = {}
        if self._boost:
            try:
                advanced = self._screen_advanced(img_uint8)
            except Exception as exc:
                if is_advanced:
                    raise ModelUnavailableError(f"boosting model failed: {exc}") from exc
                advanced = {}

        is_ensemble = (model_key == config.ENSEMBLE_KEY)
        ens_cfg = self.ensemble_config if is_ensemble else None
        if is_advanced:
            prob = float(advanced[model_key]["probability"])
            threshold = float(advanced[model_key]["threshold"])
            gradcam_source = "transfer" if "transfer" in self._predictors else None
        elif is_ensemble:
            if not ens_cfg or not member_raw:
                raise ModelUnavailableError(
                    "Ensemble is not available. Run 'python -m scripts.fit_ensemble'."
                )
            prob = float(ens_cfg.combine(member_raw)[0])
            threshold = float(ens_cfg.threshold)
            gradcam_source = "transfer" if "transfer" in self._predictors else None
        else:
            prob = (member_raw.get(model_key)
                    if model_key in member_raw
                    else predictor.predict_array(img_uint8).probability)
            prob = float(prob)
            threshold = get_threshold(model_key)
            gradcam_source = model_key if model_key == "transfer" else None

        # A non-finite probability must never reach the response or the DB.
        # EnsembleConfig.combine returns NaN when no member weight applies, and
        # NaN would (a) compare False against the threshold and be reported as
        # a silent negative, (b) violate predictions.probability REAL NOT NULL,
        # and (c) fail JSON serialisation, since Starlette renders with
        # allow_nan=False. All three surface as an opaque HTTP 500.
        if not math.isfinite(prob):
            raise ModelUnavailableError(
                f"Model '{model_key}' produced a non-finite probability "
                f"({prob}). The ensemble weights or a member model are "
                f"misconfigured; re-run 'python -m scripts.fit_ensemble'."
            )

        positive = prob >= threshold
        neg_name = config.CLASS_NAMES[1 - config.POSITIVE_INDEX]

        class _P:                      # lightweight stand-in for Prediction
            probability = prob
            label = config.POSITIVE_CLASS if positive else neg_name
        primary = _P()

        # 3. Crop+resize once to model input for Grad-CAM / MC dropout, using
        #    the shared helper so this matches training exactly.
        from .preprocessing import prepare_model_input, preprocess_for_dl

        small = prepare_model_input(img_uint8)
        img01 = preprocess_for_dl(small)

        # 4. Grad-CAM attention + coherence. Only the MobileNetV2 member is a
        #    gradient-explainable model, so the heatmap always comes from it and
        #    is labelled as such - it does NOT explain the SVM.
        heatmap = None
        attention: dict = {"available": False}
        if with_gradcam and gradcam_source:
            try:
                from .gradcam import compute_gradcam
                heatmap = compute_gradcam(self._predictors[gradcam_source]._model, img01)
                attention = describe_attention(heatmap)
                coherence = check_gradcam_coherence(heatmap)
                validity.warnings.extend(coherence.warnings)
                validity.metrics.update(coherence.metrics)
            except Exception as exc:  # explainability must never break screening
                attention = {"available": False, "error": str(exc)}

        # 5. MC Dropout uncertainty - again only meaningful for the neural member.
        uncertainty: dict = {}
        if with_uncertainty and gradcam_source:
            try:
                mc = self._mc_model_for(gradcam_source)
                unc = mc_dropout_predict(self._predictors[gradcam_source]._model,
                                         img01, mc_model=mc)
                uncertainty = _finite_only(unc.to_dict())
            except Exception as exc:
                uncertainty = {"supported": False, "reason": f"MC dropout failed: {exc}"}
        else:
            uncertainty = {
                "supported": False,
                "reason": "uncertainty sampling applies to the neural models only",
            }

        # 6. Two-model agreement. Each member is judged at its OWN validated
        #    threshold, so "agree" means both independently reach the same verdict.
        from .ensemble import MEMBER_LABELS

        per_model: dict[str, dict] = {}
        for mk, praw in member_raw.items():
            mthr = get_threshold(mk)
            per_model[mk] = {
                "probability": float(praw),
                "label": config.POSITIVE_CLASS if praw >= mthr else neg_name,
                "threshold": float(mthr),
                "display_name": MEMBER_LABELS.get(mk, mk),
            }
        dis = assess_disagreement(per_model)

        # 7. Screening note (4.9).
        note = generate_note(
            label=primary.label, probability=primary.probability,
            threshold=threshold, model_key=model_key,
            uncertainty=uncertainty, disagreement=dis.to_dict(),
            attention=attention, validity=validity.to_dict(),
            synthetic=self.synthetic,
        )

        manual_review = bool(
            dis.flagged or uncertainty.get("flagged") or validity.warnings
        )

        result = ScreeningResult(
            prediction_id=None, model_key=model_key,
            probability=primary.probability, threshold=threshold,
            label=primary.label, positive=positive,
            uncertainty=uncertainty, disagreement=dis.to_dict(),
            per_model=per_model, attention=attention,
            validity=validity.to_dict(), screening_note=note,
            synthetic_data=self.synthetic, case_id=case_id,
            recommend_manual_review=manual_review,
        )
        result.ensemble = {
            "used": is_ensemble,
            "method": (ens_cfg.method if ens_cfg else None),
            "weights": (ens_cfg.weights if ens_cfg else None),
            "threshold": threshold,
            "probability": primary.probability if is_ensemble else None,
            "members": list(member_raw),
            "gradcam_source": gradcam_source,
        }
        if advanced:
            boost_only = {k: v for k, v in advanced.items() if k != "boost_committee"}
            result.advanced = {
                "models": advanced,
                "agreement": assess_disagreement(boost_only).to_dict(),
                "agrees_with_primary": all(
                    v["label"] == result.label for v in boost_only.values()),
            }

        # 8. Persist image, heatmap and record (6.3).
        if persist:
            self._persist(result, img_uint8, heatmap, original_filename, store_image)
        return result

    def _persist(self, result: ScreeningResult, img_uint8: np.ndarray,
                 heatmap: np.ndarray | None, original_filename: str | None,
                 store_image: bool) -> None:
        from PIL import Image

        pred_id = db.new_id()
        case_dir = config.CASE_STORE_DIR / pred_id
        try:
            case_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Read-only filesystem or a full disk must not discard a screening
            # that already succeeded. Return the result unpersisted.
            print(f"[service] could not create case dir {case_dir}: {exc!r}")
            return

        if store_image:
            try:
                img_path = case_dir / "input.png"
                Image.fromarray(img_uint8).convert("RGB").save(img_path)
                result.image_path = str(img_path)
            except OSError as exc:
                print(f"[service] could not store input image: {exc!r}")

        if heatmap is not None:
            try:
                from .gradcam import overlay_heatmap
                from .preprocessing import prepare_model_input, preprocess_for_dl

                small = prepare_model_input(img_uint8)
                overlay = overlay_heatmap(preprocess_for_dl(small), heatmap)
                gc_path = case_dir / "gradcam.png"
                Image.fromarray(overlay).save(gc_path)
                result.gradcam_path = str(gc_path)
            except Exception as exc:
                print(f"[service] gradcam overlay not stored: {exc!r}")

        # Classical member's own feature visualisation (never implied to explain
        # the neural model, and vice versa).
        if "classical" in self._predictors:
            try:
                from .explain_classical import render_classical_explanation
                from .preprocessing import prepare_model_input

                p = render_classical_explanation(prepare_model_input(img_uint8),
                                                 case_dir / "classical_features.png")
                result.classical_explain_path = str(p)
            except Exception as exc:
                print(f"[service] classical explanation not stored: {exc!r}")

        record = {
            "id": pred_id,
            "case_id": result.case_id,
            "image_path": result.image_path,
            "original_filename": original_filename,
            "model_key": result.model_key,
            "probability": result.probability,
            "threshold": result.threshold,
            "label": result.label,
            "uncertainty_std": result.uncertainty.get("std"),
            "uncertainty_flag": result.uncertainty.get("flagged", False),
            "disagreement_flag": result.disagreement.get("flagged", False),
            "per_model": result.per_model,
            "validity_ok": result.validity.get("ok", True),
            "validity": result.validity,
            "screening_note": result.screening_note,
            "gradcam_path": result.gradcam_path,
            "synthetic_data": result.synthetic_data,
        }
        try:
            db.insert_prediction(record)
        except Exception as exc:
            # A DB failure (locked, read-only, FK violation) must not turn a
            # completed screening into a 500. The result is still returned;
            # it just has no prediction_id, so history/report links are absent.
            print(f"[service] prediction not persisted: {exc!r}")
            return
        result.prediction_id = pred_id

    # -- convenience ------------------------------------------------------ #
    def screen_file(self, path: str | Path, **kwargs) -> ScreeningResult:
        from .preprocessing import load_image_array
        from PIL import Image

        # Load at native size so validity checks see the real image.
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype="uint8")
        kwargs.setdefault("original_filename", Path(path).name)
        return self.screen(arr, **kwargs)

    def model_catalogue(self) -> dict:
        """ACTIVE models plus the ensemble, with their frozen test-set metrics.

        Only the two active members are listed; the archived CNN never appears.
        """
        import json

        from .ensemble import MEMBER_LABELS

        ens_path = config.METRICS_DIR / "ensemble_results.json"
        ens_res = json.loads(ens_path.read_text()) if ens_path.exists() else {}
        test_block = ens_res.get("test", {})
        member_metrics = test_block.get("members", {})

        out = []
        for key in self.loaded_models:
            out.append({
                "key": key,
                "name": MEMBER_LABELS.get(key, key),
                "threshold": get_threshold(key),
                "supports_gradcam": key == "transfer",
                "supports_uncertainty": key == "transfer",
                "explainability": ("LayerCAM / Grad-CAM" if key == "transfer"
                                   else "HOG + Canny feature maps"),
                "test_metrics": member_metrics.get(key),
            })

        cfg = self.ensemble_config
        ensemble_entry = None
        if self.ensemble_available() and cfg:
            ensemble_entry = {
                "key": config.ENSEMBLE_KEY,
                "name": "RADFRAC Ensemble",
                "method": cfg.method,
                "weights": cfg.weights,
                "threshold": cfg.threshold,
                "members": [MEMBER_LABELS.get(m, m) for m in cfg.members],
                "test_metrics": test_block.get("ensemble"),
            }

        advanced = []
        if self._boost:
            from .model_boosting import DESCRIPTIONS, DISPLAY_NAMES, load_registry
            reg = load_registry().get("models", {})
            for key in self._boost.keys:
                entry = reg.get(key, {})
                advanced.append({
                    "key": key,
                    "name": DISPLAY_NAMES.get(key, key),
                    "description": DESCRIPTIONS.get(key, ""),
                    "threshold": self._boost.thresholds.get(key),
                    "pca_components": entry.get("pca_components"),
                    "n_estimators": entry.get("n_estimators"),
                    "best_params": entry.get("best_params"),
                    "explainability": "SHAP (TreeExplainer) on PCA components",
                    "test_metrics": self._boost.test_metrics.get(key),
                })

        return {
            "active_models": out,
            "advanced_models": advanced,
            "ensemble": ensemble_entry,
            "default_model": self.default_model(),
            "archived_models": list(config.ARCHIVED_MODEL_KEYS),
            "verdict": ens_res.get("verdict"),
            "synthetic_data": self.synthetic,
        }

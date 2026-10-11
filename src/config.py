"""Central configuration for RadFrac.

All paths and hyperparameters live here so training, evaluation, and the demo
app stay consistent. Paths are resolved relative to the project root (the
directory that contains this ``src`` package), so the code runs the same way
regardless of the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# PROJECT_ROOT = .../radfrac/radfrac  (the folder that contains this src/ package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow overriding the data root via env var (useful for deployment / CI).
DATA_ROOT = Path(os.environ.get("RADFRAC_DATA_ROOT", PROJECT_ROOT.parent / "data"))
RAW_DIR = DATA_ROOT / "raw"                 # data/raw/<class>/*.jpg
# Split manifests. Overridable so a leakage-safe group-aware split can be used
# without touching the original: RADFRAC_SPLIT_DIR=../data/splits_grouped
SPLIT_DIR = Path(os.environ.get("RADFRAC_SPLIT_DIR", DATA_ROOT / "splits"))

# Marker file written by scripts/make_synthetic_data.py so every downstream
# component (evaluate, app, report) can hard-detect synthetic data and refuse
# to present it as clinically meaningful.
SYNTHETIC_MARKER_FILE = RAW_DIR / ".SYNTHETIC_DATA"
# Filename prefix the synthetic generator uses (fallback detection when the
# marker file is missing). Deliberately a distinctive prefix that real ingested
# images never use: an earlier version keyed on "<class>_" which collided with
# the naming used for real ingested files and produced false synthetic warnings
# on genuine data.
SYNTHETIC_FILENAME_PREFIX = "synthetic_"
SYNTHETIC_FILENAME_PREFIXES = (SYNTHETIC_FILENAME_PREFIX,)
# Neutral prefix used for real images normalised into data/raw by the ingest
# scripts. Must never start with SYNTHETIC_FILENAME_PREFIX.
REAL_IMAGE_PREFIX = "xr_"

OUTPUT_DIR = Path(os.environ.get("RADFRAC_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# --------------------------------------------------------------------------- #
# Classes / labels
# --------------------------------------------------------------------------- #
# Directory-per-class layout. Index in this list IS the integer label.
CLASS_NAMES = ["not_fractured", "fractured"]   # 0 = negative, 1 = positive
POSITIVE_CLASS = "fractured"
POSITIVE_INDEX = CLASS_NAMES.index(POSITIVE_CLASS)

# --------------------------------------------------------------------------- #
# Image / preprocessing
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (128, 128)      # (H, W) fed to the CNN / transfer models
IMAGE_CHANNELS = 3
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# --------------------------------------------------------------------------- #
# Anatomy-focused cropping (shortcut mitigation)
# --------------------------------------------------------------------------- #
# Measured problem: in the Kaggle multi-region dataset, fractured and
# not-fractured films differ systematically in their PERIPHERY - black rotation
# corners, padding/border width, and burned-in radiographic markers such as
# "R09". A CNN reached AUC 0.951 from the peripheral ring ALONE with the central
# anatomy blanked out, i.e. it was reading acquisition artefacts rather than
# bone. Grad-CAM faithfully showed heat on borders and on marker text.
#
# Cropping to the central region before resizing removes most of that leakage.
# Applied identically in the tf.data pipeline and the PIL loader so training and
# inference never diverge. Set to 1.0 to disable.
try:
    CENTER_CROP_FRACTION = float(os.environ.get("RADFRAC_CENTER_CROP", "0.72"))
except ValueError:
    CENTER_CROP_FRACTION = 0.72

# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20
VAL_SIZE = 0.20              # fraction of the *remaining* train after test split
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Deep-learning training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 16
EPOCHS_CNN = 40
EPOCHS_TRANSFER = 12
LR_CNN = 1e-3
LR_TRANSFER = 1e-4
EARLY_STOPPING_PATIENCE = 8

# --------------------------------------------------------------------------- #
# Classical baseline
# --------------------------------------------------------------------------- #
CLASSICAL_MODEL_TYPE = "svm"    # "svm" or "rf"
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
CANNY_SIGMA = 1.0

# --------------------------------------------------------------------------- #
# Model artifact filenames
# --------------------------------------------------------------------------- #
CNN_MODEL_FILE = MODELS_DIR / "cnn.keras"
TRANSFER_MODEL_FILE = MODELS_DIR / "transfer.keras"
CLASSICAL_MODEL_FILE = MODELS_DIR / "classical.pkl"
COMPARISON_METRICS_FILE = METRICS_DIR / "comparison_metrics.json"
# Decision thresholds tuned on the validation split (per model key).
THRESHOLDS_FILE = MODELS_DIR / "thresholds.json"

# Every model key the repository knows about, including archived experiments.
MODEL_KEYS = ("cnn", "transfer", "classical")

# --------------------------------------------------------------------------- #
# ACTIVE application models
# --------------------------------------------------------------------------- #
# The custom CNN is retained for training scripts, history and the report, but it
# is NOT part of the running application: on the leakage-free split it reached
# only ROC-AUC 0.711 versus 0.826 (SVM) and 0.813 (MobileNetV2), and it
# duplicates the deep-feature role the transfer model already fills. The live
# system runs the two complementary approaches - hand-crafted features + SVM and
# a pretrained CNN backbone - and blends them.
ACTIVE_MODEL_KEYS = ("classical", "transfer")
ARCHIVED_MODEL_KEYS = ("cnn",)
ENSEMBLE_KEY = "ensemble"
ENSEMBLE_CONFIG_FILE = MODELS_DIR / "ensemble.json"

DEFAULT_THRESHOLD = 0.5

# Self-supervised / distilled artifacts (optional, produced by advanced scripts)
SSL_ENCODER_FILE = MODELS_DIR / "ssl_encoder.keras"
SSL_FINETUNED_FILE = MODELS_DIR / "ssl_finetuned.keras"
STUDENT_MODEL_FILE = MODELS_DIR / "student.keras"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Deployment / application configuration (6.5)
# All env-overridable so the same code runs on a laptop, a server, or a rural
# clinic machine without edits.
# --------------------------------------------------------------------------- #
# SQLite database for prediction history / referral records.
DB_PATH = Path(os.environ.get("RADFRAC_DB_PATH", OUTPUT_DIR / "radfrac.db"))
# Where uploaded images and generated artifacts are stored per case.
CASE_STORE_DIR = Path(os.environ.get("RADFRAC_CASE_STORE", OUTPUT_DIR / "cases"))

# Monte Carlo Dropout (4.1)
MC_DROPOUT_PASSES = _env_int("RADFRAC_MC_PASSES", 40)
# Std-dev of MC predictions above which we call the model genuinely uncertain.
UNCERTAINTY_STD_THRESHOLD = _env_float("RADFRAC_UNCERTAINTY_STD", 0.15)

# Clinical cost weighting (section 3): a missed fracture is far costlier than a
# false alarm at a screening/triage step.
FALSE_NEGATIVE_COST = _env_float("RADFRAC_FN_COST", 5.0)
FALSE_POSITIVE_COST = _env_float("RADFRAC_FP_COST", 1.0)

# Hard constraint on the cost-optimal operating point.
#
# Unconstrained weighted-cost minimisation DEGENERATES on a weak classifier: if
# false negatives are weighted 5x and the classes are roughly balanced, then
# "flag every image" costs ~N_neg while "flag nothing" costs ~5*N_pos, so the
# optimiser drives the threshold to zero and the tool reports every patient as
# fractured. That is mathematically optimal for the stated cost function and
# clinically worthless.
#
# A screening aid must retain usable specificity, so the cost search is
# restricted to thresholds whose false-positive rate stays at or below this
# value (i.e. specificity >= 1 - MAX_FPR).
MAX_FALSE_POSITIVE_RATE = _env_float("RADFRAC_MAX_FPR", 0.25)

# Model disagreement (4.7): flag when model probabilities span more than this.
DISAGREEMENT_SPREAD = _env_float("RADFRAC_DISAGREEMENT_SPREAD", 0.30)

# Input validity gate (4.8)
MIN_MEAN_INTENSITY = _env_float("RADFRAC_MIN_MEAN_INTENSITY", 0.06)
MAX_MEAN_INTENSITY = _env_float("RADFRAC_MAX_MEAN_INTENSITY", 0.96)
MAX_ASPECT_RATIO = _env_float("RADFRAC_MAX_ASPECT_RATIO", 3.0)
MAX_COLOR_SATURATION = _env_float("RADFRAC_MAX_COLOR_SATURATION", 0.18)
MIN_INTENSITY_STD = _env_float("RADFRAC_MIN_INTENSITY_STD", 0.035)

# Which model the service layer uses as primary when none is specified.
DEFAULT_SERVICE_MODEL = os.environ.get("RADFRAC_DEFAULT_MODEL", ENSEMBLE_KEY)

# Cost-optimal thresholds (written by scripts/threshold_analysis.py)
COST_THRESHOLDS_FILE = MODELS_DIR / "cost_thresholds.json"
CALIBRATION_FILE = METRICS_DIR / "calibration.json"
THRESHOLD_ANALYSIS_FILE = METRICS_DIR / "threshold_analysis.json"


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR, SPLIT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_runtime_dirs() -> None:
    """Create directories the running application needs (DB + case store)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASE_STORE_DIR.mkdir(parents=True, exist_ok=True)

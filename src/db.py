"""SQLite persistence for RadFrac predictions and cases (6.3).

Schema is intentionally designed up front for the eventual UI so it does not
need reshaping later:

``cases``
    One row per patient encounter. A case can hold several X-rays (multiple
    views / regions), which is why predictions reference a case rather than
    standing alone.

``predictions``
    One row per screened image: which model ran, the probability, the threshold
    actually applied, the MC-dropout uncertainty, the model-disagreement flag,
    the input-validity verdict, and paths to stored artifacts (image, Grad-CAM,
    PDF report). Per-model probabilities from the disagreement check are stored
    as JSON so the full three-model picture is recoverable for audit.

Everything is plain ``sqlite3`` from the stdlib - no ORM - to keep the runtime
dependency footprint small enough for a low-resource clinic machine.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id            TEXT PRIMARY KEY,
    patient_ref   TEXT,
    body_region   TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                 TEXT PRIMARY KEY,
    case_id            TEXT REFERENCES cases(id) ON DELETE SET NULL,
    image_path         TEXT,
    original_filename  TEXT,
    model_key          TEXT NOT NULL,
    probability        REAL NOT NULL,
    threshold          REAL NOT NULL,
    label              TEXT NOT NULL,
    uncertainty_std    REAL,
    uncertainty_flag   INTEGER NOT NULL DEFAULT 0,
    disagreement_flag  INTEGER NOT NULL DEFAULT 0,
    per_model_json     TEXT,
    validity_ok        INTEGER NOT NULL DEFAULT 1,
    validity_json      TEXT,
    screening_note     TEXT,
    gradcam_path       TEXT,
    report_path        TEXT,
    synthetic_data     INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_case    ON predictions(case_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection with row access by name and FK enforcement on."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait for a competing writer instead of raising "database is
    # locked" immediately. The container runs the API and the Streamlit UI as
    # two processes against this one file.
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets readers proceed while a write is in flight. Skipped silently if
    # the filesystem does not support it (e.g. some network mounts).
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """Create tables if absent. Safe to call on every application start."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def create_case(patient_ref: str | None = None, body_region: str | None = None,
                notes: str | None = None, db_path: Path | str | None = None) -> str:
    case_id = new_id()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cases (id, patient_ref, body_region, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, patient_ref, body_region, notes, _now()),
        )
    return case_id


def get_case(case_id: str, db_path: Path | str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def insert_prediction(record: dict[str, Any], db_path: Path | str | None = None) -> str:
    """Insert a prediction record. Returns the new prediction id."""
    pred_id = record.get("id") or new_id()
    payload = {
        "id": pred_id,
        "case_id": record.get("case_id"),
        "image_path": record.get("image_path"),
        "original_filename": record.get("original_filename"),
        "model_key": record["model_key"],
        "probability": float(record["probability"]),
        "threshold": float(record["threshold"]),
        "label": record["label"],
        "uncertainty_std": record.get("uncertainty_std"),
        "uncertainty_flag": int(bool(record.get("uncertainty_flag"))),
        "disagreement_flag": int(bool(record.get("disagreement_flag"))),
        "per_model_json": json.dumps(record.get("per_model") or {}),
        "validity_ok": int(bool(record.get("validity_ok", True))),
        "validity_json": json.dumps(record.get("validity") or {}),
        "screening_note": record.get("screening_note"),
        "gradcam_path": record.get("gradcam_path"),
        "report_path": record.get("report_path"),
        "synthetic_data": int(bool(record.get("synthetic_data"))),
        "created_at": record.get("created_at") or _now(),
    }
    cols = ", ".join(payload)
    marks = ", ".join("?" for _ in payload)
    with connect(db_path) as conn:
        conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({marks})",
                     tuple(payload.values()))
    return pred_id


def get_prediction(pred_id: str, db_path: Path | str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if not row:
        return None
    rec = dict(row)
    rec["per_model"] = json.loads(rec.pop("per_model_json") or "{}")
    rec["validity"] = json.loads(rec.pop("validity_json") or "{}")
    for flag in ("uncertainty_flag", "disagreement_flag", "validity_ok", "synthetic_data"):
        rec[flag] = bool(rec[flag])
    return rec


def update_prediction(pred_id: str, fields: dict[str, Any],
                      db_path: Path | str | None = None) -> None:
    """Patch specific columns (used to attach a report/gradcam path later)."""
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE predictions SET {assignments} WHERE id = ?",
                     (*fields.values(), pred_id))


def list_predictions(limit: int = 50, case_id: str | None = None,
                     db_path: Path | str | None = None) -> list[dict]:
    query = "SELECT * FROM predictions"
    params: tuple = ()
    if case_id:
        query += " WHERE case_id = ?"
        params = (case_id,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params = (*params, limit)
    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for row in rows:
        rec = dict(row)
        rec["per_model"] = json.loads(rec.pop("per_model_json") or "{}")
        rec["validity"] = json.loads(rec.pop("validity_json") or "{}")
        for flag in ("uncertainty_flag", "disagreement_flag", "validity_ok",
                     "synthetic_data"):
            rec[flag] = bool(rec[flag])
        out.append(rec)
    return out


def triage_queue(limit: int = 50, db_path: Path | str | None = None) -> list[dict]:
    """Predictions ordered by clinical urgency, for the ER worklist use case.

    Ordering rationale: positive screens first, then highest probability, with
    uncertain / disagreeing cases boosted because those most need a human look.
    This re-sorts an existing worklist; it does not replace sequential review.
    """
    rows = list_predictions(limit=limit, db_path=db_path)
    def urgency(r: dict) -> tuple:
        positive = r["label"] == config.POSITIVE_CLASS
        needs_review = r["uncertainty_flag"] or r["disagreement_flag"]
        # A NaN probability is stored by SQLite as NULL; negating that would
        # raise TypeError and take the whole worklist down.
        prob = r.get("probability")
        prob = prob if isinstance(prob, (int, float)) else 0.0
        return (not positive, not needs_review, -prob)
    return sorted(rows, key=urgency)

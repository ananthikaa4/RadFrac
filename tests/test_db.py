"""Tests for the SQLite persistence layer (6.3)."""
from __future__ import annotations

import pytest

from src import config, db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    db.init_db(path)
    return path


def _record(**over):
    rec = {
        "model_key": "cnn",
        "probability": 0.81,
        "threshold": 0.4,
        "label": "fractured",
        "uncertainty_std": 0.06,
        "uncertainty_flag": False,
        "disagreement_flag": False,
        "per_model": {"cnn": {"probability": 0.81, "label": "fractured"}},
        "validity_ok": True,
        "validity": {"ok": True, "warnings": []},
        "screening_note": "note text",
        "synthetic_data": True,
    }
    rec.update(over)
    return rec


def test_init_is_idempotent(temp_db):
    db.init_db(temp_db)
    db.init_db(temp_db)   # must not raise


def test_insert_and_get_roundtrip(temp_db):
    pid = db.insert_prediction(_record(), db_path=temp_db)
    got = db.get_prediction(pid, db_path=temp_db)
    assert got is not None
    assert got["model_key"] == "cnn"
    assert got["label"] == "fractured"
    assert got["per_model"]["cnn"]["probability"] == 0.81
    assert got["validity"]["ok"] is True
    assert got["synthetic_data"] is True
    assert isinstance(got["disagreement_flag"], bool)


def test_get_missing_returns_none(temp_db):
    assert db.get_prediction("nope", db_path=temp_db) is None


def test_update_prediction(temp_db):
    pid = db.insert_prediction(_record(), db_path=temp_db)
    db.update_prediction(pid, {"report_path": "/tmp/r.pdf"}, db_path=temp_db)
    assert db.get_prediction(pid, db_path=temp_db)["report_path"] == "/tmp/r.pdf"


def test_case_links_predictions(temp_db):
    case_id = db.create_case("PHC-7", "wrist", db_path=temp_db)
    assert db.get_case(case_id, db_path=temp_db)["patient_ref"] == "PHC-7"
    db.insert_prediction(_record(case_id=case_id), db_path=temp_db)
    db.insert_prediction(_record(case_id=case_id), db_path=temp_db)
    db.insert_prediction(_record(), db_path=temp_db)          # unlinked
    linked = db.list_predictions(case_id=case_id, db_path=temp_db)
    assert len(linked) == 2
    assert len(db.list_predictions(db_path=temp_db)) == 3


def test_triage_queue_prioritises_positive_and_review(temp_db):
    # Negative, low urgency.
    db.insert_prediction(_record(label="not_fractured", probability=0.05),
                         db_path=temp_db)
    # Positive but clean.
    db.insert_prediction(_record(label="fractured", probability=0.70),
                         db_path=temp_db)
    # Positive AND needs review -> should rank first.
    db.insert_prediction(_record(label="fractured", probability=0.60,
                                 disagreement_flag=True), db_path=temp_db)

    queue = db.triage_queue(db_path=temp_db)
    assert queue[0]["disagreement_flag"] is True         # needs-review first
    assert queue[-1]["label"] == "not_fractured"         # negatives last

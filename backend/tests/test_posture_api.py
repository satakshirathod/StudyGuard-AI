"""Minimal API-level smoke tests for the Phase 6 posture integration.

Verifies the FastAPI wiring (root phase/modules + `/api/video/status` posture
block) through an in-process TestClient. Environment for a temp SQLite database
is set *before* importing ``main`` so settings are resolved correctly.

Updated in Phase 7: root phase/modules now reflect the phone-detection stage.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.environ["SG_DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app)


def test_root_reports_phone_phase():
    with client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "phone-detection"
        assert "posture" in body["modules"]
        assert "phone" in body["modules"]
        assert body["version"].startswith("0.8")


def test_health_ok():
    with client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_status_includes_posture_block():
    with client:
        resp = client.get("/api/video/status")
        assert resp.status_code == 200
        body = resp.json()
        posture = body.get("posture")
        assert posture is not None
        # Shape matches the frontend contract.
        for key in ("state", "posture_score", "head_pitch_score",
                    "face_offset_score", "confidence", "error", "posture_enabled"):
            assert key in posture
        assert posture["state"] in ("GOOD", "MODERATE", "POOR", "UNKNOWN")
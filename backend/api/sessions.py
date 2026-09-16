"""Study-session endpoints (start / end / list / live status)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

import runtime
from api.deps import get_db
from database.models import AlertRecord, StudySession
from services import student_service
from services.session_service import SessionError

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class StartSessionRequest(BaseModel):
    student_id: int


class EndSessionRequest(BaseModel):
    session_id: int | None = None  # defaults to the active session


@router.get("/active")
def active_session(db: Session = Depends(get_db)) -> dict | None:
    session = runtime.session_service.get_active_session(db)
    return session.to_dict() if session else None


@router.get("")
def list_sessions(
    student_id: int | None = None, limit: int = 50, db: Session = Depends(get_db)
) -> list[dict]:
    return [s.to_dict() for s in runtime.session_service.list_sessions(db, student_id, limit)]


@router.get("/live")
def live_status(db: Session = Depends(get_db)) -> dict:
    """Aggregated status for the Live Monitoring page (session + behavior + alerts)."""
    session = runtime.session_service.get_active_session(db)
    alert_rows: list[AlertRecord] = []
    if session is not None:
        alert_rows = runtime.alert_service.latest_for_session(db, session.id, limit=20)
    return {
        "session": session.to_dict() if session else None,
        "behavior": runtime.latest_behavior.to_dict() if runtime.latest_behavior else None,
        "alerts": [a.to_dict() for a in alert_rows],
        "focus_score": runtime.latest_behavior.focus_score if runtime.latest_behavior else None,
        "focus_level": runtime.latest_behavior.focus_level if runtime.latest_behavior else None,
    }


@router.post("/start", status_code=201)
def start_session(body: StartSessionRequest, db: Session = Depends(get_db)) -> dict:
    if student_service.get_student(db, body.student_id) is None:
        raise HTTPException(status_code=404, detail="Student not found")
    try:
        session, created = runtime.session_service.start_session(db, body.student_id)
    except SessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    runtime.reset_alert_state_for_new_session()
    return {"session": session.to_dict(), "created": created}


@router.post("/{session_id}/end")
def end_session(session_id: int, db: Session = Depends(get_db)) -> dict:
    try:
        session = runtime.session_service.end_session(db, session_id)
    except SessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return session.to_dict()


@router.get("/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)) -> dict:
    session: StudySession | None = runtime.session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()
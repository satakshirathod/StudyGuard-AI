"""Analytics endpoints (per-session drill-down + weekly trends)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import runtime
from api.deps import get_db
from services import student_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/sessions/{session_id}")
def session_stats(session_id: int, db: Session = Depends(get_db)) -> dict:
    stats = runtime.analytics_service.session_stats(db, session_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats


@router.get("/weekly")
def weekly(days: int = 7, db: Session = Depends(get_db)) -> dict:
    student = student_service.get_active_student(db)
    if student is None:
        raise HTTPException(status_code=400, detail="No active student — create a profile first")
    return runtime.analytics_service.weekly(db, student.id, days=min(max(days, 1), 30))
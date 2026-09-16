"""Dashboard endpoints (today's overview + weekly trend)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import runtime
from api.deps import get_db
from services import student_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _student_id_required(db: Session) -> int:
    student = student_service.get_active_student(db)
    if student is None:
        raise HTTPException(status_code=400, detail="No active student — create a profile first")
    return student.id


@router.get("/overview")
def overview(db: Session = Depends(get_db)) -> dict:
    student_id = _student_id_required(db)
    return runtime.dashboard_service.today_overview(db, student_id)


@router.get("/trend")
def trend(days: int = 7, db: Session = Depends(get_db)) -> dict:
    student_id = _student_id_required(db)
    return runtime.dashboard_service.trend(db, student_id, days=min(max(days, 1), 30))


@router.get("/recent")
def recent(limit: int = 5, db: Session = Depends(get_db)) -> list[dict]:
    student_id = _student_id_required(db)
    return runtime.dashboard_service.recent_sessions(db, student_id, limit=limit)
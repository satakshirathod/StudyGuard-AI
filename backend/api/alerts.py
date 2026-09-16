"""Alert endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import runtime
from api.deps import get_db

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("")
def list_alerts(
    session_id: int | None = None,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[dict]:
    return [a.to_dict() for a in runtime.alert_service.list_alerts(db, session_id, limit=limit)]


@router.get("/types")
def alert_types() -> dict:
    """Metadata: which alert types are live vs gated on not-yet-built modules."""
    from ai.behavior_engine import pending_alert_types

    return {"live": ["STUDENT_AWAY", "LOOKING_AWAY"], "pending": pending_alert_types()}
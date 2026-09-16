"""Alert querying (persistence lives in SessionService.add_alerts)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import AlertRecord


class AlertService:
    def list_alerts(
        self,
        db: Session,
        session_id: int | None = None,
        student_id: int | None = None,
        limit: int = 20,
    ) -> list[AlertRecord]:
        stmt = select(AlertRecord).order_by(AlertRecord.timestamp.desc()).limit(limit)
        if session_id is not None:
            stmt = stmt.where(AlertRecord.session_id == session_id)
        elif student_id is not None:
            stmt = stmt.join(AlertRecord.session).where(
                AlertRecord.session.has(student_id=student_id)
            )
        return list(db.scalars(stmt))

    def latest_for_session(self, db: Session, session_id: int, limit: int = 20) -> list[AlertRecord]:
        return self.list_alerts(db, session_id=session_id, limit=limit)
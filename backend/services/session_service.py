"""Session lifecycle + behavior-log persistence.

Study sessions are the core unit of the app: start → live recording of
periodic :class:`BehaviorSnapshot` rows → end with aggregated summary.

None of the aggregate fields are fabricated: if a module was unavailable during
a session its aggregate stays ``None`` (which the UI renders as "Not available").
"""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai.behavior_engine import BehaviorSnapshot
from config import Settings
from database.models import AlertRecord, BehaviorLog, StudySession
from utils.logger import get_logger

_log = get_logger("studygard.services.session")

# Focus levels stored in behavior_logs (kept in sync with behavior_engine)
_FOCUS_HIGH = "high"
_FOCUS_LOW = "low"
_FOCUS_UNKNOWN = "unknown"


class SessionError(ValueError):
    pass


class SessionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log_interval = settings.behavior_log_interval_seconds

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start_session(self, db: Session, student_id: int) -> tuple[StudySession, bool]:
        """Create a new active session (or return the existing one).

        Returns ``(session, created)``; only one live session exists at a time.
        """
        existing = self.get_active_session(db)
        if existing is not None and existing.student_id == student_id:
            return existing, False
        if existing is not None:
            raise SessionError(
                "A session is already active — end it before starting a new one"
            )

        session = StudySession(student_id=student_id, started_at=datetime.utcnow())
        db.add(session)
        db.commit()
        db.refresh(session)
        _log.info("study session %s started (student %s)", session.id, student_id)
        return session, True

    def get_active_session(self, db: Session) -> StudySession | None:
        return db.scalar(
            select(StudySession)
            .where(StudySession.status == "active")
            .order_by(StudySession.started_at.desc())
            .limit(1)
        )

    def get_session(self, db: Session, session_id: int) -> StudySession | None:
        return db.get(StudySession, session_id)

    def list_sessions(self, db: Session, student_id: int | None = None, limit: int = 50) -> list[StudySession]:
        stmt = select(StudySession)
        if student_id is not None:
            stmt = stmt.where(StudySession.student_id == student_id)
        stmt = stmt.order_by(StudySession.started_at.desc()).limit(limit)
        return list(db.scalars(stmt))

    def end_session(self, db: Session, session_id: int) -> StudySession:
        session = db.get(StudySession, session_id)
        if session is None:
            raise SessionError("Session not found")
        if session.status != "active":
            raise SessionError("Session is already ended")

        session.ended_at = datetime.utcnow()
        session.status = "ended"
        logs = list(db.scalars(select(BehaviorLog).where(BehaviorLog.session_id == session_id)))
        self._write_aggregates(db, session, logs)
        session.alert_count = db.scalar(
            select(func.count(AlertRecord.id)).where(AlertRecord.session_id == session_id)
        ) or 0
        db.commit()
        db.refresh(session)
        _log.info("study session %s ended (%.0fs)", session_id, session.duration_seconds or 0)
        return session

    def _write_aggregates(self, db: Session, session: StudySession, logs: list[BehaviorLog]) -> None:
        interval = self._log_interval
        focus_scores = [log.focus_score for log in logs if log.focus_score is not None]
        session.avg_focus_score = round(mean(focus_scores), 1) if focus_scores else None
        session.focused_seconds = int(sum(1 for l in logs if l.focus_level == _FOCUS_HIGH) * interval)
        session.distracted_seconds = int(
            sum(1 for l in logs if l.focus_level == _FOCUS_LOW) * interval
        )
        drowsy = [l for l in logs if l.drowsiness_state is not None]
        session.drowsy_seconds = (
            int(sum(1 for l in drowsy if l.drowsiness_state == "DROWSY") * interval) if drowsy else None
        )
        phone = [l for l in logs if l.phone_detected is not None]
        session.phone_seconds = (
            int(sum(1 for l in phone if l.phone_detected) * interval) if phone else None
        )
        distances = [l.distance_cm for l in logs if l.distance_cm is not None]
        session.avg_distance_cm = round(mean(distances), 1) if distances else None
        posture = [l for l in logs if l.posture_state is not None]
        session.posture_score = (
            round(mean({"GOOD": 100, "MODERATE": 50, "POOR": 0}[l.posture_state] for l in posture), 1)
            if posture
            else None
        )
        if session.started_at is not None and session.ended_at is not None:
            session.duration_seconds = int((session.ended_at - session.started_at).total_seconds())

    # ── live recording ─────────────────────────────────────────────────────

    def record_behavior_log(self, db: Session, session_id: int, snapshot: BehaviorSnapshot) -> BehaviorLog:
        """Persist one periodic behavior snapshot for an active session.

        Only values from *available* modules are stored — future/unimplemented
        modules stay ``None`` (never fabricated as UNKNOWN/zero).
        """
        drowsiness = (
            snapshot.drowsiness.get("state") if snapshot.drowsiness.get("available") else None
        )
        posture = snapshot.posture.get("state") if snapshot.posture.get("available") else None
        distance = (
            snapshot.distance.get("distance_cm") if snapshot.distance.get("available") else None
        )
        phone = snapshot.phone.get("detected") if snapshot.phone.get("available") else None
        log = BehaviorLog(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            presence=snapshot.presence.get("state", "UNKNOWN"),
            gaze_direction=snapshot.gaze.get("state", "UNKNOWN"),
            gaze_attention=snapshot.gaze.get("attention"),
            drowsiness_state=drowsiness,
            posture_state=posture,
            distance_cm=distance,
            phone_detected=phone,
            focus_score=snapshot.focus_score,
            focus_level=snapshot.focus_level or _FOCUS_UNKNOWN,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log

    def add_alerts(self, db: Session, session_id: int, alerts: list[dict]) -> list[AlertRecord]:
        """Persist freshly-fired alerts, bumping the session's alert counter."""
        records = []
        for alert in alerts:
            record = AlertRecord(
                session_id=session_id,
                alert_type=alert["alert_type"],
                severity=alert.get("severity", "warning"),
                message=alert["message"],
                timestamp=datetime.utcnow(),
            )
            db.add(record)
            records.append(record)
        db.flush()
        db.commit()
        for record in records:
            db.refresh(record)
        return records
"""Dashboard aggregation: today's numbers + daily trend (real, DB-backed)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from statistics import mean

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import AlertRecord, BehaviorLog, Student, StudySession
from utils.logger import get_logger

_log = get_logger("studygard.services.dashboard")

_TODAY = date.today()


class DashboardService:
    def __init__(self, log_interval_seconds: float = 2.0) -> None:
        self._interval = log_interval_seconds

    def active_session(self, db: Session) -> StudySession | None:
        return db.scalar(
            select(StudySession)
            .where(StudySession.status == "active")
            .order_by(StudySession.started_at.desc())
            .limit(1)
        )

    def today_overview(self, db: Session, student_id: int) -> dict:
        """Aggregate today's real study data for the active student."""
        start = datetime.combine(_TODAY, datetime.min.time())
        active = self.active_session(db)
        active_elapsed = 0
        if active is not None and active.started_at is not None:
            active_elapsed = int((datetime.utcnow() - active.started_at).total_seconds())

        # Sessions ended today (or still active).
        sessions = db.scalars(
            select(StudySession).where(
                StudySession.student_id == student_id,
                StudySession.started_at >= start,
            )
        ).all()
        ended = [s for s in sessions if s.status == "ended" and s.duration_seconds is not None]
        duration = sum(s.duration_seconds for s in ended) + active_elapsed

        logs = db.scalars(
            select(BehaviorLog).join(StudySession).where(
                StudySession.student_id == student_id,
                BehaviorLog.timestamp >= start,
            )
        ).all()

        focus = [l.focus_score for l in logs if l.focus_score is not None]
        iface = self._interval
        focused_secs = sum(1 for l in logs if l.focus_level == "high") * iface
        distracted_secs = sum(1 for l in logs if l.focus_level == "low") * iface
        drowsy = [l for l in logs if l.drowsiness_state is not None]
        phone = [l for l in logs if l.phone_detected is not None]
        distances = [l.distance_cm for l in logs if l.distance_cm is not None]

        return {
            "student_id": student_id,
            "today": _TODAY.isoformat(),
            "session_count": len(sessions),
            "active_session": active.to_dict() if active else None,
            "study_seconds": duration,
            "avg_focus_score": round(mean(focus), 1) if focus else None,
            "focused_seconds": int(focused_secs),
            "distracted_seconds": int(distracted_secs),
            "drowsy_seconds": (
                int(sum(1 for l in drowsy if l.drowsiness_state == "DROWSY") * iface) if drowsy else None
            ),
            "phone_seconds": (
                int(sum(1 for l in phone if l.phone_detected) * iface) if phone else None
            ),
            "avg_distance_cm": round(mean(distances), 1) if distances else None,
            "alert_count": db.scalar(select(func.count(AlertRecord.id)).join(AlertRecord.session).where(
                AlertRecord.session.has(student_id=student_id), AlertRecord.timestamp >= start
            )) or 0,
        }

    def trend(self, db: Session, student_id: int, days: int = 7) -> list[dict]:
        """Per-day study/total and average focus for the last ``days`` days."""
        today = _TODAY
        results = []
        for offset in range(days - 1, -1, -1):
            day = today - timedelta(days=offset)
            day_start = datetime.combine(day, datetime.min.time())
            day_end = day_start + timedelta(days=1)

            sessions = db.scalars(
                select(StudySession).where(
                    StudySession.student_id == student_id,
                    StudySession.started_at >= day_start,
                    StudySession.started_at < day_end,
                )
            ).all()
            duration = sum(
                s.duration_seconds for s in sessions if s.status == "ended" and s.duration_seconds
            )
            # Include the active session's elapsed time for today.
            if day == today:
                active = self.active_session(db)
                if active is not None and active.started_at is not None:
                    duration += int((datetime.utcnow() - active.started_at).total_seconds())

            logs = db.scalars(
                select(BehaviorLog).join(StudySession).where(
                    StudySession.student_id == student_id,
                    BehaviorLog.timestamp >= day_start,
                    BehaviorLog.timestamp < day_end,
                )
            ).all()
            focus = [l.focus_score for l in logs if l.focus_score is not None]
            results.append(
                {
                    "date": day.isoformat(),
                    "study_seconds": duration,
                    "avg_focus_score": round(mean(focus), 1) if focus else None,
                    "focused_logs": sum(1 for l in logs if l.focus_level == "high"),
                    "distracted_logs": sum(1 for l in logs if l.focus_level == "low"),
                }
            )
        return results

    def recent_sessions(self, db: Session, student_id: int, limit: int = 5) -> list[dict]:
        rows = db.scalars(
            select(StudySession)
            .where(StudySession.student_id == student_id)
            .order_by(StudySession.started_at.desc())
            .limit(limit)
        ).all()
        return [s.to_dict() for s in rows]
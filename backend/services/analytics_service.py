"""Analytics: per-session drill-down and weekly trends (real, DB-backed)."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import BehaviorLog, StudySession
from services.dashboard_service import DashboardService


class AnalyticsService:
    def __init__(self, log_interval_seconds: float = 2.0) -> None:
        self._dashboard = DashboardService(log_interval_seconds)
        self._interval = log_interval_seconds

    def session_stats(self, db: Session, session_id: int) -> dict | None:
        """Summarise one session from its behavior_logs (no fabricated numbers)."""
        session = db.get(StudySession, session_id)
        if session is None:
            return None
        logs = list(db.scalars(select(BehaviorLog).where(BehaviorLog.session_id == session_id)))

        focus = [l.focus_score for l in logs if l.focus_score is not None]
        iface = self._interval
        return {
            "session": session.to_dict(),
            "student_name": session.student.name if session.student else None,
            "log_count": len(logs),
            "focus": {
                "avg": round(mean(focus), 1) if focus else None,
                "min": min(focus) if focus else None,
                "max": max(focus) if focus else None,
                "high_seconds": int(sum(1 for l in logs if l.focus_level == "high") * iface),
                "medium_seconds": int(sum(1 for l in logs if l.focus_level == "medium") * iface),
                "low_seconds": int(sum(1 for l in logs if l.focus_level == "low") * iface),
                "unknown_seconds": int(sum(1 for l in logs if l.focus_level == "unknown") * iface),
            },
            "gaze_distribution": self._distribute(db, session_id, lambda l: l.gaze_direction),
            "presence_distribution": self._distribute(db, session_id, lambda l: l.presence),
            "modules": {
                "drowsiness": any(l.drowsiness_state is not None for l in logs),
                "posture": any(l.posture_state is not None for l in logs),
                "distance": any(l.distance_cm is not None for l in logs),
                "phone": any(l.phone_detected is not None for l in logs),
            },
        }

    def weekly(self, db: Session, student_id: int, days: int = 7) -> dict:
        return {
            "days": self._dashboard.trend(db, student_id, days=days),
            "start_date": (date.today() - timedelta(days=days - 1)).isoformat(),
            "end_date": date.today().isoformat(),
        }

    @staticmethod
    def _distribute(db: Session, session_id: int, pick) -> dict:
        logs = db.scalars(select(BehaviorLog).where(BehaviorLog.session_id == session_id))
        counts: dict[str, int] = {}
        total = 0
        for log in logs:
            key = pick(log) or "UNKNOWN"
            counts[key] = counts.get(key, 0) + 1
            total += 1
        if total == 0:
            return {}
        return {k: round(v / total * 100, 1) for k, v in counts.items()}
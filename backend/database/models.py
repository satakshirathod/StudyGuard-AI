"""SQLAlchemy data model for StudyGuard AI.

Tables
======
* ``students``       — learner profiles (name, email).
* ``sessions``       — one study session per student with start/end + summary.
* ``behavior_logs``  — periodic snapshots of the AI pipeline during a session.
* ``alerts``         — real-time alerts raised during a session.

Only behavioral *metadata* is stored; raw webcam video is never persisted.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all StudyGuard tables."""


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sessions: Mapped[list["StudySession"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StudySession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)

    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_focus_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    focused_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distracted_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    drowsy_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phone_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_distance_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    posture_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    alert_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    student: Mapped["Student"] = relationship(back_populates="sessions")
    behavior_logs: Mapped[list["BehaviorLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["AlertRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "avg_focus_score": self.avg_focus_score,
            "focused_seconds": self.focused_seconds,
            "distracted_seconds": self.distracted_seconds,
            "drowsy_seconds": self.drowsy_seconds,
            "phone_seconds": self.phone_seconds,
            "avg_distance_cm": self.avg_distance_cm,
            "posture_score": self.posture_score,
            "alert_count": self.alert_count,
        }


class BehaviorLog(Base):
    __tablename__ = "behavior_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    presence: Mapped[str] = mapped_column(String(20), nullable=False)
    gaze_direction: Mapped[str] = mapped_column(String(30), nullable=False)
    gaze_attention: Mapped[int | None] = mapped_column(Integer, nullable=True)
    drowsiness_state: Mapped[str | None] = mapped_column(String(20), nullable=True)
    posture_state: Mapped[str | None] = mapped_column(String(20), nullable=True)
    distance_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    phone_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    focus_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    focus_level: Mapped[str] = mapped_column(String(10), nullable=False)

    session: Mapped["StudySession"] = relationship(back_populates="behavior_logs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "presence": self.presence,
            "gaze_direction": self.gaze_direction,
            "gaze_attention": self.gaze_attention,
            "drowsiness_state": self.drowsiness_state,
            "posture_state": self.posture_state,
            "distance_cm": self.distance_cm,
            "phone_detected": self.phone_detected,
            "focus_score": self.focus_score,
            "focus_level": self.focus_level,
        }


class AlertRecord(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    alert_type: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="warning", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped["StudySession"] = relationship(back_populates="alerts")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
        }
"""Student profile management.

A single StudyGuard installation generally tracks one learner at a time; the
``is_active`` flag marks the current learner used by the dashboard/live flow.
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Student


class StudentError(ValueError):
    pass


def _validate_email(email: str | None) -> None:
    if not email:
        return
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise StudentError("Invalid email address")


def create_student(db: Session, name: str, email: str | None = None) -> Student:
    name = (name or "").strip()
    if not name:
        raise StudentError("Name is required")
    if len(name) > 120:
        raise StudentError("Name is too long (max 120 characters)")
    email = (email or "").strip() or None
    _validate_email(email)

    if email:
        existing = db.scalar(select(Student).where(Student.email == email))
        if existing is not None:
            raise StudentError("A student with that email already exists")

    student = Student(name=name, email=email)
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def get_student(db: Session, student_id: int) -> Student | None:
    return db.get(Student, student_id)


def list_students(db: Session) -> list[Student]:
    return list(db.scalars(select(Student).order_by(Student.created_at.desc())))


def set_active_student(db: Session, student_id: int) -> Student:
    """Deactivate all, then activate ``student_id``. Returns the student."""
    student = db.get(Student, student_id)
    if student is None:
        raise StudentError("Student not found")
    for other in db.scalars(select(Student).where(Student.is_active.is_(True))):
        other.is_active = False
    student.is_active = True
    db.commit()
    db.refresh(student)
    return student


def get_active_student(db: Session) -> Student | None:
    return db.scalar(select(Student).where(Student.is_active.is_(True)))


def clear_active_student(db: Session) -> None:
    for student in db.scalars(select(Student).where(Student.is_active.is_(True))):
        student.is_active = False
    db.commit()
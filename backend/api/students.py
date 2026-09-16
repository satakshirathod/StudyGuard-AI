"""Student profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import get_db
from database.models import Student
from services import student_service

router = APIRouter(prefix="/api/students", tags=["students"])


class CreateStudentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str | None = None


@router.get("")
def list_students(db: Session = Depends(get_db)) -> list[dict]:
    return [s.to_dict() for s in student_service.list_students(db)]


@router.post("", status_code=201)
def create_student(body: CreateStudentRequest, db: Session = Depends(get_db)) -> dict:
    try:
        student = student_service.create_student(db, body.name, body.email)
    except student_service.StudentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return student.to_dict()


@router.get("/active")
def active_student(db: Session = Depends(get_db)) -> dict | None:
    student = student_service.get_active_student(db)
    return student.to_dict() if student else None


@router.post("/active")
def set_active(body: dict, db: Session = Depends(get_db)) -> dict:
    student_id = body.get("student_id")
    if not isinstance(student_id, int):
        raise HTTPException(status_code=400, detail="student_id (int) is required")
    try:
        student = student_service.set_active_student(db, student_id)
    except student_service.StudentError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return student.to_dict()


@router.get("/{student_id}")
def get_student(student_id: int, db: Session = Depends(get_db)) -> dict:
    student: Student | None = student_service.get_student(db, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student.to_dict()
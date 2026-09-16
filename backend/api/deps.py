"""Shared FastAPI dependencies / request schemas."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from database.db import get_session


def get_db() -> Session:
    """FastAPI dependency — one short-lived SQLAlchemy session per request."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()
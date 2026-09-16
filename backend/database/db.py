"""Database engine/session management for StudyGuard AI.

SQLite by default (no extra services needed). The engine is created lazily so
services and tests can supply their own SQLAlchemy ``Session`` objects.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base
from utils.logger import get_logger

_log = get_logger("studygard.database")

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(database_url: str) -> None:
    """Create the engine + tables for ``database_url`` and expose a sessionmaker."""
    global _engine, _SessionLocal
    _engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    _log.info("database initialised (%s)", database_url)
    return None


def get_engine() -> Engine:
    """Return the shared engine, raising if :func:`init_db` was not called."""
    if _engine is None:
        raise RuntimeError("database not initialised — call init_db() first")
    return _engine


def get_session() -> Session:
    """Open a new database session (caller must close/commit)."""
    if _SessionLocal is None:
        raise RuntimeError("database not initialised — call init_db() first")
    return _SessionLocal()


def create_all_from_url(database_url: str) -> sessionmaker:
    """Test helper — build engine + tables for an arbitrary URL (e.g. in-memory)."""
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
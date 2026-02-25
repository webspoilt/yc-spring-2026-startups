from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import os

from ..models.database import Base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pmuser:pmpass@localhost:5432/pmcursor"
)

# For SQLite (development)
SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./pmcursor.db")

# Use SQLite if explicitly set or as fallback
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"

if USE_SQLITE or "sqlite" in DATABASE_URL:
    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

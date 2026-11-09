"""PostgreSQL / Relational Database Engine and Session Management.

Configures connection pooling, SessionLocal factory, Declarative Base,
and the get_db FastAPI dependency with automatic transaction rollbacks.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings
from app.core.logging import logger

# Configure SQLAlchemy engine according to target database dialect
engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
}

if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite configuration for local development / testing
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Production PostgreSQL connection pooling parameters
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20
    engine_kwargs["pool_recycle"] = 1800  # recycle connections after 30 min
    engine_kwargs["pool_timeout"] = 30

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session management.

    Yields an active database session and ensures automatic rollback
    if an unhandled exception occurs before closing the session.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception as exc:
        db.rollback()
        logger.error(f"Database transaction error, rolled back: {exc}")
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initializes relational database schema tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Relational database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database schema: {e}")
        raise

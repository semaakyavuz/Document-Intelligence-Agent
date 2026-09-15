"""
session.py

Settings.DATABASE_URL'den bir SQLAlchemy engine + session factory kurar.
init_db() tablolari olusturur (create_all) - gelistirme icin yeterli; semada
degisiklik gerektiginde ileride Alembic gibi bir migration araci eklenebilir,
simdilik gerekmiyor.
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Base


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or Settings()
    return create_engine(settings.DATABASE_URL)


def get_session_factory(engine: Engine | None = None, settings: Settings | None = None) -> sessionmaker[Session]:
    engine = engine or get_engine(settings)
    return sessionmaker(bind=engine)


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> None:
    """Henuz var olmayan tablolari olusturur (CREATE TABLE IF NOT EXISTS mantiginda)."""
    engine = engine or get_engine(settings)
    Base.metadata.create_all(engine)

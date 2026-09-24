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
    # Iki havuz ayari, iki farkli kopma nedeni icin - ikisi birlikte gerekiyor:
    #
    # pool_recycle=240: bulut Postgres'i (Neon) 300 sn bosta kalinca askiya aliyor ve
    #   havuzdaki soket oluyor. 240 sn'den eski baglantiyi havuz kendisi yeniler, yani
    #   bayat soket hic kullanilmaz. Sicak durumda ek maliyeti YOK.
    # pool_pre_ping=True: guvenlik agi. Yaslanmayla ilgisi olmayan kopmalar (ag kesintisi,
    #   havuzlayicinin yeniden baslamasi, sunucu tarafi zaman asimi) icin kullanimdan
    #   once baglantiyi yoklar. Bedeli: baglanti alimi basina bir ek gidis-donus.
    #
    # Neden ikisi birden: recycle tek basina yalnizca "yasi gecmis" baglantiyi yakalar,
    # pre_ping tek basina her alimda o ek gidis-donusu odetir (olculdu: sicak sorgu
    # ortancasi 131 -> 275 ms). Birlikte, sik kullanilan baglantilar recycle sayesinde
    # zaten taze kaldigi icin pre_ping cogu zaman ucuz bir dogrulamaya donusuyor.
    #
    # Olculen sorun: 7 dakika bosta bekledikten sonra havuzdaki baglantiyi kullanmak
    # ayarsiz halde 2 ms'de "SSL connection has been closed unexpectedly" veriyordu -
    # yani her sessizlik sonrasi ILK istek hata aliyor, ikincisi calisiyordu.
    return create_engine(settings.DATABASE_URL, pool_recycle=240, pool_pre_ping=True)


def get_session_factory(engine: Engine | None = None, settings: Settings | None = None) -> sessionmaker[Session]:
    engine = engine or get_engine(settings)
    return sessionmaker(bind=engine)


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> None:
    """Henuz var olmayan tablolari olusturur (CREATE TABLE IF NOT EXISTS mantiginda)."""
    engine = engine or get_engine(settings)
    Base.metadata.create_all(engine)

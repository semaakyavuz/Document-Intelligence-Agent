"""
InvoiceRecord modelini ve app/db/session.py'nin engine/session/init_db yardimcilarini
gercek Postgres'e baglanmadan, SQLite in-memory ile test eder.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.models import Base, InvoiceRecord
from app.db.session import get_engine, get_session_factory, init_db


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_insert_and_query_invoice_record(session_factory):
    with session_factory() as session:
        session.add(InvoiceRecord(
            image_filename="invoice_0001.png",
            invoice_no="2025123456",
            seller_name="Test A.Ş.",
            grand_total=1200.0,
            is_valid=True,
            anomaly_count=0,
            anomalies=[],
            full_report={"summary": "Fatura başarıyla işlendi, anomali yok.", "invoice_no": "2025123456"},
        ))
        session.commit()

    with session_factory() as session:
        fetched = session.query(InvoiceRecord).filter_by(invoice_no="2025123456").one()
        assert fetched.id is not None
        assert fetched.processed_at is not None
        assert fetched.seller_name == "Test A.Ş."
        assert fetched.is_valid is True
        assert fetched.anomalies == []
        assert fetched.full_report["summary"] == "Fatura başarıyla işlendi, anomali yok."


def test_invoice_fields_are_nullable_when_extraction_failed(session_factory):
    """ReportAgent, raw_extraction yoksa invoice_no/seller_name/grand_total'i None birakiyor."""
    with session_factory() as session:
        session.add(InvoiceRecord(
            image_filename="invoice_bad.png",
            invoice_no=None,
            seller_name=None,
            grand_total=None,
            is_valid=False,
            anomaly_count=1,
            anomalies=[{"rule": "missing_extraction", "field": None, "message": "..."}],
            full_report={"summary": "Fatura okunamadı."},
        ))
        session.commit()

    with session_factory() as session:
        fetched = session.query(InvoiceRecord).filter_by(image_filename="invoice_bad.png").one()
        assert fetched.invoice_no is None
        assert fetched.seller_name is None
        assert fetched.grand_total is None
        assert fetched.anomalies[0]["rule"] == "missing_extraction"


def test_anomalies_and_full_report_json_round_trip_turkish_chars(session_factory):
    anomalies = [{
        "rule": "price_range", "field": "items[0].unit_price",
        "message": "'Yazıcı Mürekkebi' için tipik aralık 150-600 TL, bulunan 1915.68.",
    }]
    with session_factory() as session:
        record = InvoiceRecord(
            image_filename="x.png", invoice_no="1", seller_name="Ülker Ltd.", grand_total=1.0,
            is_valid=False, anomaly_count=1, anomalies=anomalies,
            full_report={"seller_name": "Ülker Ltd.", "anomalies": anomalies},
        )
        session.add(record)
        session.commit()
        record_id = record.id

    with session_factory() as session:
        fetched = session.get(InvoiceRecord, record_id)
        assert fetched.seller_name == "Ülker Ltd."
        assert fetched.anomalies == anomalies
        assert fetched.full_report["seller_name"] == "Ülker Ltd."


def test_get_engine_and_init_db_work_end_to_end_with_sqlite(tmp_path):
    """app/db/session.py'nin kamuya acik fonksiyonlarini (settings uzerinden) dogrular."""
    settings = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}")
    engine = get_engine(settings)
    init_db(engine=engine)

    session_factory = get_session_factory(engine=engine)
    with session_factory() as session:
        session.add(InvoiceRecord(
            image_filename="x.png", is_valid=True, anomaly_count=0, anomalies=[], full_report={},
        ))
        session.commit()

    with session_factory() as session:
        assert session.query(InvoiceRecord).count() == 1

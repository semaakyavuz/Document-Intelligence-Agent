"""
app/db/price_history.py'yi ve ReportAgent'in price_history entegrasyonunu, gercek bir
SQLite in-memory veritabanina karsi test eder (Postgres'e ihtiyac yok - bkz. test_db_models.py'deki
ayni desen). Basit metin eslestirmesi (buyuk/kucuk harf ve bosluk normalize edilerek TAM
esitlik) dogrulanir; bulanik eslestirme kapsam disidir.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.report_agent import ReportAgent
from app.db.models import Base, InvoiceRecord
from app.db.price_history import get_price_history
from app.state import PipelineState


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _seed(session_factory, records: list[dict]) -> None:
    with session_factory() as session:
        for report in records:
            session.add(InvoiceRecord.from_final_report(f"{report.get('invoice_no', 'x')}.png", report))
        session.commit()


# --- get_price_history() birim testleri ----------------------------------------

def test_get_price_history_returns_empty_when_no_matching_description():
    session_factory = _session_factory()
    _seed(session_factory, [{
        "is_valid": True, "anomaly_count": 0, "anomalies": [],
        "items": [{"description": "Mouse", "unit_price": 120.0}],
    }])
    with session_factory() as session:
        assert get_price_history(session, ["Klavye"]) == []


def test_get_price_history_collects_prices_case_and_whitespace_insensitively():
    session_factory = _session_factory()
    _seed(session_factory, [
        {"is_valid": True, "anomaly_count": 0, "anomalies": [], "items": [{"description": "Klavye", "unit_price": 450.0}]},
        {"is_valid": True, "anomaly_count": 0, "anomalies": [], "items": [{"description": "  klavye  ", "unit_price": 480.5}]},
        {"is_valid": True, "anomaly_count": 0, "anomalies": [], "items": [{"description": "KLAVYE", "unit_price": 399.0}]},
    ])
    with session_factory() as session:
        result = get_price_history(session, ["Klavye"])

    assert len(result) == 1
    assert result[0]["description"] == "Klavye"
    assert set(result[0]["previous_prices"]) == {450.0, 480.5, 399.0}


def test_get_price_history_ignores_items_without_numeric_unit_price():
    session_factory = _session_factory()
    _seed(session_factory, [{
        "is_valid": True, "anomaly_count": 0, "anomalies": [],
        "items": [{"description": "Klavye", "unit_price": None}, {"description": "Klavye", "unit_price": "okunamadi"}],
    }])
    with session_factory() as session:
        assert get_price_history(session, ["Klavye"]) == []


def test_get_price_history_caps_prices_per_description():
    session_factory = _session_factory()
    records = [
        {"is_valid": True, "anomaly_count": 0, "anomalies": [], "items": [{"description": "Klavye", "unit_price": float(100 + i)}]}
        for i in range(10)
    ]
    _seed(session_factory, records)
    with session_factory() as session:
        result = get_price_history(session, ["Klavye"])
    assert len(result[0]["previous_prices"]) == 5  # MAX_PRICES_PER_DESCRIPTION


def test_get_price_history_returns_empty_for_blank_or_missing_descriptions():
    session_factory = _session_factory()
    with session_factory() as session:
        assert get_price_history(session, ["", "   ", None]) == []  # type: ignore[list-item]


# --- ReportAgent entegrasyonu (gercek session_factory enjekte edilerek) -----------

def test_report_agent_fills_price_history_when_session_factory_given():
    session_factory = _session_factory()
    _seed(session_factory, [
        {"is_valid": True, "anomaly_count": 0, "anomalies": [], "items": [{"description": "Klavye", "unit_price": 450.0}]},
    ])

    extraction = {"invoice_no": "2", "items": [{"description": "Klavye", "unit_price": 999.0}]}
    state = PipelineState(image_path="x.png", raw_extraction=extraction)
    result = ReportAgent(session_factory=session_factory).run(state)

    assert result.final_report["price_history"] == [{"description": "Klavye", "previous_prices": [450.0]}]


def test_report_agent_price_history_accumulates_across_sequential_invoices():
    """Bulk yukleme senaryosu: ayni aciklamali onceki fatura, sonrakinin gecmisinde gorunmeli."""
    session_factory = _session_factory()
    agent = ReportAgent(session_factory=session_factory)

    first = PipelineState(image_path="a.png", raw_extraction={"invoice_no": "1", "items": [{"description": "Klavye", "unit_price": 450.0}]})
    first_result = agent.run(first)
    assert first_result.final_report["price_history"] == []  # henuz gecmis yok

    with session_factory() as session:
        session.add(InvoiceRecord.from_final_report("a.png", first_result.final_report))
        session.commit()

    second = PipelineState(image_path="b.png", raw_extraction={"invoice_no": "2", "items": [{"description": "Klavye", "unit_price": 470.0}]})
    second_result = agent.run(second)

    assert second_result.final_report["price_history"] == [{"description": "Klavye", "previous_prices": [450.0]}]

"""
ReportAgent'in raw_extraction/anomalies/is_valid'den final_report ozeti kurma
mantigini test eder. LLM/ag yok, tamamen yerel/saf mantik (risk skoru, aciklama,
validation_checklist/execution_trace aktarimi dahil). price_history icin gercek bir
DB'ye baglanmadan: session_factory verilmezse hep bos liste doner (bkz. app/db/price_history.py),
gercek DB'ye karsi davranis tests/test_report_agent_price_history.py'de (SQLite ile) test edilir.
"""

from app.agents.report_agent import ReportAgent
from app.state import PipelineState

VALID_EXTRACTION = {
    "invoice_no": "2025123456",
    "seller_name": "Test A.Ş.",
    "grand_total": 1200.0,
}

ONE_ANOMALY = [{"rule": "vat_rate", "field": "items[0].vat_rate", "message": "..."}]
TWO_ANOMALIES = ONE_ANOMALY + [{"rule": "format", "field": "invoice_no", "message": "..."}]


def test_report_summarizes_valid_invoice_with_no_anomalies():
    state = PipelineState(
        image_path="x.png", raw_extraction=VALID_EXTRACTION, anomalies=[], is_valid=True,
    )
    result = ReportAgent().run(state)

    assert result.final_report == {
        "summary": "Fatura başarıyla işlendi, anomali yok.",
        "invoice_no": "2025123456",
        "seller_name": "Test A.Ş.",
        "grand_total": 1200.0,
        "anomaly_count": 0,
        "anomalies": [],
        "is_valid": True,
        "items": [],
        "validation_checklist": [],
        "risk_score": 100,
        "risk_level": "passed",
        "explanation": "Fatura tüm kural kontrollerinden sorunsuz geçti; herhangi bir tutarsızlık tespit edilmedi.",
        "price_history": [],
        "execution_trace": [],
    }


def test_report_summarizes_invoice_with_anomalies():
    state = PipelineState(
        image_path="x.png", raw_extraction=VALID_EXTRACTION, anomalies=TWO_ANOMALIES, is_valid=False,
    )
    result = ReportAgent().run(state)

    assert result.final_report["summary"] == "Fatura işlendi, 2 anomali tespit edildi."
    assert result.final_report["anomaly_count"] == 2
    assert result.final_report["anomalies"] == TWO_ANOMALIES  # aynen aktarilir
    assert result.final_report["is_valid"] is False
    assert result.final_report["invoice_no"] == "2025123456"
    assert result.final_report["risk_score"] == 70  # 100 - 2*15
    assert result.final_report["risk_level"] == "review_required"
    assert result.final_report["explanation"].startswith("Faturada 2 anomali tespit edildi:")


def test_report_handles_missing_raw_extraction_without_crashing():
    """Vision Agent basarisiz olmus (raw_extraction None): summary 'Fatura okunamadi',
    fatura alanlari None, ama ValidationAgent'in urettigi anomalies/is_valid aynen aktarilir."""
    missing_extraction_anomaly = [{"rule": "missing_extraction", "field": None, "message": "Vision Agent veri çıkaramadı."}]
    state = PipelineState(
        image_path="x.png", raw_extraction=None, anomalies=missing_extraction_anomaly, is_valid=False,
    )
    result = ReportAgent().run(state)

    assert result.final_report == {
        "summary": "Fatura okunamadı.",
        "invoice_no": None,
        "seller_name": None,
        "grand_total": None,
        "anomaly_count": 1,
        "anomalies": missing_extraction_anomaly,
        "is_valid": False,
        "items": [],
        "validation_checklist": [],
        "risk_score": 0,  # missing_extraction ozel durumu: hicbir sey dogrulanamadi, otomatik 0
        "risk_level": "rejected",
        "explanation": "Vision Agent veri çıkaramadı.",  # ozel durum: tek mesaj aynen aktarilir
        "price_history": [],
        "execution_trace": [],
    }


def test_report_handles_extraction_missing_reported_fields():
    """raw_extraction var ama invoice_no/seller_name/grand_total icermiyor: cokmeden None doner."""
    state = PipelineState(image_path="x.png", raw_extraction={"items": []}, anomalies=[], is_valid=True)
    result = ReportAgent().run(state)

    assert result.final_report["invoice_no"] is None
    assert result.final_report["seller_name"] is None
    assert result.final_report["grand_total"] is None
    assert result.final_report["summary"] == "Fatura başarıyla işlendi, anomali yok."


def test_report_does_not_mutate_input_state():
    original = PipelineState(image_path="x.png", raw_extraction=VALID_EXTRACTION)
    ReportAgent().run(original)
    assert original.final_report is None


def test_report_agent_requires_no_constructor_arguments():
    ReportAgent()  # llm_provider gerekmiyor; hata firlatmamali


# --- items / validation_checklist / execution_trace aktarimi ------------------

def test_report_copies_items_from_raw_extraction():
    extraction = {**VALID_EXTRACTION, "items": [{"description": "Klavye", "unit_price": 450.0}]}
    state = PipelineState(image_path="x.png", raw_extraction=extraction)
    result = ReportAgent().run(state)
    assert result.final_report["items"] == [{"description": "Klavye", "unit_price": 450.0}]


def test_report_passes_through_validation_checklist_and_execution_trace():
    checklist = [{"rule": "math_consistency", "passed": True, "message": "Sorun tespit edilmedi."}]
    trace = [{"step": "vision", "duration_ms": 12.3}]
    state = PipelineState(
        image_path="x.png", raw_extraction=VALID_EXTRACTION,
        validation_checklist=checklist, execution_trace=trace,
    )
    result = ReportAgent().run(state)
    assert result.final_report["validation_checklist"] == checklist
    assert result.final_report["execution_trace"] == trace


# --- risk_score / risk_level -----------------------------------------------------

def test_risk_score_decreases_by_15_per_anomaly_and_floors_at_zero():
    for count, expected_score, expected_level in [
        (0, 100, "passed"),
        (1, 85, "review_required"),
        (3, 55, "review_required"),
        (4, 40, "review_required"),
        (5, 25, "rejected"),
        (10, 0, "rejected"),
    ]:
        anomalies = [{"rule": "vat_rate", "field": None, "message": f"anomali {i}"} for i in range(count)]
        state = PipelineState(image_path="x.png", raw_extraction=VALID_EXTRACTION, anomalies=anomalies, is_valid=count == 0)
        result = ReportAgent().run(state)
        assert result.final_report["risk_score"] == expected_score, f"count={count}"
        assert result.final_report["risk_level"] == expected_level, f"count={count}"


def test_missing_extraction_is_automatically_rejected_regardless_of_formula():
    anomaly = [{"rule": "missing_extraction", "field": None, "message": "..."}]
    state = PipelineState(image_path="x.png", raw_extraction=None, anomalies=anomaly, is_valid=False)
    result = ReportAgent().run(state)
    assert result.final_report["risk_score"] == 0
    assert result.final_report["risk_level"] == "rejected"


# --- price_history varsayilani (session_factory verilmezse) --------------------

def test_price_history_is_empty_list_when_no_session_factory_configured():
    extraction = {**VALID_EXTRACTION, "items": [{"description": "Klavye", "unit_price": 450.0}]}
    state = PipelineState(image_path="x.png", raw_extraction=extraction)
    result = ReportAgent().run(state)  # session_factory verilmedi
    assert result.final_report["price_history"] == []

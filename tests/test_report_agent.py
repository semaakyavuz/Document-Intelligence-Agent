"""
ReportAgent'in raw_extraction/anomalies/is_valid'den final_report ozeti kurma
mantigini test eder. LLM/ag yok, tamamen yerel/saf mantik.
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


def test_report_handles_missing_raw_extraction_without_crashing():
    """Vision Agent basarisiz olmus (raw_extraction None): summary 'Fatura okunamadi',
    fatura alanlari None, ama ValidationAgent'in urettigi anomalies/is_valid aynen aktarilir."""
    missing_extraction_anomaly = [{"rule": "missing_extraction", "field": None, "message": "..."}]
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

"""
report_agent.py

raw_extraction, anomalies ve is_valid'i tek, okunabilir bir ozet sozluge
(state.final_report) donusturur. LLM kullanmaz, sadece pipeline'in daha
onceki agent'larinin urettigi state alanlarini bir araya getirir.

ReportAgent de RAGAgent/ValidationAgent gibi BaseAgent.__init__'i (llm_provider
bekler) miras almiyor: hicbir bagimliligi olmadan kurulur. BaseAgent.__init__
soyut olmadigi icin bu gecerli bir ABC kullanimi (bkz. rag_agent.py'deki ayni gerekce).
"""

from app.agents.base import BaseAgent
from app.state import PipelineState


class ReportAgent(BaseAgent):
    """anomalies/is_valid/raw_extraction'i tek bir ozet rapora (final_report) toplar."""

    def __init__(self) -> None:
        pass

    def run(self, state: PipelineState) -> PipelineState:
        extraction = state.raw_extraction
        anomaly_count = len(state.anomalies)

        if not extraction:
            summary = "Fatura okunamadı."
        elif anomaly_count == 0:
            summary = "Fatura başarıyla işlendi, anomali yok."
        else:
            summary = f"Fatura işlendi, {anomaly_count} anomali tespit edildi."

        final_report = {
            "summary": summary,
            "invoice_no": extraction.get("invoice_no") if extraction else None,
            "seller_name": extraction.get("seller_name") if extraction else None,
            "grand_total": extraction.get("grand_total") if extraction else None,
            "anomaly_count": anomaly_count,
            "anomalies": state.anomalies,
            "is_valid": state.is_valid,
        }
        return state.model_copy(update={"final_report": final_report})

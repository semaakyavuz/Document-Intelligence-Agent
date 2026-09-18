"""
report_agent.py

raw_extraction, anomalies ve is_valid'i tek, okunabilir bir ozet sozluge
(state.final_report) donusturur. LLM kullanmaz, sadece pipeline'in daha
onceki agent'larinin urettigi state alanlarini bir araya getirir; ek olarak
kod-ile (LLM'e tekrar sormadan) turetilen risk skoru/aciklama ve (varsa) DB'den
gecmis fiyat karsilastirmasi hesaplar.

ReportAgent de RAGAgent/ValidationAgent gibi BaseAgent.__init__'i (llm_provider
bekler) miras almiyor: hicbir bagimliligi olmadan kurulur. BaseAgent.__init__
soyut olmadigi icin bu gecerli bir ABC kullanimi (bkz. rag_agent.py'deki ayni gerekce).

session_factory verilmezse (varsayilan) price_history her zaman bos liste doner -
bu bir hata degil (bkz. app/db/price_history.py'nin ayni notu); yalnizca DB'ye
baglanmasi istenen yerlerde (app/api/main.py, scripts/save_pipeline_result.py)
acikca enjekte edilir.
"""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.agents.base import BaseAgent
from app.db.price_history import get_price_history
from app.state import PipelineState

# Basit, aciklanabilir risk formulu: her anomali -15 puan, 100'den baslar, 0'in altina inmez.
# Istisna: "missing_extraction" (Vision Agent hic veri uretememis) tek basina bir kural
# ihlali degil, hicbir sey dogrulanamadigi anlamina gelir - otomatik "rejected"/0 puan.
RISK_POINTS_PER_ANOMALY = 15
RISK_REVIEW_THRESHOLD = 40  # bu puanin altinda "rejected", esit/uzeri "review_required"


def _is_missing_extraction(anomalies: list[dict]) -> bool:
    return len(anomalies) == 1 and anomalies[0]["rule"] == "missing_extraction"


def _risk_score(anomalies: list[dict]) -> int:
    if _is_missing_extraction(anomalies):
        return 0
    return max(0, 100 - RISK_POINTS_PER_ANOMALY * len(anomalies))


def _risk_level(score: int, anomalies: list[dict]) -> str:
    if not anomalies:
        return "passed"
    if _is_missing_extraction(anomalies):
        return "rejected"
    if score >= RISK_REVIEW_THRESHOLD:
        return "review_required"
    return "rejected"


def _build_explanation(anomalies: list[dict]) -> str:
    """LLM'e tekrar sormadan, anomali listesinden sablon bazli Turkce bir aciklama kurar."""
    if not anomalies:
        return "Fatura tüm kural kontrollerinden sorunsuz geçti; herhangi bir tutarsızlık tespit edilmedi."

    if len(anomalies) == 1 and anomalies[0]["rule"] == "missing_extraction":
        return anomalies[0]["message"]

    lines = [f"Faturada {len(anomalies)} anomali tespit edildi:"]
    lines.extend(f"- {a['message']}" for a in anomalies)
    lines.append("Bu bulgular manuel incelemeyi gerektirebilir.")
    return "\n".join(lines)


class ReportAgent(BaseAgent):
    """anomalies/is_valid/raw_extraction'i tek bir ozet rapora (final_report) toplar."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory

    def run(self, state: PipelineState) -> PipelineState:
        extraction = state.raw_extraction
        anomaly_count = len(state.anomalies)
        items = extraction.get("items") if extraction else None
        items = items if isinstance(items, list) else []

        if not extraction:
            summary = "Fatura okunamadı."
        elif anomaly_count == 0:
            summary = "Fatura başarıyla işlendi, anomali yok."
        else:
            summary = f"Fatura işlendi, {anomaly_count} anomali tespit edildi."

        score = _risk_score(state.anomalies)

        final_report = {
            "summary": summary,
            "invoice_no": extraction.get("invoice_no") if extraction else None,
            "seller_name": extraction.get("seller_name") if extraction else None,
            "grand_total": extraction.get("grand_total") if extraction else None,
            "anomaly_count": anomaly_count,
            "anomalies": state.anomalies,
            "is_valid": state.is_valid,
            "items": items,
            "validation_checklist": state.validation_checklist,
            "risk_score": score,
            "risk_level": _risk_level(score, state.anomalies),
            "explanation": _build_explanation(state.anomalies),
            "price_history": self._get_price_history(items),
            "execution_trace": state.execution_trace,
        }
        return state.model_copy(update={"final_report": final_report})

    def _get_price_history(self, items: list) -> list[dict]:
        if not self._session_factory:
            return []
        descriptions = [item.get("description") for item in items if isinstance(item, dict)]
        if not descriptions:
            return []
        try:
            with self._session_factory() as session:
                return get_price_history(session, descriptions)
        except SQLAlchemyError:
            # price_history "varsa ekle, yoksa bos liste" bir zenginlestirme - gecici bir
            # DB sorunu yuzunden tum pipeline'i (Vision/RAG/Validation zaten tamamlanmis)
            # basarisiz saymak yanlis olur.
            return []

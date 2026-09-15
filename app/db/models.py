"""
models.py

SQLAlchemy 2.0 stili (Mapped/mapped_column) ORM modelleri. InvoiceRecord, Report
Agent'in urettigi final_report'un (bkz. app/agents/report_agent.py) veritabanina
yazilmis halidir.
"""

import datetime

from sqlalchemy import JSON, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Tum ORM modellerinin ortak temeli; init_db()'deki create_all bunun uzerinden calisir."""


class InvoiceRecord(Base):
    """Bir pipeline kosusunun (Vision->RAG->Validation->Report) kaydedilmis sonucu."""

    __tablename__ = "invoice_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_filename: Mapped[str] = mapped_column()
    processed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # final_report'tan ayri sutunlara cikarilan, sorgulamasi sik olabilecek alanlar.
    # raw_extraction'da bulunmayabilirler (Vision Agent basarisiz olmus olabilir), o yuzden
    # nullable - ReportAgent zaten bu durumda None birakiyor (bkz. report_agent.py).
    invoice_no: Mapped[str | None] = mapped_column()
    seller_name: Mapped[str | None] = mapped_column()
    grand_total: Mapped[float | None] = mapped_column()

    is_valid: Mapped[bool] = mapped_column()
    anomaly_count: Mapped[int] = mapped_column()
    anomalies: Mapped[list] = mapped_column(JSON)  # final_report["anomalies"] aynen

    full_report: Mapped[dict] = mapped_column(JSON)  # final_report'un tamami, aynen

    @classmethod
    def from_final_report(cls, image_filename: str, final_report: dict) -> "InvoiceRecord":
        """ReportAgent'in urettigi final_report'u (bkz. app/agents/report_agent.py) bir
        InvoiceRecord'a cevirir. scripts/save_pipeline_result.py ve app/api/main.py
        ayni donusumu kullanir, mantik tek yerde tutulur."""
        return cls(
            image_filename=image_filename,
            invoice_no=final_report.get("invoice_no"),
            seller_name=final_report.get("seller_name"),
            grand_total=final_report.get("grand_total"),
            is_valid=final_report["is_valid"],
            anomaly_count=final_report["anomaly_count"],
            anomalies=final_report["anomalies"],
            full_report=final_report,
        )

    def __repr__(self) -> str:
        return (
            f"InvoiceRecord(id={self.id!r}, image_filename={self.image_filename!r}, "
            f"invoice_no={self.invoice_no!r}, is_valid={self.is_valid!r})"
        )

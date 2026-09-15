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

    def __repr__(self) -> str:
        return (
            f"InvoiceRecord(id={self.id!r}, image_filename={self.image_filename!r}, "
            f"invoice_no={self.invoice_no!r}, is_valid={self.is_valid!r})"
        )

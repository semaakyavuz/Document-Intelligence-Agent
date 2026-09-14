"""
state.py

LangGraph node'lari (agent'lar) arasinda tasinan ortak durum.
Her agent bu nesneyi alir, kendi sorumlu oldugu alanlari doldurur ve geri dondurur.
"""

from pydantic import BaseModel, ConfigDict, Field


class PipelineState(BaseModel):
    """Bir faturanin pipeline boyunca urettigi tum ara ve nihai sonuclar.

    Alanlar pipeline sirasiyla dizilidir. Yeni bir agent eklerken alanini
    buraya ekleyin; extra="forbid" sayesinde yanlis yazilan alan adi aninda hata verir.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    raw_extraction: dict | None = None        # Vision Agent'in ham ciktisi
    retrieved_rules: list[str] = Field(default_factory=list)  # RAG Agent'in getirdigi ilgili kural metinleri
    validated_data: dict | None = None        # Validation Agent'in onayladigi veri
    validation_errors: list[str] = Field(default_factory=list)  # bos degilse retry dongusu tetiklenir
    retry_count: int = 0
    final_report: dict | None = None          # Report Agent'in son ciktisi

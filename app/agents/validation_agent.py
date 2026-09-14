"""
validation_agent.py

Cikarilan fatura verisini (state.raw_extraction) ve RAG Agent'in getirdigi kural
metinlerini (state.retrieved_rules) kural bazli, kucuk tek-sorumluluklu kontrol
siniflariyla denetler. LLM kullanmaz.

ValidationAgent de RAGAgent gibi BaseAgent.__init__'i (llm_provider bekler) miras
almiyor: bir Checker listesi aliyor. BaseAgent.__init__ soyut olmadigi icin bu
gecerli bir ABC kullanimi (bkz. rag_agent.py'deki ayni gerekce).
"""

import re
from abc import ABC, abstractmethod

from app.agents.base import BaseAgent
from app.state import PipelineState


def _to_float(value) -> float | None:
    """Sayiya donusturmeyi dener; olmuyorsa None dondurur (asla hata firlatmaz).

    LLM ciktisinin semasi garanti degildir: sayi virgullu string, eksik ya da
    yanlis tipte gelebilir. Boyle durumlarda ilgili alt-kontrol sessizce atlanir."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return None
    return None


def _anomaly(rule: str, field: str | None, message: str) -> dict:
    return {"rule": rule, "field": field, "message": message}


class Checker(ABC):
    """Tek bir kural setini denetleyen kucuk, tek sorumluluklu kontrol sinifi arayuzu."""

    @abstractmethod
    def check(self, extraction: dict, retrieved_rules: list[str]) -> list[dict]:
        """Anomali sozlukleri listesi dondurur (sorun yoksa bos liste)."""


class MathConsistencyCheck(Checker):
    """Kalem ve toplam matematiginin tutarliligini kontrol eder (0.01 yuvarlama toleransi)."""

    RULE_NAME = "math_consistency"
    TOLERANCE = 0.01

    def check(self, extraction: dict, retrieved_rules: list[str]) -> list[dict]:
        anomalies: list[dict] = []
        items = extraction.get("items")
        if not isinstance(items, list):
            return anomalies

        line_totals: list[float] = []
        all_line_totals_known = True

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                all_line_totals_known = False
                continue

            quantity = _to_float(item.get("quantity"))
            unit_price = _to_float(item.get("unit_price"))
            line_total = _to_float(item.get("line_total"))
            vat_rate = _to_float(item.get("vat_rate"))
            vat_amount = _to_float(item.get("vat_amount"))

            if line_total is not None:
                line_totals.append(line_total)
            else:
                all_line_totals_known = False

            if quantity is not None and unit_price is not None and line_total is not None:
                expected = round(quantity * unit_price, 2)
                if abs(expected - line_total) > self.TOLERANCE:
                    anomalies.append(_anomaly(
                        self.RULE_NAME, f"items[{index}].line_total",
                        f"Beklenen {expected} (miktar*birim_fiyat), bulunan {line_total}.",
                    ))

            if line_total is not None and vat_rate is not None and vat_amount is not None:
                expected = round(line_total * vat_rate, 2)
                if abs(expected - vat_amount) > self.TOLERANCE:
                    anomalies.append(_anomaly(
                        self.RULE_NAME, f"items[{index}].vat_amount",
                        f"Beklenen {expected} (satır_toplamı*kdv_oranı), bulunan {vat_amount}.",
                    ))

        subtotal = _to_float(extraction.get("subtotal"))
        # Tum kalemlerin line_total'i bilinmiyorsa kismi toplamla yanlis pozitif uretmemek
        # icin subtotal kontrolu hic calistirilmaz.
        if all_line_totals_known and line_totals and subtotal is not None:
            expected_subtotal = round(sum(line_totals), 2)
            if abs(expected_subtotal - subtotal) > self.TOLERANCE:
                anomalies.append(_anomaly(
                    self.RULE_NAME, "subtotal",
                    f"Beklenen {expected_subtotal} (kalemlerin toplamı), bulunan {subtotal}.",
                ))

        vat_total = _to_float(extraction.get("vat_total"))
        grand_total = _to_float(extraction.get("grand_total"))
        if subtotal is not None and vat_total is not None and grand_total is not None:
            expected_grand = round(subtotal + vat_total, 2)
            if abs(expected_grand - grand_total) > self.TOLERANCE:
                anomalies.append(_anomaly(
                    self.RULE_NAME, "grand_total",
                    f"Beklenen {expected_grand} (ara_toplam+kdv_toplamı), bulunan {grand_total}.",
                ))

        return anomalies


class VatRateCheck(Checker):
    """Her kalemin KDV oraninin Turkiye'deki gecerli oranlardan biri oldugunu kontrol eder."""

    RULE_NAME = "vat_rate"
    VALID_RATES = (0.01, 0.10, 0.20)
    RATE_TOLERANCE = 1e-6

    def check(self, extraction: dict, retrieved_rules: list[str]) -> list[dict]:
        anomalies: list[dict] = []
        items = extraction.get("items")
        if not isinstance(items, list):
            return anomalies

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            vat_rate = _to_float(item.get("vat_rate"))
            if vat_rate is None:
                continue
            if not any(abs(vat_rate - valid) <= self.RATE_TOLERANCE for valid in self.VALID_RATES):
                anomalies.append(_anomaly(
                    self.RULE_NAME, f"items[{index}].vat_rate",
                    f"Geçerli KDV oranlarından biri değil (%1/%10/%20 beklenir), bulunan {vat_rate}.",
                ))
        return anomalies


class FormatCheck(Checker):
    """Fatura numarasi (yil+6 hane) ve vergi numarasi (10 hane) formatini kontrol eder."""

    RULE_NAME = "format"
    _TEN_DIGITS = re.compile(r"^\d{10}$")

    def check(self, extraction: dict, retrieved_rules: list[str]) -> list[dict]:
        anomalies: list[dict] = []

        invoice_no = extraction.get("invoice_no")
        if isinstance(invoice_no, str) and not self._TEN_DIGITS.match(invoice_no):
            anomalies.append(_anomaly(
                self.RULE_NAME, "invoice_no",
                f"Fatura numarası formatı geçersiz (yıl+6 hane, 10 haneli sayısal beklenir), bulunan {invoice_no!r}.",
            ))

        seller_tax_no = extraction.get("seller_tax_no")
        if isinstance(seller_tax_no, str) and not self._TEN_DIGITS.match(seller_tax_no):
            anomalies.append(_anomaly(
                self.RULE_NAME, "seller_tax_no",
                f"Vergi numarası formatı geçersiz (10 haneli sayısal beklenir), bulunan {seller_tax_no!r}.",
            ))

        return anomalies


class PriceRangeParser:
    """retrieved_rules metinlerinden 'Ürün ... genellikle X-Y TL arasındadır' bicimindeki
    tipik fiyat araliklarini cikarir (bkz. data/knowledge_base/birim_fiyat_*.txt)."""

    _PATTERN = re.compile(
        r"([A-ZÇĞİÖŞÜ][\w]*(?:\s+[A-ZÇĞİÖŞÜ0-9][\w]*)*)\s*(?:\([^)]*\)\s*)?genellikle\s+(\d+)-(\d+)\s*TL"
    )

    def parse(self, retrieved_rules: list[str]) -> dict[str, tuple[float, float]]:
        ranges: dict[str, tuple[float, float]] = {}
        for text in retrieved_rules:
            for name, low, high in self._PATTERN.findall(text):
                ranges[name.strip().casefold()] = (float(low), float(high))
        return ranges


class PriceRangeCheck(Checker):
    """Kalem birim fiyatlarinin bilgi tabanindaki tipik araliklar icinde olup olmadigini kontrol eder.

    Bir kalemin urunu icin aralik cikarilamazsa (ilgili kural getirilmemis ya da metin
    regex'e uymuyorsa) o kalem sessizce atlanir; bu bir hata degildir."""

    RULE_NAME = "price_range"

    def __init__(self, parser: PriceRangeParser | None = None):
        self._parser = parser or PriceRangeParser()

    def check(self, extraction: dict, retrieved_rules: list[str]) -> list[dict]:
        anomalies: list[dict] = []
        items = extraction.get("items")
        if not isinstance(items, list) or not retrieved_rules:
            return anomalies

        ranges = self._parser.parse(retrieved_rules)
        if not ranges:
            return anomalies

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            if not isinstance(description, str):
                continue
            price_range = ranges.get(description.strip().casefold())
            if price_range is None:
                continue
            unit_price = _to_float(item.get("unit_price"))
            if unit_price is None:
                continue

            low, high = price_range
            if not (low <= unit_price <= high):
                anomalies.append(_anomaly(
                    self.RULE_NAME, f"items[{index}].unit_price",
                    f"'{description}' için tipik aralık {low:g}-{high:g} TL, bulunan {unit_price}.",
                ))
        return anomalies


DEFAULT_CHECKERS: tuple[Checker, ...] = (
    MathConsistencyCheck(),
    VatRateCheck(),
    FormatCheck(),
    PriceRangeCheck(),
)


class ValidationAgent(BaseAgent):
    """Cikarilan fatura verisini kural bazli kontrol sinifi listesiyle denetler; LLM kullanmaz."""

    def __init__(self, checkers: tuple[Checker, ...] | list[Checker] | None = None):
        self._checkers = tuple(checkers) if checkers is not None else DEFAULT_CHECKERS

    def run(self, state: PipelineState) -> PipelineState:
        if not state.raw_extraction:
            # Vision Agent hic veri uretmemis: hangi kuralin ihlal edildigini soylemek
            # anlamsiz, dogrudan gecersiz sayilir.
            anomaly = _anomaly(
                "missing_extraction", None,
                "Vision Agent veri çıkaramadığı için doğrulama yapılamadı.",
            )
            return state.model_copy(update={"anomalies": [anomaly], "is_valid": False})

        anomalies: list[dict] = []
        for checker in self._checkers:
            anomalies.extend(checker.check(state.raw_extraction, state.retrieved_rules))

        return state.model_copy(update={"anomalies": anomalies, "is_valid": not anomalies})

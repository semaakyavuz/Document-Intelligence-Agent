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
from typing import ClassVar, Literal

from app.agents.base import BaseAgent
from app.state import PipelineState

FormatProfile = Literal["strict_tr", "lenient"]


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
    def check(self, extraction: dict, retrieved_rules: list[dict]) -> list[dict]:
        """Anomali sozlukleri listesi dondurur (sorun yoksa bos liste)."""


class MathConsistencyCheck(Checker):
    """Kalem ve toplam matematiginin tutarliligini kontrol eder (0.01 yuvarlama toleransi)."""

    RULE_NAME = "math_consistency"
    TOLERANCE = 0.01

    # Not: asagidaki yardimcilara bolunmus hali, tek govdeli halinin BIREBIR aynisi -
    # anomalilerin uretim SIRASI da korunuyor (kalem basina once line_total, sonra
    # vat_amount; tum kalemler bittikten sonra subtotal, en son grand_total).
    def check(self, extraction: dict, retrieved_rules: list[dict]) -> list[dict]:
        items = extraction.get("items")
        if not isinstance(items, list):
            return []

        anomalies, line_totals, all_line_totals_known = self._check_items(items)
        anomalies.extend(self._check_totals(extraction, line_totals, all_line_totals_known))
        return anomalies

    def _check_items(self, items: list) -> tuple[list[dict], list[float], bool]:
        """Kalem bazli kontroller.

        (anomaliler, bilinen line_total degerleri, hepsi_bilindi_mi) dondurur; son iki
        deger subtotal kontrolunun calisip calismayacagina karar vermek icin gerekiyor."""
        anomalies: list[dict] = []
        line_totals: list[float] = []
        all_line_totals_known = True

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                all_line_totals_known = False
                continue

            line_total = _to_float(item.get("line_total"))
            if line_total is not None:
                line_totals.append(line_total)
            else:
                all_line_totals_known = False

            anomalies.extend(self._check_line_total(index, item, line_total))
            anomalies.extend(self._check_vat_amount(index, item, line_total))

        return anomalies, line_totals, all_line_totals_known

    def _check_line_total(self, index: int, item: dict, line_total: float | None) -> list[dict]:
        """miktar * birim_fiyat == kalem toplami"""
        quantity = _to_float(item.get("quantity"))
        unit_price = _to_float(item.get("unit_price"))
        if quantity is None or unit_price is None or line_total is None:
            return []

        expected = round(quantity * unit_price, 2)
        if abs(expected - line_total) <= self.TOLERANCE:
            return []
        return [_anomaly(
            self.RULE_NAME, f"items[{index}].line_total",
            f"Beklenen {expected} (miktar*birim_fiyat), bulunan {line_total}.",
        )]

    def _check_vat_amount(self, index: int, item: dict, line_total: float | None) -> list[dict]:
        """kalem toplami * kdv orani == kdv tutari"""
        vat_rate = _to_float(item.get("vat_rate"))
        vat_amount = _to_float(item.get("vat_amount"))
        if line_total is None or vat_rate is None or vat_amount is None:
            return []

        expected = round(line_total * vat_rate, 2)
        if abs(expected - vat_amount) <= self.TOLERANCE:
            return []
        return [_anomaly(
            self.RULE_NAME, f"items[{index}].vat_amount",
            f"Beklenen {expected} (satır_toplamı*kdv_oranı), bulunan {vat_amount}.",
        )]

    def _check_totals(
        self, extraction: dict, line_totals: list[float], all_line_totals_known: bool
    ) -> list[dict]:
        """Fatura duzeyindeki toplamlar: ara toplam ve genel toplam."""
        anomalies: list[dict] = []
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

    def check(self, extraction: dict, retrieved_rules: list[dict]) -> list[dict]:
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
    """Fatura numarasi ve vergi numarasinin formatini denetler; iki profil destekler.

    strict_tr: Turk formatinin tam 10 haneli sayisal olmasini zorunlu kilar (sadece
    Turk formatinda sentetik veride kullanilmali). lenient: sadece alanin bos/anlamsiz
    olmadigini kontrol eder, herhangi bir uzunluk/hane formatini dayatmaz - yabanci
    formatli gercek faturalari yanlislikla anomali diye isaretlememek icin (bkz.
    Settings.FORMAT_PROFILE, app/config.py)."""

    RULE_NAME = "format"
    _TEN_DIGITS = re.compile(r"^\d{10}$")
    _PLACEHOLDER_VALUES: ClassVar[set[str]] = {
        "", "none", "null", "n/a", "na", "yok", "bilinmiyor", "unknown", "-", "belirtilmemis",
    }

    def __init__(self, profile: FormatProfile):
        self._profile = profile

    def check(self, extraction: dict, retrieved_rules: list[dict]) -> list[dict]:
        if self._profile == "strict_tr":
            return self._check_strict_tr(extraction)
        return self._check_lenient(extraction)

    def _check_strict_tr(self, extraction: dict) -> list[dict]:
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

    def _check_lenient(self, extraction: dict) -> list[dict]:
        anomalies: list[dict] = []
        for field, label in (("invoice_no", "Fatura numarası"), ("seller_tax_no", "Vergi numarası")):
            value = extraction.get(field)
            if not self._looks_meaningful(value):
                anomalies.append(_anomaly(
                    self.RULE_NAME, field, f"{label} alanı boş ya da anlamsız görünüyor: {value!r}.",
                ))
        return anomalies

    @classmethod
    def _looks_meaningful(cls, value) -> bool:
        if value is None:
            return False
        text = str(value).strip().casefold()
        return bool(text) and text not in cls._PLACEHOLDER_VALUES


class PriceRangeParser:
    """retrieved_rules metinlerinden 'Ürün ... genellikle X-Y TL arasındadır' bicimindeki
    tipik fiyat araliklarini cikarir (bkz. data/knowledge_base/birim_fiyat_*.txt).

    Her rule ya {"text": ...} seklinde bir sozluk (RAGAgent'in gercek MCP ciktisi -
    ayrica "score"/"source" da tasir ama burada kullanilmaz) ya da duz bir metin
    (testlerde dogrudan parser/checker birim testi icin) olabilir - ikisi de kabul edilir."""

    # Desen parcalanip yorumlandi (re.VERBOSE): tek satirlik hali okunamaz haldeydi ve
    # gereksiz [\w] sarmalamalari iceriyordu. ESLESTIGI SEY BIREBIR AYNI - eski ve yeni
    # desen 36 girdide (12'si gercek data/knowledge_base dosyasi) karsilastirildi, fark yok.
    # VERBOSE guvenli: desende bosluk eslestirmesinin tamami \s ile yapiliyor.
    _UPPER = "A-ZÇĞİÖŞÜ"  # Turkce buyuk harfler
    _PATTERN = re.compile(
        rf"""
        (                             # 1) urun adi
          [{_UPPER}]\w*               #    ilk kelime buyuk harfle baslar
          (?:\s+[{_UPPER}0-9]\w*)*    #    sonraki kelimeler buyuk harf ya da rakamla
        )
        \s*
        (?:\([^)]*\)\s*)?             # opsiyonel parantezli ek, orn. "(HP uyumlu)"
        genellikle\s+
        (\d+)-(\d+)                   # 2) alt sinir   3) ust sinir
        \s*TL
        """,
        re.VERBOSE,
    )

    def parse(self, retrieved_rules: list[dict | str]) -> dict[str, tuple[float, float]]:
        ranges: dict[str, tuple[float, float]] = {}
        for rule in retrieved_rules:
            text = self._rule_text(rule)
            for name, low, high in self._PATTERN.findall(text):
                ranges[name.strip().casefold()] = (float(low), float(high))
        return ranges

    @staticmethod
    def _rule_text(rule: dict | str) -> str:
        if isinstance(rule, dict):
            text = rule.get("text")
            return text if isinstance(text, str) else ""
        return rule if isinstance(rule, str) else ""


class PriceRangeCheck(Checker):
    """Kalem birim fiyatlarinin bilgi tabanindaki tipik araliklar icinde olup olmadigini kontrol eder.

    Bir kalemin urunu icin aralik cikarilamazsa (ilgili kural getirilmemis ya da metin
    regex'e uymuyorsa) o kalem sessizce atlanir; bu bir hata degildir."""

    RULE_NAME = "price_range"

    def __init__(self, parser: PriceRangeParser | None = None):
        self._parser = parser or PriceRangeParser()

    def check(self, extraction: dict, retrieved_rules: list[dict]) -> list[dict]:
        items = extraction.get("items")
        if not isinstance(items, list) or not retrieved_rules:
            return []

        ranges = self._parser.parse(retrieved_rules)
        if not ranges:
            return []

        anomalies: list[dict] = []
        for index, item in enumerate(items):
            anomalies.extend(self._check_item(index, item, ranges))
        return anomalies

    def _check_item(
        self, index: int, item: object, ranges: dict[str, tuple[float, float]]
    ) -> list[dict]:
        """Tek kalemi denetler. Aralik bulunamayan, aciklamasi/birim fiyati okunamayan
        kalemler sessizce atlanir - bu bir hata degil (bkz. sinif docstring'i)."""
        if not isinstance(item, dict):
            return []
        description = item.get("description")
        if not isinstance(description, str):
            return []
        price_range = ranges.get(description.strip().casefold())
        if price_range is None:
            return []
        unit_price = _to_float(item.get("unit_price"))
        if unit_price is None:
            return []

        low, high = price_range
        if low <= unit_price <= high:
            return []
        return [_anomaly(
            self.RULE_NAME, f"items[{index}].unit_price",
            f"'{description}' için tipik aralık {low:g}-{high:g} TL, bulunan {unit_price}.",
        )]


# "lenient" (Settings.FORMAT_PROFILE'in varsayilani) icin sabit checker seti - ValidationAgent()
# no-arg cagrisi bunu kullanir; "strict_tr" istenirse ValidationAgent format_profile'a gore
# taze bir set kurar (bkz. asagisi).
DEFAULT_CHECKERS: tuple[Checker, ...] = (
    MathConsistencyCheck(),
    VatRateCheck(),
    FormatCheck(profile="lenient"),
    PriceRangeCheck(),
)


class ValidationAgent(BaseAgent):
    """Cikarilan fatura verisini kural bazli kontrol sinifi listesiyle denetler; LLM kullanmaz."""

    def __init__(
        self,
        checkers: tuple[Checker, ...] | list[Checker] | None = None,
        format_profile: FormatProfile = "lenient",
    ):
        if checkers is not None:
            self._checkers = tuple(checkers)
        elif format_profile == "lenient":
            self._checkers = DEFAULT_CHECKERS
        else:
            self._checkers = (MathConsistencyCheck(), VatRateCheck(), FormatCheck(profile=format_profile), PriceRangeCheck())

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
        checklist: list[dict] = []
        for checker in self._checkers:
            checker_anomalies = checker.check(state.raw_extraction, state.retrieved_rules)
            anomalies.extend(checker_anomalies)
            checklist.append({
                "rule": getattr(checker, "RULE_NAME", type(checker).__name__),
                "passed": not checker_anomalies,
                "message": "Sorun tespit edilmedi." if not checker_anomalies
                else "; ".join(a["message"] for a in checker_anomalies),
            })

        return state.model_copy(update={
            "anomalies": anomalies, "is_valid": not anomalies, "validation_checklist": checklist,
        })

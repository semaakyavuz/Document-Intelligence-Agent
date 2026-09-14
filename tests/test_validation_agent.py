"""
ValidationAgent ve her kontrol sinifinin (Checker) kural bazli mantigini test eder.

LLM/ag yok: ValidationAgent tamamen yerel, kural bazli calisir. Her checker icin
ayri ayri (gecerli / matematik hatasi / gecersiz KDV / format hatasi / fiyat araligi
disi) testler + hepsi gecerliyse is_valid=True olan butunsel bir test.
"""

import copy

import pytest

from app.agents.validation_agent import (
    DEFAULT_CHECKERS,
    Checker,
    FormatCheck,
    MathConsistencyCheck,
    PriceRangeCheck,
    PriceRangeParser,
    ValidationAgent,
    VatRateCheck,
)
from app.state import PipelineState

VALID_ITEM = {
    "description": "Klavye",
    "quantity": 2,
    "unit_price": 500.0,
    "vat_rate": 0.20,
    "line_total": 1000.0,
    "vat_amount": 200.0,
}

VALID_EXTRACTION = {
    "invoice_no": "2025123456",
    "invoice_date": "01.01.2025",
    "seller_name": "Test A.S.",
    "seller_tax_no": "1234567890",
    "buyer_name": "Alıcı Ltd.",
    "items": [VALID_ITEM],
    "subtotal": 1000.0,
    "vat_total": 200.0,
    "grand_total": 1200.0,
}

KLAVYE_RULE_TEXT = "Tipik birim fiyat aralığı: Klavye genellikle 300-2500 TL arasındadır."


def _extraction(**overrides) -> dict:
    """VALID_EXTRACTION'in derin kopyasi uzerine ust duzey alan degisiklikleri uygular."""
    data = copy.deepcopy(VALID_EXTRACTION)
    data.update(overrides)
    return data


def _with_item(index: int = 0, **item_overrides) -> dict:
    """VALID_EXTRACTION'in derin kopyasinda items[index]'i gunceller."""
    data = copy.deepcopy(VALID_EXTRACTION)
    data["items"][index] = {**data["items"][index], **item_overrides}
    return data


# --- MathConsistencyCheck ----------------------------------------------------

def test_math_check_passes_for_consistent_invoice():
    assert MathConsistencyCheck().check(VALID_EXTRACTION, []) == []


def test_math_check_detects_line_total_error():
    extraction = _with_item(line_total=999.0)
    anomalies = MathConsistencyCheck().check(extraction, [])
    fields = [a["field"] for a in anomalies]
    assert "items[0].line_total" in fields


def test_math_check_detects_vat_amount_error():
    extraction = _with_item(vat_amount=1.0)
    anomalies = MathConsistencyCheck().check(extraction, [])
    fields = [a["field"] for a in anomalies]
    assert "items[0].vat_amount" in fields


def test_math_check_detects_subtotal_error():
    extraction = _extraction(subtotal=1.0)
    anomalies = MathConsistencyCheck().check(extraction, [])
    fields = [a["field"] for a in anomalies]
    assert "subtotal" in fields


def test_math_check_detects_grand_total_error():
    extraction = _extraction(grand_total=1.0)
    anomalies = MathConsistencyCheck().check(extraction, [])
    fields = [a["field"] for a in anomalies]
    assert "grand_total" in fields


def test_math_check_tolerates_kurus_level_rounding():
    extraction = _with_item(line_total=1000.005)
    assert MathConsistencyCheck().check(extraction, []) == []


def test_math_check_does_not_crash_on_missing_or_malformed_fields():
    extraction = {"items": [{"description": "X"}, "bozuk-kalem", 42], "subtotal": "yok"}
    assert MathConsistencyCheck().check(extraction, []) == []


def test_math_check_skips_subtotal_when_a_line_total_is_unparseable():
    """Bir kalemin line_total'i bilinmiyorsa kismi toplamla yanlis pozitif uretilmemeli."""
    extraction = copy.deepcopy(VALID_EXTRACTION)
    extraction["items"].append({"description": "Mouse", "quantity": 1, "unit_price": 200.0,
                                 "vat_rate": 0.20, "line_total": "okunamadi", "vat_amount": 40.0})
    anomalies = MathConsistencyCheck().check(extraction, [])
    assert all(a["field"] != "subtotal" for a in anomalies)


# --- VatRateCheck -------------------------------------------------------------

@pytest.mark.parametrize("rate", [0.01, 0.10, 0.20])
def test_vat_rate_check_accepts_valid_rates(rate):
    extraction = _with_item(vat_rate=rate)
    assert VatRateCheck().check(extraction, []) == []


def test_vat_rate_check_flags_invalid_rate():
    extraction = _with_item(vat_rate=0.18)
    anomalies = VatRateCheck().check(extraction, [])
    assert len(anomalies) == 1
    assert anomalies[0]["rule"] == "vat_rate"
    assert anomalies[0]["field"] == "items[0].vat_rate"


def test_vat_rate_check_skips_unparseable_rate():
    extraction = _with_item(vat_rate=None)
    assert VatRateCheck().check(extraction, []) == []


# --- FormatCheck ---------------------------------------------------------------

def test_format_check_passes_for_valid_numbers():
    assert FormatCheck().check(VALID_EXTRACTION, []) == []


@pytest.mark.parametrize("invoice_no", ["202512345", "20251234567", "2025-12345", "abc1234567"])
def test_format_check_flags_invalid_invoice_no(invoice_no):
    extraction = _extraction(invoice_no=invoice_no)
    anomalies = FormatCheck().check(extraction, [])
    fields = [a["field"] for a in anomalies]
    assert "invoice_no" in fields


@pytest.mark.parametrize("tax_no", ["123456789", "12345678901", "12345abcd0"])
def test_format_check_flags_invalid_tax_no(tax_no):
    extraction = _extraction(seller_tax_no=tax_no)
    anomalies = FormatCheck().check(extraction, [])
    fields = [a["field"] for a in anomalies]
    assert "seller_tax_no" in fields


def test_format_check_accepts_tax_no_with_leading_zero():
    extraction = _extraction(seller_tax_no="0123456789")
    assert FormatCheck().check(extraction, []) == []


def test_format_check_skips_when_field_missing_or_wrong_type():
    extraction = _extraction(invoice_no=None, seller_tax_no=2025123456)
    assert FormatCheck().check(extraction, []) == []


# --- PriceRangeParser / PriceRangeCheck ----------------------------------------

def test_price_range_parser_extracts_range_from_real_knowledge_base_text():
    ranges = PriceRangeParser().parse([KLAVYE_RULE_TEXT])
    assert ranges["klavye"] == (300.0, 2500.0)


def test_price_range_parser_extracts_multiple_products_from_one_text():
    text = ("Tipik birim fiyat aralıkları: Laptop Standı genellikle 300-1200 TL arasındadır. "
            "Monitör Kolu genellikle 400-1800 TL arasındadır (tekli/çiftli kol farkına göre değişir).")
    ranges = PriceRangeParser().parse([text])
    assert ranges["laptop standı"] == (300.0, 1200.0)
    assert ranges["monitör kolu"] == (400.0, 1800.0)


def test_price_range_check_passes_when_price_within_range():
    assert PriceRangeCheck().check(VALID_EXTRACTION, [KLAVYE_RULE_TEXT]) == []


def test_price_range_check_flags_out_of_range_price():
    extraction = _with_item(unit_price=99999.0)
    anomalies = PriceRangeCheck().check(extraction, [KLAVYE_RULE_TEXT])
    assert len(anomalies) == 1
    assert anomalies[0]["rule"] == "price_range"
    assert anomalies[0]["field"] == "items[0].unit_price"


def test_price_range_check_skips_silently_when_product_not_in_retrieved_rules():
    """Urun icin hic kural getirilmemis: kontrol atlanir, anomali uretilmez."""
    unrelated_rule = "Türkiye'de genel KDV oranı yüzde 20'dir."
    assert PriceRangeCheck().check(VALID_EXTRACTION, [unrelated_rule]) == []


def test_price_range_check_skips_silently_when_no_retrieved_rules():
    assert PriceRangeCheck().check(VALID_EXTRACTION, []) == []


def test_price_range_check_skips_item_with_unparseable_unit_price():
    extraction = _with_item(unit_price="okunamadi")
    assert PriceRangeCheck().check(extraction, [KLAVYE_RULE_TEXT]) == []


# --- ValidationAgent (butunsel) ------------------------------------------------

def test_checker_is_abstract():
    with pytest.raises(TypeError):
        Checker()


def test_agent_uses_default_checkers_by_default():
    agent = ValidationAgent()
    assert agent._checkers == DEFAULT_CHECKERS


def test_agent_reports_valid_when_extraction_is_fully_consistent():
    state = PipelineState(
        image_path="data/golden/images/invoice_0001.png",
        raw_extraction=VALID_EXTRACTION,
        retrieved_rules=[KLAVYE_RULE_TEXT],
    )
    result = ValidationAgent().run(state)
    assert result.anomalies == []
    assert result.is_valid is True


def test_agent_aggregates_anomalies_from_multiple_checkers():
    bad_extraction = _extraction(invoice_no="bozuk", subtotal=1.0)
    bad_extraction["items"][0] = {**bad_extraction["items"][0], "vat_rate": 0.18}
    state = PipelineState(image_path="x.png", raw_extraction=bad_extraction, retrieved_rules=[])
    result = ValidationAgent().run(state)

    rules_found = {a["rule"] for a in result.anomalies}
    assert {"format", "math_consistency", "vat_rate"} <= rules_found
    assert result.is_valid is False


def test_agent_sets_invalid_with_single_anomaly_when_raw_extraction_is_missing():
    state = PipelineState(image_path="x.png", raw_extraction=None)
    result = ValidationAgent().run(state)

    assert result.is_valid is False
    assert result.anomalies == [{
        "rule": "missing_extraction",
        "field": None,
        "message": "Vision Agent veri çıkaramadığı için doğrulama yapılamadı.",
    }]


def test_agent_does_not_mutate_input_state():
    original = PipelineState(
        image_path="x.png", raw_extraction=VALID_EXTRACTION, retrieved_rules=[KLAVYE_RULE_TEXT],
    )
    ValidationAgent().run(original)
    assert original.anomalies == []
    assert original.is_valid is True


def test_agent_accepts_custom_checker_list():
    calls: list[tuple[dict, list]] = []

    class RecordingChecker(Checker):
        def check(self, extraction, retrieved_rules):
            calls.append((extraction, retrieved_rules))
            return [{"rule": "custom", "field": None, "message": "test"}]

    state = PipelineState(image_path="x.png", raw_extraction=VALID_EXTRACTION)
    result = ValidationAgent(checkers=[RecordingChecker()]).run(state)

    assert len(calls) == 1
    assert result.anomalies == [{"rule": "custom", "field": None, "message": "test"}]
    assert result.is_valid is False

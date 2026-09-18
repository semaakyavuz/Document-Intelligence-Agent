"""
scripts/convert_kaggle_labels.py'nin sayi parse mantigini ve Kaggle JSON -> golden-set
etiket eslemesini test eder. Gercek CSV/dosya sistemi yok, saf fonksiyonlar dogrudan
cagrilir.
"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_kaggle_labels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_kaggle_labels", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def convert_module():
    return _load_module()


# --- _parse_number: bosluk=binlik, virgul/nokta=ondalik (canlica dogrulanan kural) ------

@pytest.mark.parametrize("raw, expected", [
    ("46.55", 46.55),
    ("14,82", 14.82),
    ("34 320.00", 34320.00),
    ("3 135,10", 3135.10),
    ("1 369,50", 1369.50),
    ("232.95", 232.95),
    ("0.00", 0.0),
])
def test_parse_number_handles_all_observed_formats(convert_module, raw, expected):
    assert convert_module._parse_number(raw) == pytest.approx(expected)


def test_parse_number_raises_on_blank(convert_module):
    with pytest.raises(convert_module.ConversionError):
        convert_module._parse_number("")


def test_parse_number_raises_on_unparseable(convert_module):
    with pytest.raises(convert_module.ConversionError):
        convert_module._parse_number("N/A")


# --- convert_record: alan eslemesi ------------------------------------------------

SAMPLE_JSON = json.dumps({
    "invoice": {
        "client_name": "Clark-Foster",
        "seller_name": "Nguyen-Roach",
        "invoice_number": "84652373",
        "invoice_date": "02/23/2021",
    },
    "items": [
        {"description": "Stemware Rack", "quantity": "1.00", "total_price": "46.55"},
        {"description": "Wine Carafe", "quantity": "2.00", "total_price": "15.40"},
    ],
    "subtotal": {"tax": "21.18", "discount": "", "total": "232.95"},
})


def test_convert_record_maps_fields_correctly(convert_module):
    label = convert_module.convert_record(SAMPLE_JSON)

    assert label["invoice_no"] == "84652373"
    assert label["invoice_date"] == "02/23/2021"
    assert label["seller_name"] == "Nguyen-Roach"
    assert label["buyer_name"] == "Clark-Foster"
    assert "seller_tax_no" not in label


def test_convert_record_grand_total_comes_directly_from_source_not_computed(convert_module):
    """Kaggle'in subtotal.total'i zaten KDV dahil nihai toplam (canlica dogrulandi) -
    grand_total dogrudan buradan gelir, subtotal ise KDV cikarilarak HESAPLANIR."""
    label = convert_module.convert_record(SAMPLE_JSON)

    assert label["grand_total"] == 232.95
    assert label["grand_total_computed"] is False
    assert label["vat_total"] == 21.18
    assert label["subtotal"] == pytest.approx(232.95 - 21.18)
    assert label["subtotal_computed"] is True


def test_convert_record_items_are_gross_and_missing_fields_are_none(convert_module):
    label = convert_module.convert_record(SAMPLE_JSON)

    assert label["items_gross"] is True
    assert label["items"] == [
        {"description": "Stemware Rack", "quantity": 1.0, "unit_price": None, "vat_rate": None,
         "line_total": 46.55, "vat_amount": None},
        {"description": "Wine Carafe", "quantity": 2.0, "unit_price": None, "vat_rate": None,
         "line_total": 15.40, "vat_amount": None},
    ]


def test_convert_record_raises_on_invalid_json(convert_module):
    with pytest.raises(convert_module.ConversionError, match="JSON"):
        convert_module.convert_record("{bozuk")


def test_convert_record_raises_on_empty_items(convert_module):
    data = json.loads(SAMPLE_JSON)
    data["items"] = []
    with pytest.raises(convert_module.ConversionError, match="items"):
        convert_module.convert_record(json.dumps(data))


def test_convert_record_raises_on_unparseable_subtotal(convert_module):
    data = json.loads(SAMPLE_JSON)
    data["subtotal"]["total"] = "yok"
    with pytest.raises(convert_module.ConversionError):
        convert_module.convert_record(json.dumps(data))


# --- convert_all: dosya sistemi entegrasyonu -----------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["File Name", "Json Data", "OCRed Text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(file_name: str, json_data: str = SAMPLE_JSON) -> dict:
    return {"File Name": file_name, "Json Data": json_data, "OCRed Text": ""}


def test_convert_all_creates_labels_only_for_images_present(convert_module, tmp_path):
    _write_csv(tmp_path / "batch1_1.csv", [_row("a.jpg"), _row("b.jpg")])
    (tmp_path / "a.jpg").write_bytes(b"fake")  # b.jpg yok

    summary = convert_module.convert_all(tmp_path, ("batch1_1.csv",))

    assert summary.converted == 1
    assert summary.skipped_no_image == 1
    assert summary.errors == []
    assert (tmp_path / "images" / "a.jpg").is_file()
    assert (tmp_path / "labels" / "a.json").is_file()
    assert not (tmp_path / "labels" / "b.json").exists()


def test_convert_all_does_not_rewrite_already_labeled_images(convert_module, tmp_path):
    _write_csv(tmp_path / "batch1_1.csv", [_row("a.jpg")])
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"fake")
    (tmp_path / "labels" / "a.json").write_text('{"sentinel": true}', encoding="utf-8")

    summary = convert_module.convert_all(tmp_path, ("batch1_1.csv",))

    assert summary.converted == 0
    assert summary.already_labeled == 1
    assert json.loads((tmp_path / "labels" / "a.json").read_text(encoding="utf-8")) == {"sentinel": True}


def test_convert_all_reports_images_with_no_csv_match(convert_module, tmp_path):
    _write_csv(tmp_path / "batch1_1.csv", [_row("a.jpg")])
    (tmp_path / "images").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"fake")
    # CSV'de hic gecmeyen, "yetim" bir gorsel - dogrudan images/ altina konmus.
    (tmp_path / "images" / "orphan.jpg").write_bytes(b"fake")

    summary = convert_module.convert_all(tmp_path, ("batch1_1.csv",))

    assert summary.converted == 1
    assert summary.no_csv_match == ["orphan.jpg"]


def test_convert_all_moves_unconvertable_image_out_of_images_dir(convert_module, tmp_path):
    """images/ altinda olup donusumu basarisiz olan bir gorsel, images/'ten disari
    tasinir (etiketi olmadigi icin orada birakilmamali) ve no_csv_match'e girmez."""
    broken = json.loads(SAMPLE_JSON)
    broken["subtotal"]["tax"] = "10%"
    _write_csv(tmp_path / "batch1_1.csv", [_row("a.jpg", json.dumps(broken))])
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"fake")

    summary = convert_module.convert_all(tmp_path, ("batch1_1.csv",))

    assert summary.converted == 0
    assert len(summary.errors) == 1
    assert summary.errors[0][0] == "a.jpg"
    assert not (tmp_path / "images" / "a.jpg").exists()
    assert (tmp_path / "a.jpg").is_file()  # disari tasindi
    assert summary.no_csv_match == []  # artik images/ icinde degil, yetim sayilmaz


def test_convert_all_is_idempotent_across_two_runs(convert_module, tmp_path):
    _write_csv(tmp_path / "batch1_1.csv", [_row("a.jpg")])
    (tmp_path / "a.jpg").write_bytes(b"fake")

    first = convert_module.convert_all(tmp_path, ("batch1_1.csv",))
    second = convert_module.convert_all(tmp_path, ("batch1_1.csv",))

    assert first.converted == 1
    assert second.converted == 0
    assert second.already_labeled == 1

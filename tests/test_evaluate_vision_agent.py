"""
evaluate_vision_agent.py'nin karsilastirma ve hata toleransi mantigini test eder.

Gercek Ollama yoktur: sahte provider'lar ya hazir JSON dondurur ya da
ProviderTimeoutError / ProviderUnavailableError firlatir.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from app.agents.vision_agent import VisionAgent
from app.providers.base import LLMProvider, ProviderTimeoutError, ProviderUnavailableError

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_vision_agent.py"

LABEL = {
    "invoice_no": "2025864544",
    "invoice_date": "14.12.2025",
    "seller_name": "Çalık Enerji",
    "seller_tax_no": "4835030564",
    "buyer_name": "Çorlu Ltd.",
    "items": [{"description": "Klavye", "quantity": 2, "unit_price": 2145.53, "vat_rate": 0.01, "line_total": 4291.06, "vat_amount": 42.91}],
    "subtotal": 4291.06,
    "vat_total": 42.91,
    "grand_total": 4333.97,
}


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_vision_agent", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate_module():
    return _load_module()


def _make_golden_dir(tmp_path: Path, stems: list[str]) -> Path:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for stem in stems:
        (tmp_path / "images" / f"{stem}.png").write_bytes(b"\x89PNG fake")
        (tmp_path / "labels" / f"{stem}.json").write_text(json.dumps(LABEL, ensure_ascii=False), encoding="utf-8")
    return tmp_path


class ScriptedProvider(LLMProvider):
    """Her cagrida sirayla bir senaryo uygular: str ise cevap, Exception ise firlatir."""

    def __init__(self, scenarios: list):
        self._scenarios = list(scenarios)

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        scenario = self._scenarios.pop(0)
        if isinstance(scenario, Exception):
            raise scenario
        return scenario


# --- FieldComparator -------------------------------------------------------

@pytest.mark.parametrize(
    "expected, found, is_equal",
    [
        (20789.03, 20789.031, True),      # kurus alti fark tolere edilir
        (20789.03, "20789,03", True),     # virgullu string sayi olarak okunur
        (20789.03, 20790.0, False),
        (2, 2.0, True),
        ("Çalık Enerji", "  Çalık   Enerji ", True),  # bosluklar sadelestirilir
        ("Çalık Enerji", "Calik Enerji", False),      # Turkce karakter farki hata sayilir
        ("2025864544", 2025864544, False),            # tip farki: string beklenirken sayi
        (None, None, True),
    ],
)
def test_field_comparator_scalars(evaluate_module, expected, found, is_equal):
    assert evaluate_module.FieldComparator().equal(expected, found) is is_equal


@pytest.mark.parametrize(
    "expected, found, is_equal",
    [
        ("13.04.2013", "13.04.2013", True),     # ayni format
        ("04/13/2013", "13.04.2013", True),     # MM/DD/YYYY (Kaggle) vs DD.MM.YYYY (Vision)
        ("02/23/2021", "23.02.2021", True),
        ("13.04.2013", "14.04.2013", False),    # gercekten farkli tarih
        ("01/02/2020", "01.02.2020", True),     # belirsiz durum: herhangi bir yorum kesisirse dogru
    ],
)
def test_field_comparator_equal_date_ignores_format(evaluate_module, expected, found, is_equal):
    assert evaluate_module.FieldComparator().equal_date(expected, found) is is_equal


@pytest.mark.parametrize("expected, found", [
    ("", "13.04.2013"), ("13.04.2013", ""), ("bozuk tarih", "13.04.2013"), (None, "13.04.2013"),
])
def test_field_comparator_equal_date_returns_none_when_unparseable(evaluate_module, expected, found):
    assert evaluate_module.FieldComparator().equal_date(expected, found) is None


def test_field_comparator_items(evaluate_module):
    comparator = evaluate_module.FieldComparator()
    items = LABEL["items"]
    assert comparator.equal(items, json.loads(json.dumps(items)))
    assert not comparator.equal(items, [])                                # satir sayisi farkli
    assert not comparator.equal(items, [{**items[0], "quantity": 3}])     # alt alan farkli
    assert comparator.equal(items, [{**items[0], "extra": "ignored"}])    # fazladan alan zarar vermez


def test_field_comparator_skips_unknown_item_fields_when_none(evaluate_module):
    """data/real_world_test (Kaggle) etiketlerinde unit_price/vat_rate/vat_amount None -
    bu alanlar Vision Agent'in bulduguyla eslesmese bile items yine dogru sayilmali."""
    comparator = evaluate_module.FieldComparator()
    expected_items = [{
        "description": "Klavye", "quantity": 2, "unit_price": None, "vat_rate": None,
        "line_total": 4291.06, "vat_amount": None,
    }]
    found_items = [{
        "description": "Klavye", "quantity": 2, "unit_price": 2145.53, "vat_rate": 0.01,
        "line_total": 4291.06, "vat_amount": 42.91,
    }]
    assert comparator.equal(expected_items, found_items)


def test_field_comparator_still_flags_line_total_mismatch_when_other_fields_unknown(evaluate_module):
    comparator = evaluate_module.FieldComparator()
    expected_items = [{"description": "Klavye", "quantity": 2, "unit_price": None, "vat_rate": None,
                        "line_total": 4291.06, "vat_amount": None}]
    found_items = [{"description": "Klavye", "quantity": 2, "unit_price": 1.0, "vat_rate": 0.5,
                     "line_total": 1.0, "vat_amount": 1.0}]
    assert not comparator.equal(expected_items, found_items)


def test_gross_adjusted_items_adds_vat_amount_to_line_total(evaluate_module):
    """items_gross=true olan etiketlerde (Kaggle) Vision'in net line_total'i, yine
    Vision'in kendi cikardigi vat_amount ile toplanip gross'a cevrilmeli."""
    found_items = [{"description": "Klavye", "quantity": 2, "unit_price": 100.0, "vat_rate": 0.1,
                     "line_total": 200.0, "vat_amount": 20.0}]
    adjusted = evaluate_module._gross_adjusted_items(found_items)
    assert adjusted[0]["line_total"] == 220.0
    assert found_items[0]["line_total"] == 200.0  # orijinal degismedi


def test_gross_adjusted_items_leaves_line_total_when_vat_amount_unparseable(evaluate_module):
    found_items = [{"description": "Klavye", "quantity": 2, "line_total": 200.0, "vat_amount": None}]
    adjusted = evaluate_module._gross_adjusted_items(found_items)
    assert adjusted[0]["line_total"] == 200.0


# --- Evaluator -------------------------------------------------------------

def test_evaluator_scores_correct_and_wrong_fields(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a", "b"])
    wrong_total = {**LABEL, "grand_total": 1.0}
    provider = ScriptedProvider([json.dumps(LABEL), json.dumps(wrong_total)])

    samples = evaluate_module.GoldenDataset(golden).samples()
    report = evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    assert report.accuracy("invoice_no") == 1.0
    assert report.accuracy("grand_total") == 0.5
    assert report.mismatches == [{"image": "b", "field": "grand_total", "expected": 4333.97, "found": 1.0}]
    assert report.parse_failures == [] and report.provider_errors == []


def test_evaluator_continues_after_timeout_and_parse_failure(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a", "b", "c"])
    provider = ScriptedProvider([ProviderTimeoutError("600 sn"), "bozuk {{{", json.dumps(LABEL)])

    samples = evaluate_module.GoldenDataset(golden).samples()
    report = evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    assert [e["image"] for e in report.provider_errors] == ["a"]
    assert [p["image"] for p in report.parse_failures] == ["b"]
    assert report.total["grand_total"] == 3           # hatali ornekler de toplam icinde
    assert report.correct["grand_total"] == 1         # yalnizca c dogru
    assert report.accuracy("grand_total") == pytest.approx(1 / 3)


def test_evaluator_adjusts_items_for_gross_ground_truth(evaluate_module, tmp_path):
    """items_gross=true olan bir etikette, Vision'in net line_total + kendi vat_amount'u
    ground truth'un gross line_total'ina esitse items dogru sayilmali."""
    gross_label = {
        **LABEL,
        "items": [{"description": "Klavye", "quantity": 2, "unit_price": None, "vat_rate": None,
                    "line_total": 220.0, "vat_amount": None}],
        "items_gross": True,
    }
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "labels" / "a.json").write_text(json.dumps(gross_label, ensure_ascii=False), encoding="utf-8")

    found_extraction = {
        **LABEL,
        "items": [{"description": "Klavye", "quantity": 2, "unit_price": 100.0, "vat_rate": 0.1,
                    "line_total": 200.0, "vat_amount": 20.0}],
    }
    provider = ScriptedProvider([json.dumps(found_extraction)])

    samples = evaluate_module.GoldenDataset(tmp_path).samples()
    report = evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    assert report.accuracy("items") == 1.0
    assert report.mismatches == []


def test_evaluator_pauses_between_real_provider_calls_when_configured(evaluate_module, tmp_path, monkeypatch):
    """--pause-s rate limit icin: her gercek saglayici cagrisindan sonra beklenmeli."""
    sleep_calls = []
    monkeypatch.setattr(evaluate_module.time, "sleep", lambda s: sleep_calls.append(s))

    golden = _make_golden_dir(tmp_path, ["a", "b"])
    provider = ScriptedProvider([json.dumps(LABEL), json.dumps(LABEL)])
    samples = evaluate_module.GoldenDataset(golden).samples()

    evaluate_module.VisionAgentEvaluator(VisionAgent(provider), pause_s=4.5).evaluate(samples)

    assert sleep_calls == [4.5, 4.5]


def test_evaluator_does_not_pause_by_default(evaluate_module, tmp_path, monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(evaluate_module.time, "sleep", lambda s: sleep_calls.append(s))

    golden = _make_golden_dir(tmp_path, ["a"])
    provider = ScriptedProvider([json.dumps(LABEL)])
    samples = evaluate_module.GoldenDataset(golden).samples()

    evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    assert sleep_calls == []


def test_evaluator_aborts_when_provider_unavailable(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a", "b"])
    provider = ScriptedProvider([ProviderUnavailableError("ollama serve"), json.dumps(LABEL)])

    samples = evaluate_module.GoldenDataset(golden).samples()
    with pytest.raises(ProviderUnavailableError):
        evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)


def test_report_json_is_written_utf8(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a"])
    provider = ScriptedProvider([json.dumps({**LABEL, "seller_name": "Yanlış Şirket"})])
    samples = evaluate_module.GoldenDataset(golden).samples()
    report = evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    out = tmp_path / "report" / "eval_report.json"
    report.write_json(out, evaluate_module.REPORTED_FIELDS, model_name="fake")
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["model"] == "fake"
    assert set(data) == {
        "model", "overall_accuracy", "accuracy", "totals", "not_applicable",
        "parse_failures", "provider_errors", "label_errors", "mismatches",
    }
    assert data["accuracy"]["seller_name"] == 0.0 and data["accuracy"]["grand_total"] == 1.0
    assert data["mismatches"][0]["found"] == "Yanlış Şirket"  # Turkce karakterler bozulmadan yazildi


def test_missing_label_file_is_skipped_not_raised(evaluate_module, tmp_path):
    """Etiket dosyasinin kendisi yoksa (isim eslesmiyor): hata firlatilmaz, o gorsel
    sessizce atlanir ve skipped_unlabeled sayacina eklenir (bkz. convert_kaggle_labels.py'nin
    gorselleri etiketlerinden ayri/once ekleyebildigi senaryo)."""
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "orphan.png").write_bytes(b"x")

    dataset = evaluate_module.GoldenDataset(tmp_path)
    samples = dataset.samples()

    assert samples == []
    assert dataset.skipped_unlabeled == 1


def test_samples_processes_labeled_and_skips_unlabeled_in_a_mixed_set(evaluate_module, tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for stem in ("a", "b", "orphan1", "orphan2", "orphan3"):
        (tmp_path / "images" / f"{stem}.png").write_bytes(b"\x89PNG fake")
    for stem in ("a", "b"):
        (tmp_path / "labels" / f"{stem}.json").write_text(json.dumps(LABEL), encoding="utf-8")

    dataset = evaluate_module.GoldenDataset(tmp_path)
    samples = dataset.samples()

    assert [s.stem for s in samples] == ["a", "b"]
    assert dataset.skipped_unlabeled == 3


def test_samples_recomputes_skip_count_on_each_call(evaluate_module, tmp_path):
    """skipped_unlabeled her samples() cagrisinda sifirlanip yeniden hesaplanir (birikmez)."""
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "orphan.png").write_bytes(b"x")

    dataset = evaluate_module.GoldenDataset(tmp_path)
    dataset.samples()
    dataset.samples()

    assert dataset.skipped_unlabeled == 1


# --- Bozuk etiket toleransi -------------------------------------------------

def test_invalid_json_label_raises_label_error(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a"])
    (golden / "labels" / "a.json").write_text("{bozuk json", encoding="utf-8")
    sample = evaluate_module.GoldenDataset(golden).samples()[0]
    with pytest.raises(evaluate_module.LabelError, match="a.json"):
        sample.load_label()


def test_label_missing_seller_tax_no_is_allowed(evaluate_module, tmp_path):
    """seller_tax_no FIELDS_MAY_BE_UNKNOWN'da - anahtarin kendisi hic yoksa bile etiket gecerli
    sayilmali (bkz. scripts/convert_kaggle_labels.py'nin bu alani hic eklememesi)."""
    golden = _make_golden_dir(tmp_path, ["a"])
    without_tax_no = {k: v for k, v in LABEL.items() if k != "seller_tax_no"}
    (golden / "labels" / "a.json").write_text(json.dumps(without_tax_no, ensure_ascii=False), encoding="utf-8")
    sample = evaluate_module.GoldenDataset(golden).samples()[0]
    loaded = sample.load_label()
    assert "seller_tax_no" not in loaded


def test_label_missing_required_key_raises_label_error(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a"])
    incomplete = {k: v for k, v in LABEL.items() if k != "grand_total"}
    (golden / "labels" / "a.json").write_text(json.dumps(incomplete), encoding="utf-8")
    sample = evaluate_module.GoldenDataset(golden).samples()[0]
    with pytest.raises(evaluate_module.LabelError, match="grand_total"):
        sample.load_label()


def test_evaluator_skips_bad_label_without_calling_llm_and_without_crashing(evaluate_module, tmp_path):
    """Bozuk etiket modelin cagrilmasini engeller, toplamlara girmez, kosuyu cokertmez;
    digerleri normal degerlendirilmeye devam eder."""
    golden = _make_golden_dir(tmp_path, ["a", "bad", "c"])
    (golden / "labels" / "bad.json").write_text("{bozuk json", encoding="utf-8")
    provider = ScriptedProvider([json.dumps(LABEL), json.dumps(LABEL)])  # yalnizca a ve c icin

    samples = evaluate_module.GoldenDataset(golden).samples()
    report = evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    assert [e["image"] for e in report.label_errors] == ["bad"]
    assert report.total["grand_total"] == 2               # bad, toplamlara girmedi
    assert report.correct["grand_total"] == 2
    assert report.parse_failures == [] and report.provider_errors == []


def test_evaluator_marks_missing_top_level_field_as_not_applicable(evaluate_module, tmp_path):
    """seller_tax_no etikette hic yoksa: totals/correct'a girmez, not_applicable sayacina girer,
    fields sozlugunde None olarak gorunur - diger alanlar normal puanlanmaya devam eder."""
    golden = _make_golden_dir(tmp_path, ["a"])
    without_tax_no = {k: v for k, v in LABEL.items() if k != "seller_tax_no"}
    (golden / "labels" / "a.json").write_text(json.dumps(without_tax_no, ensure_ascii=False), encoding="utf-8")
    provider = ScriptedProvider([json.dumps(LABEL)])

    samples = evaluate_module.GoldenDataset(golden).samples()
    report = evaluate_module.VisionAgentEvaluator(VisionAgent(provider)).evaluate(samples)

    assert report.not_applicable["seller_tax_no"] == 1
    assert "seller_tax_no" not in report.total
    assert report.total["grand_total"] == 1
    assert report.correct["grand_total"] == 1


# --- JSONL aninda yazma ------------------------------------------------------

def test_jsonl_writer_persists_before_context_closes(evaluate_module, tmp_path):
    """write() sonrasi flush edilir: dosya henuz 'with' blogu icindeyken bile
    ayri bir okuyucu tam satiri gorebilmeli (kosu ortasinda cokerse veri kaybolmaz)."""
    path = tmp_path / "results.jsonl"
    with evaluate_module.JsonlResultWriter(path) as writer:
        writer.write({"image": "a", "status": "ok"})
        persisted = path.read_text(encoding="utf-8")  # ayri okuma, kapanmadan once
    assert json.loads(persisted.strip()) == {"image": "a", "status": "ok"}


def test_jsonl_writer_requires_context_manager(evaluate_module, tmp_path):
    writer = evaluate_module.JsonlResultWriter(tmp_path / "results.jsonl")
    with pytest.raises(RuntimeError):
        writer.write({"image": "a"})


def test_evaluate_writes_one_jsonl_line_per_sample_with_status(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["ok_img", "bad_parse", "timeout_img"])
    # GoldenDataset.samples() dosyalari alfabetik sirayla dondurur: bad_parse, ok_img, timeout_img.
    provider = ScriptedProvider(["bozuk {{{", json.dumps(LABEL), ProviderTimeoutError("600 sn")])
    samples = evaluate_module.GoldenDataset(golden).samples()

    results_path = tmp_path / "results.jsonl"
    with evaluate_module.JsonlResultWriter(results_path) as writer:
        evaluate_module.VisionAgentEvaluator(VisionAgent(provider), result_writer=writer).evaluate(samples)

    lines = {json.loads(line)["image"]: json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()}
    assert {img: rec["status"] for img, rec in lines.items()} == {
        "ok_img": "ok", "bad_parse": "parse_failure", "timeout_img": "provider_error",
    }
    assert lines["ok_img"]["extraction"]["invoice_no"] == "2025864544"
    assert lines["ok_img"]["fields"]["grand_total"] is True
    assert lines["bad_parse"]["extraction"] is None
    assert lines["timeout_img"]["extraction"] is None


# --- --resume: onceki kosudan devam ---------------------------------------------

def test_load_previous_results_keeps_latest_record_per_image(evaluate_module, tmp_path):
    """Ayni gorsel icin JSONL'de birden fazla kayit varsa (iki kez resume edilmis gibi),
    en son (dosyadaki son) kayit gecerli olmali."""
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"image": "a", "status": "provider_error"}) + "\n"
        + json.dumps({"image": "a", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    records = evaluate_module._load_previous_results(path)
    assert records["a"]["status"] == "ok"


def test_load_previous_results_returns_empty_when_file_missing(evaluate_module, tmp_path):
    assert evaluate_module._load_previous_results(tmp_path / "yok.jsonl") == {}


def test_replay_done_result_populates_report_like_a_fresh_ok_run(evaluate_module):
    report = evaluate_module.EvaluationReport()
    record = {
        "image": "a", "status": "ok", "duration_s": 2.5,
        "fields": {name: True for name in evaluate_module.REPORTED_FIELDS},
        "extraction": LABEL, "errors": [],
    }
    evaluate_module._replay_done_result(report, record, LABEL)

    assert report.total["grand_total"] == 1
    assert report.correct["grand_total"] == 1
    assert report.durations_s == [2.5]
    assert report.mismatches == []


def test_replay_done_result_records_mismatch_for_wrong_field(evaluate_module):
    report = evaluate_module.EvaluationReport()
    fields = {name: True for name in evaluate_module.REPORTED_FIELDS}
    fields["grand_total"] = False
    record = {"image": "a", "status": "ok", "duration_s": 1.0, "fields": fields,
              "extraction": {**LABEL, "grand_total": 1.0}, "errors": []}
    evaluate_module._replay_done_result(report, record, LABEL)

    assert report.correct.get("grand_total", 0) == 0
    assert report.total["grand_total"] == 1
    assert report.mismatches == [{"image": "a", "field": "grand_total", "expected": 4333.97, "found": 1.0}]


def test_replay_done_result_handles_parse_failure_and_label_error(evaluate_module):
    report = evaluate_module.EvaluationReport()
    evaluate_module._replay_done_result(report, {
        "image": "b", "status": "parse_failure", "duration_s": 1.2,
        "fields": {name: False for name in evaluate_module.REPORTED_FIELDS},
        "extraction": None, "errors": ["bozuk cevap"],
    }, LABEL)
    evaluate_module._replay_done_result(report, {
        "image": "c", "status": "label_error", "duration_s": None,
        "fields": {}, "extraction": None, "errors": ["etiket bozuk"],
    }, LABEL)

    assert [p["image"] for p in report.parse_failures] == ["b"]
    assert [e["image"] for e in report.label_errors] == ["c"]


def test_resume_skips_done_retries_provider_error_and_merges_final_report(evaluate_module, tmp_path):
    """Uctan uca: ilk kosu (a basarili, b provider_error) -> resume kosusu (sadece b ve
    hic islenmemis c calisir, a'ya hic dokunulmaz) -> nihai rapor ucunu de icerir."""
    golden = _make_golden_dir(tmp_path, ["a", "b"])
    # c'yi ilk kosudan SONRA ekliyoruz (ilk kosude hic yoktu - "hic islenmemis" senaryosu).
    results_path = tmp_path / "results.jsonl"

    first_provider = ScriptedProvider([json.dumps(LABEL), ProviderTimeoutError("600 sn")])
    samples_v1 = evaluate_module.GoldenDataset(golden).samples()
    with evaluate_module.JsonlResultWriter(results_path) as writer:
        evaluate_module.VisionAgentEvaluator(VisionAgent(first_provider), result_writer=writer).evaluate(samples_v1)

    (golden / "images" / "c.png").write_bytes(b"\x89PNG fake")
    (golden / "labels" / "c.json").write_text(json.dumps(LABEL, ensure_ascii=False), encoding="utf-8")

    dataset = evaluate_module.GoldenDataset(golden)
    all_samples = dataset.samples()
    previous = evaluate_module._load_previous_results(results_path)
    samples_by_stem = {s.stem: s for s in all_samples}
    done_stems = {stem for stem, rec in previous.items() if rec.get("status") != "provider_error"}

    report = evaluate_module.EvaluationReport()
    for stem in done_stems:
        evaluate_module._replay_done_result(report, previous[stem], samples_by_stem[stem].load_label())
    samples_to_run = [s for s in all_samples if s.stem not in done_stems]

    assert done_stems == {"a"}
    assert [s.stem for s in samples_to_run] == ["b", "c"]

    second_provider = ScriptedProvider([json.dumps(LABEL), json.dumps(LABEL)])
    with evaluate_module.JsonlResultWriter(results_path, append=True) as writer:
        evaluate_module.VisionAgentEvaluator(VisionAgent(second_provider), result_writer=writer).evaluate(
            samples_to_run, report=report,
        )

    assert report.total["grand_total"] == 3
    assert report.correct["grand_total"] == 3

    lines = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert [rec["image"] for rec in lines] == ["a", "b", "b", "c"]  # eski b (hatali) SATIR OLARAK korunur, silinmez
    assert lines[-2]["status"] == "ok"  # b'nin yeniden denemesi basarili


# --- CLI dogrulama ------------------------------------------------------------

@pytest.mark.parametrize("value", ["0", "-1", "-5"])
def test_limit_rejects_non_positive_values(evaluate_module, value):
    parser = evaluate_module.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--limit", value])


def test_limit_accepts_positive_value(evaluate_module):
    parser = evaluate_module.build_arg_parser()
    args = parser.parse_args(["--limit", "3"])
    assert args.limit == 3

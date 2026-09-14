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


def test_field_comparator_items(evaluate_module):
    comparator = evaluate_module.FieldComparator()
    items = LABEL["items"]
    assert comparator.equal(items, json.loads(json.dumps(items)))
    assert not comparator.equal(items, [])                                # satir sayisi farkli
    assert not comparator.equal(items, [{**items[0], "quantity": 3}])     # alt alan farkli
    assert comparator.equal(items, [{**items[0], "extra": "ignored"}])    # fazladan alan zarar vermez


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
        "model", "overall_accuracy", "accuracy", "totals",
        "parse_failures", "provider_errors", "label_errors", "mismatches",
    }
    assert data["accuracy"]["seller_name"] == 0.0 and data["accuracy"]["grand_total"] == 1.0
    assert data["mismatches"][0]["found"] == "Yanlış Şirket"  # Turkce karakterler bozulmadan yazildi


def test_missing_label_file_fails_dataset_construction(evaluate_module, tmp_path):
    """Etiket dosyasinin kendisi yoksa (isim eslesmiyor) veri kumesi hic kurulmaz."""
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "orphan.png").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="orphan"):
        evaluate_module.GoldenDataset(tmp_path).samples()


# --- Bozuk etiket toleransi -------------------------------------------------

def test_invalid_json_label_raises_label_error(evaluate_module, tmp_path):
    golden = _make_golden_dir(tmp_path, ["a"])
    (golden / "labels" / "a.json").write_text("{bozuk json", encoding="utf-8")
    sample = evaluate_module.GoldenDataset(golden).samples()[0]
    with pytest.raises(evaluate_module.LabelError, match="a.json"):
        sample.load_label()


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

"""
evaluate_vision_agent.py

Golden set uzerinde VisionAgent'i calistirir ve ground-truth ile alan alan karsilastirir.

Kullanim:
    python scripts/evaluate_vision_agent.py
    python scripts/evaluate_vision_agent.py --limit 3          # hizli deneme
    python scripts/evaluate_vision_agent.py --golden-dir data/golden --report-path data/golden/eval_report.json

Ciktilar:
    - terminal: alan bazli dogruluk tablosu
    - --results-path (JSONL): her gorselin sonucu, degerlendirilir degerlendirilmez bir satir olarak
      eklenir; kosu ortada coksa bile onceki satirlar kaybolmaz
    - --report-path (JSON): kosu sonunda ozet, yanlislar, parse/saglayici/etiket hatalari

Provider .env'den (LLM_PROVIDER, OLLAMA_VISION_MODEL) gelir; Ollama'nin calisiyor olmasi gerekir.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dateutil import parser as date_parser

# scripts/ klasorunden calistirildiginda app paketinin bulunabilmesi icin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.vision_agent import VisionAgent  # noqa: E402
from app.providers.base import ProviderError, ProviderUnavailableError  # noqa: E402
from app.providers.factory import get_llm_provider  # noqa: E402
from app.state import PipelineState  # noqa: E402

SCALAR_FIELDS = (
    "invoice_no",
    "invoice_date",
    "seller_name",
    "seller_tax_no",
    "buyer_name",
    "subtotal",
    "vat_total",
    "grand_total",
)
REPORTED_FIELDS = SCALAR_FIELDS + ("items",)
NUMERIC_TOLERANCE = 0.01  # kurus farki yuvarlama sayilir

# Bazi golden setlerde (ozellikle data/real_world_test - Kaggle'dan donusturulmus etiketler,
# bkz. scripts/convert_kaggle_labels.py) bu alanlar kaynak veride hic yok. seller_tax_no ust
# seviyede etiketten TAMAMEN CIKARILIR (anahtar bile yok); items[].unit_price/vat_rate/vat_amount
# ise her kalemde None olarak birakilir (bkz. donusum script'i). Boyle alanlar karsilastirmaya
# hic girmez, "N/A" olarak ayrica sayilir - golden_mixed/golden_classic'te bu alanlar hep dolu
# oldugu icin (sentetik veri hicbir zaman None/eksik birakmiyor) davranislari degismez.
FIELDS_MAY_BE_UNKNOWN = frozenset({"seller_tax_no", "unit_price", "vat_rate", "vat_amount"})


class LabelError(ValueError):
    """Ground-truth etiketi okunamadi ya da beklenen semaya uymuyor."""


@dataclass(frozen=True)
class GoldenSample:
    """Bir golden gorsel ve onun ground-truth etiketi."""

    stem: str
    image_path: Path
    label_path: Path

    def load_label(self) -> dict:
        """Etiketi okur ve REPORTED_FIELDS anahtarlarinin var oldugunu dogrular; aksi halde LabelError.

        FIELDS_MAY_BE_UNKNOWN'daki alanlar (ornegin seller_tax_no) istisna: bazi golden setlerde
        bu anahtar hic yok, bu eksik sayilmaz."""
        try:
            data = json.loads(self.label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabelError(f"{self.label_path.name}: okunamadi ({exc})") from exc
        if not isinstance(data, dict):
            raise LabelError(f"{self.label_path.name}: JSON nesnesi degil ({type(data).__name__})")
        missing = [
            name for name in REPORTED_FIELDS
            if name not in data and name not in FIELDS_MAY_BE_UNKNOWN
        ]
        if missing:
            raise LabelError(f"{self.label_path.name}: eksik alanlar {missing}")
        return data


class GoldenDataset:
    """data/golden/images ile data/golden/labels'i ayni isimli ciftler halinde eslestirir.

    images/ altindaki bir gorselin labels/ altinda karsiligi yoksa hata firlatilmaz -
    sessizce atlanir (skipped_unlabeled sayacina eklenir). Bu, data/real_world_test gibi
    gorsellerin etiketlerinden once/ayrik eklenebildigi setlerde (bkz. convert_kaggle_labels.py)
    kismi durumlarda kosunun durmamasi icin gerekli."""

    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")  # data/real_world_test (Kaggle) .jpg kullanir

    def __init__(self, golden_dir: Path):
        self.images_dir = golden_dir / "images"
        self.labels_dir = golden_dir / "labels"
        self.skipped_unlabeled = 0  # samples() cagrildiktan sonra gecerli

    def samples(self) -> list[GoldenSample]:
        result = []
        self.skipped_unlabeled = 0
        image_paths = [p for pattern in self.IMAGE_EXTENSIONS for p in self.images_dir.glob(pattern)]
        for image_path in sorted(image_paths):
            label_path = self.labels_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                self.skipped_unlabeled += 1
                continue
            result.append(GoldenSample(image_path.stem, image_path, label_path))
        return result


class FieldComparator:
    """Beklenen ve bulunan degeri alan turune gore karsilastirir.

    Sayilar 0.01 toleransla, metinler bosluklar sadelestirilerek, listeler ve
    sozlukler eleman eleman (ozyinelemeli) karsilastirilir.
    """

    def equal(self, expected, found) -> bool:
        if isinstance(expected, bool) or isinstance(found, bool):
            return expected == found
        if isinstance(expected, (int, float)):
            return self._numbers_equal(expected, found)
        if isinstance(expected, str):
            return isinstance(found, str) and self._normalize(expected) == self._normalize(found)
        if isinstance(expected, list):
            return (
                isinstance(found, list)
                and len(expected) == len(found)
                and all(self.equal(e, f) for e, f in zip(expected, found))
            )
        if isinstance(expected, dict):
            if not isinstance(found, dict):
                return False
            for key, value in expected.items():
                if value is None and key in FIELDS_MAY_BE_UNKNOWN:
                    continue  # bu alan kaynak veride hic yok (bkz. FIELDS_MAY_BE_UNKNOWN) - N/A, karsilastirilmaz
                if not self.equal(value, found.get(key)):
                    return False
            return True
        return expected == found

    def equal_date(self, expected, found) -> bool | None:
        """Tarihleri format farkindan bagimsiz (DD.MM.YYYY vs MM/DD/YYYY gibi) karsilastirir.

        Her iki taraf da hem "gun once" hem "ay once" varsayimiyla denenir (canlica
        dogrulandi: gun>12 olan tarihlerde dateutil zaten dogru sekli otomatik seciyor,
        gercekten belirsiz durumlarda - orn. '01/02/2020' - iki farkli sonuc uretiyor).
        Herhangi bir yorum kesisirse dogru sayilir. Taraflardan biri hic parse
        edilemezse (bos/anlamsiz metin) None (N/A) doner - yanlis sayilmaz."""
        expected_dates = self._parse_date_candidates(expected)
        found_dates = self._parse_date_candidates(found)
        if not expected_dates or not found_dates:
            return None
        return bool(expected_dates & found_dates)

    @staticmethod
    def _parse_date_candidates(value) -> set:
        if not isinstance(value, str) or not value.strip():
            return set()
        candidates = set()
        for dayfirst in (True, False):
            try:
                candidates.add(date_parser.parse(value, dayfirst=dayfirst, fuzzy=False).date())
            except (ValueError, OverflowError, TypeError):
                pass
        return candidates

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _numbers_equal(expected, found) -> bool:
        try:
            found_num = float(str(found).replace(",", "."))
        except (TypeError, ValueError):
            return False
        return abs(float(expected) - found_num) <= NUMERIC_TOLERANCE


def _gross_adjusted_items(items) -> list:
    """items_gross=true olan etiketlerde (bkz. scripts/convert_kaggle_labels.py) ground-truth
    line_total KDV DAHIL, ama Vision Agent'in kendi ciktisindaki line_total her zaman KDV
    HARIC (net) anlaminda. Iki taban dogrudan kiyaslanamaz; bu yuzden line_total'i N/A saymak
    yerine (butun items alanini olcemez birakirdi), Vision'in KENDI cikardigi vat_amount'u
    kendi line_total'ina ekleyip KARSILASTIRILABILIR bir gross deger uretiyoruz - boylece
    description/quantity/line_total uzerindeki gercek dogruluk sinyali korunmus olur.
    vat_amount sayiya cevrilemiyorsa (None/eksik) line_total oldugu gibi birakilir (muhtemelen
    yanlis sayilir, ama en azindan crash olmaz)."""
    if not isinstance(items, list):
        return items
    adjusted = []
    for item in items:
        if not isinstance(item, dict):
            adjusted.append(item)
            continue
        item = dict(item)
        line_total, vat_amount = item.get("line_total"), item.get("vat_amount")
        if _is_number(line_total) and _is_number(vat_amount):
            item["line_total"] = line_total + vat_amount
        adjusted.append(item)
    return adjusted


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class JsonlResultWriter:
    """Her ornegin sonucunu aninda dosyaya ekler (bir satir = bir gorsel).

    Kosu basinda dosya sifirlanir (append=False, varsayilan) ya da mevcut dosyanin
    sonuna eklenir (append=True - bkz. --resume, onceki bir kosunun sonuclarini korur).
    Her write() sonrasi flush edilir, boylece ortada cokme ya da Ctrl+C olsa bile o ana
    kadarki satirlar diskte kalir.
    """

    def __init__(self, path: Path, append: bool = False):
        self.path = path
        self.append = append
        self._file = None

    def __enter__(self) -> "JsonlResultWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a" if self.append else "w", encoding="utf-8")
        return self

    def __exit__(self, *_exc) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: dict) -> None:
        if self._file is None:
            raise RuntimeError("JsonlResultWriter 'with' blogu icinde kullanilmali")
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()


@dataclass
class EvaluationReport:
    """Alan bazli sayaclar ve yanlislarin ayrintili listesi."""

    correct: dict[str, int] = field(default_factory=dict)
    total: dict[str, int] = field(default_factory=dict)
    not_applicable: dict[str, int] = field(default_factory=dict)  # alan etikette hic yoksa (bkz. FIELDS_MAY_BE_UNKNOWN)
    mismatches: list[dict] = field(default_factory=list)
    parse_failures: list[dict] = field(default_factory=list)
    provider_errors: list[dict] = field(default_factory=list)
    label_errors: list[dict] = field(default_factory=list)
    durations_s: list[float] = field(default_factory=list)

    def record(self, field_name: str, stem: str, expected, found, is_correct: bool) -> None:
        self.total[field_name] = self.total.get(field_name, 0) + 1
        if is_correct:
            self.correct[field_name] = self.correct.get(field_name, 0) + 1
        else:
            self.mismatches.append({"image": stem, "field": field_name, "expected": expected, "found": found})

    def record_not_applicable(self, field_name: str) -> None:
        self.not_applicable[field_name] = self.not_applicable.get(field_name, 0) + 1

    def record_parse_failure(self, stem: str, errors: list[str]) -> None:
        self.parse_failures.append({"image": stem, "errors": errors})

    def record_provider_error(self, stem: str, message: str) -> None:
        self.provider_errors.append({"image": stem, "error": message})

    def record_label_error(self, stem: str, message: str) -> None:
        self.label_errors.append({"image": stem, "error": message})

    def accuracy(self, field_name: str) -> float:
        total = self.total.get(field_name, 0)
        return self.correct.get(field_name, 0) / total if total else 0.0

    def overall_accuracy(self) -> float:
        total = sum(self.total.values())
        return sum(self.correct.values()) / total if total else 0.0

    def print_table(self, fields: tuple[str, ...], sample_count: int) -> None:
        width = max(len(f) for f in fields + ("GENEL",))
        print(f"\n{'Alan':<{width}}  Dogru/Toplam  Dogruluk   N/A")
        print("-" * (width + 34))
        for name in fields:
            na = self.not_applicable.get(name, 0)
            print(
                f"{name:<{width}}  {self.correct.get(name, 0):>4}/{self.total.get(name, 0):<4}     "
                f"{self.accuracy(name):>6.1%}   {na}"
            )
        print("-" * (width + 34))
        print(f"{'GENEL':<{width}}  {sum(self.correct.values()):>4}/{sum(self.total.values()):<4}     {self.overall_accuracy():>6.1%}")
        avg = sum(self.durations_s) / len(self.durations_s) if self.durations_s else 0.0
        print(
            f"\nGorsel: {sample_count} | parse hatasi: {len(self.parse_failures)} | "
            f"saglayici hatasi: {len(self.provider_errors)} | etiket hatasi: {len(self.label_errors)} | "
            f"ortalama sure: {avg:.1f} sn/gorsel"
        )

    def write_json(self, path: Path, fields: tuple[str, ...], model_name: str) -> None:
        payload = {
            "model": model_name,
            "overall_accuracy": round(self.overall_accuracy(), 4),
            "accuracy": {name: round(self.accuracy(name), 4) for name in fields},
            "totals": {name: {"correct": self.correct.get(name, 0), "total": self.total.get(name, 0)} for name in fields},
            "not_applicable": {name: self.not_applicable.get(name, 0) for name in fields},
            "parse_failures": self.parse_failures,
            "provider_errors": self.provider_errors,
            "label_errors": self.label_errors,
            "mismatches": self.mismatches,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class VisionAgentEvaluator:
    """Her golden ornek icin agent'i calistirir, raporu doldurur, sonucu aninda JSONL'e yazar."""

    def __init__(
        self,
        agent: VisionAgent,
        comparator: FieldComparator | None = None,
        result_writer: JsonlResultWriter | None = None,
        pause_s: float = 0.0,
    ):
        self.agent = agent
        self.comparator = comparator or FieldComparator()
        self.result_writer = result_writer
        # Her gercek saglayici cagrisindan sonra beklenecek sure (rate limit icin, orn.
        # Gemini free-tier 15 istek/dk -> --pause-s 4.5). Varsayilan 0: yerel Ollama'da
        # ya da testlerde (sahte provider) gereksiz yavaslatma yapmaz.
        self.pause_s = pause_s

    def evaluate(self, samples: list[GoldenSample], report: EvaluationReport | None = None) -> EvaluationReport:
        """Ornekleri sirayla calistirir.

        - Etiket bozuksa model hic cagrilmaz; ornek label_errors'a yazilir, toplamlara girmez.
        - Zaman asimi / hatali cevap o ornegi yanlis sayar ve devam eder.
        - Sunucuya hic ulasilamiyorsa (ProviderUnavailableError) hemen durur.
        - Ctrl+C ile kesilirse o ana kadarki rapor korunur.
        Her durumda ornegin sonucu, varsa result_writer'a aninda yazilir.

        report verilirse (bkz. --resume) sifirdan kurulmaz, var olan sayaclara EKLENIR -
        boylece onceki bir kosunun (replay edilmis) sonuclariyla bu kosununkiler birlesir.
        """
        report = report if report is not None else EvaluationReport()
        count = len(samples)
        for index, sample in enumerate(samples, start=1):
            try:
                expected = sample.load_label()
            except LabelError as exc:
                report.record_label_error(sample.stem, str(exc))
                self._emit(sample.stem, "label_error", None, None, None, [str(exc)])
                print(f"[{index}/{count}] {sample.stem}  ETIKET HATASI: {exc}", flush=True)
                continue

            started = time.perf_counter()
            try:
                state = self.agent.run(PipelineState(image_path=str(sample.image_path)))
            except ProviderUnavailableError:
                raise
            except ProviderError as exc:
                elapsed = time.perf_counter() - started
                report.durations_s.append(elapsed)
                report.record_provider_error(sample.stem, str(exc))
                fields = self._score(sample.stem, expected, {}, report)
                self._emit(sample.stem, "provider_error", elapsed, fields, None, [str(exc)])
                print(f"[{index}/{count}] {sample.stem}  SAGLAYICI HATASI: {exc}", flush=True)
                self._pause()
                continue
            except KeyboardInterrupt:
                print(f"\nKesildi: {index - 1}/{count} gorsel tamamlandi, kismi rapor yaziliyor.", flush=True)
                break
            elapsed = time.perf_counter() - started
            report.durations_s.append(elapsed)

            if state.raw_extraction is None:
                report.record_parse_failure(sample.stem, state.validation_errors)
                status, errors = "parse_failure", list(state.validation_errors)
            else:
                status, errors = "ok", []
            fields = self._score(sample.stem, expected, state.raw_extraction or {}, report)
            self._emit(sample.stem, status, elapsed, fields, state.raw_extraction, errors)

            suffix = "  PARSE HATASI" if status == "parse_failure" else ""
            print(f"[{index}/{count}] {sample.stem}  ({elapsed:.1f} sn){suffix}", flush=True)
            self._pause()
        return report

    def _pause(self) -> None:
        if self.pause_s > 0:
            time.sleep(self.pause_s)

    def _score(self, stem: str, expected: dict, found: dict, report: EvaluationReport) -> dict[str, bool | None]:
        """Her raporlanan alani karsilastirir; alan -> dogru mu sozlugu dondurur (N/A ise None).
        items tek alan olarak puanlanir: satir sayisi ve her satirin tum alt alanlari eslesirse dogru
        (FIELDS_MAY_BE_UNKNOWN'daki alt alanlar None ise atlanir, bkz. FieldComparator.equal).
        invoice_date format-bagimsiz (bkz. FieldComparator.equal_date), items_gross=true ise
        items KDV dahil/haric tabanina gore ayarlanarak (bkz. _gross_adjusted_items) karsilastirilir."""
        results: dict[str, bool | None] = {}
        for name in REPORTED_FIELDS:
            if name not in expected:
                report.record_not_applicable(name)
                results[name] = None
                continue

            expected_value = expected[name]
            found_value = found.get(name)

            if name == "invoice_date":
                is_correct = self.comparator.equal_date(expected_value, found_value)
            else:
                if name == "items" and expected.get("items_gross"):
                    found_value = _gross_adjusted_items(found_value)
                is_correct = self.comparator.equal(expected_value, found_value)

            if is_correct is None:
                report.record_not_applicable(name)
                results[name] = None
                continue
            report.record(name, stem, expected_value, found_value, is_correct)
            results[name] = is_correct
        return results

    def _emit(self, stem, status, duration_s, fields, extraction, errors) -> None:
        if self.result_writer is None:
            return
        self.result_writer.write({
            "image": stem,
            "status": status,  # ok | parse_failure | provider_error | label_error
            "duration_s": None if duration_s is None else round(duration_s, 2),
            "fields": fields,
            "extraction": extraction,
            "errors": errors,
        })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionAgent'i golden set uzerinde degerlendirir.")
    parser.add_argument("--golden-dir", type=Path, default=Path("data/golden"))
    parser.add_argument("--report-path", type=Path, default=Path("data/golden/eval_report.json"),
                        help="Kosu sonu ozet raporu (JSON).")
    parser.add_argument("--results-path", type=Path, default=Path("data/golden/eval_results.jsonl"),
                        help="Gorsel basina aninda yazilan sonuclar (JSONL).")
    parser.add_argument("--limit", type=_positive_int, default=None,
                        help="Yalnizca ilk N gorseli calistir (hizli deneme).")
    parser.add_argument("--pause-s", type=float, default=0.0,
                        help="Her gorselden sonra beklenecek sure (rate limit icin; "
                             "orn. Gemini free-tier 15 istek/dk icin --pause-s 4.5). "
                             "Varsayilan 0 (bekleme yok - yerel Ollama icin uygundur).")
    parser.add_argument("--resume", action="store_true",
                        help="--results-path'teki onceki kosudan basariyla tamamlanmis "
                             "gorselleri (status != provider_error) atlar; sadece "
                             "provider_error alanlari ve hic islenmemis gorselleri "
                             "(yeniden) calistirir. Sonuclar JSONL'e EKLENIR (uzerine "
                             "yazilmaz); rapor onceki+yeni sonuclarin birlesimini gosterir.")
    return parser


def _load_previous_results(results_path: Path) -> dict[str, dict]:
    """--resume icin: JSONL'i okur, gorsel basina EN SON kaydi dondurur (ayni gorsel
    birden fazla kosuda tekrar islenmisse - orn. iki kez resume edilmisse - sonuncusu
    gecerlidir). Dosya yoksa bos sozluk (ilk kosu gibi davranilir, hata degil)."""
    records: dict[str, dict] = {}
    if not results_path.is_file():
        return records
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[record["image"]] = record
    return records


def _replay_done_result(report: EvaluationReport, record: dict, expected: dict) -> None:
    """Daha once basariyla tamamlanmis (status != provider_error) bir JSONL kaydini,
    YENIDEN karsilastirma yapmadan (kaydedilmis "fields" verdisine guvenerek) mevcut
    rapora ekler - boylece --resume'da onceki sonuclarla bu kosununkiler birlesir.
    Karsilastirma mantigi (equal_date/items_gross gibi) burada TEKRARLANMAZ: "fields"
    zaten orijinal kosuda bu mantikla hesaplanmis dogru/yanlis/N-A verdisidir."""
    stem = record["image"]
    status = record["status"]
    extraction = record.get("extraction") or {}
    fields = record.get("fields") or {}

    if status == "label_error":
        report.record_label_error(stem, next(iter(record.get("errors") or []), ""))
        return

    duration = record.get("duration_s")
    if duration is not None:
        report.durations_s.append(duration)
    if status == "parse_failure":
        report.record_parse_failure(stem, record.get("errors") or [])

    for name in REPORTED_FIELDS:
        is_correct = fields.get(name)
        if is_correct is None:
            report.record_not_applicable(name)
            continue
        report.record(name, stem, expected.get(name), extraction.get(name), is_correct)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"--limit 1 veya daha buyuk olmali, verilen: {value}")
    return number


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = GoldenDataset(args.golden_dir)
    all_samples = dataset.samples()
    if args.limit is not None:
        all_samples = all_samples[: args.limit]
    if not all_samples:
        raise SystemExit(f"Golden gorsel bulunamadi: {args.golden_dir / 'images'}")

    report = EvaluationReport()
    samples_to_run = all_samples

    if args.resume:
        previous = _load_previous_results(args.results_path)
        samples_by_stem = {s.stem: s for s in all_samples}
        done_stems = {stem for stem, rec in previous.items() if rec.get("status") != "provider_error"}
        for stem in done_stems:
            sample = samples_by_stem.get(stem)
            if sample is None:
                continue  # bu kosuda artik gecerli bir golden ornek degil (etiket/gorsel degismis)
            _replay_done_result(report, previous[stem], sample.load_label())
        samples_to_run = [s for s in all_samples if s.stem not in done_stems]
        print(
            f"Resume: {len(done_stems)} zaten tamamlanmis (atlanacak), "
            f"{len(samples_to_run)} islenecek (provider_error / hic islenmemis)", flush=True,
        )

    provider = get_llm_provider()
    model_name = getattr(provider, "model", type(provider).__name__)
    print(f"Provider: {type(provider).__name__} | model: {model_name} | gorsel: {len(samples_to_run)}", flush=True)
    print(f"Anlik sonuclar: {args.results_path}", flush=True)

    if samples_to_run:
        with JsonlResultWriter(args.results_path, append=args.resume) as writer:
            evaluator = VisionAgentEvaluator(VisionAgent(provider), result_writer=writer, pause_s=args.pause_s)
            evaluator.evaluate(samples_to_run, report=report)

    report.print_table(REPORTED_FIELDS, len(all_samples))
    print(f"\nEtiketli (islendi): {len(all_samples)} | Etiketsiz (atlandi): {dataset.skipped_unlabeled}")
    report.write_json(args.report_path, REPORTED_FIELDS, model_name)
    print(f"Rapor: {args.report_path}")


if __name__ == "__main__":
    main()

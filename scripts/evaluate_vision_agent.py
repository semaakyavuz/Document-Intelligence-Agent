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


class LabelError(ValueError):
    """Ground-truth etiketi okunamadi ya da beklenen semaya uymuyor."""


@dataclass(frozen=True)
class GoldenSample:
    """Bir golden gorsel ve onun ground-truth etiketi."""

    stem: str
    image_path: Path
    label_path: Path

    def load_label(self) -> dict:
        """Etiketi okur ve REPORTED_FIELDS anahtarlarinin var oldugunu dogrular; aksi halde LabelError."""
        try:
            data = json.loads(self.label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabelError(f"{self.label_path.name}: okunamadi ({exc})") from exc
        if not isinstance(data, dict):
            raise LabelError(f"{self.label_path.name}: JSON nesnesi degil ({type(data).__name__})")
        missing = [name for name in REPORTED_FIELDS if name not in data]
        if missing:
            raise LabelError(f"{self.label_path.name}: eksik alanlar {missing}")
        return data


class GoldenDataset:
    """data/golden/images ile data/golden/labels'i ayni isimli ciftler halinde eslestirir."""

    def __init__(self, golden_dir: Path):
        self.images_dir = golden_dir / "images"
        self.labels_dir = golden_dir / "labels"

    def samples(self) -> list[GoldenSample]:
        result = []
        for image_path in sorted(self.images_dir.glob("*.png")):
            label_path = self.labels_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                raise FileNotFoundError(f"Etiket yok: {label_path}")
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
            return isinstance(found, dict) and all(self.equal(v, found.get(k)) for k, v in expected.items())
        return expected == found

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


class JsonlResultWriter:
    """Her ornegin sonucunu aninda dosyaya ekler (bir satir = bir gorsel).

    Kosu basinda dosya sifirlanir; her write() sonrasi flush edilir, boylece
    ortada cokme ya da Ctrl+C olsa bile o ana kadarki satirlar diskte kalir.
    """

    def __init__(self, path: Path):
        self.path = path
        self._file = None

    def __enter__(self) -> "JsonlResultWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
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
        print(f"\n{'Alan':<{width}}  Dogru/Toplam  Dogruluk")
        print("-" * (width + 26))
        for name in fields:
            print(f"{name:<{width}}  {self.correct.get(name, 0):>4}/{self.total.get(name, 0):<4}     {self.accuracy(name):>6.1%}")
        print("-" * (width + 26))
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
    ):
        self.agent = agent
        self.comparator = comparator or FieldComparator()
        self.result_writer = result_writer

    def evaluate(self, samples: list[GoldenSample]) -> EvaluationReport:
        """Ornekleri sirayla calistirir.

        - Etiket bozuksa model hic cagrilmaz; ornek label_errors'a yazilir, toplamlara girmez.
        - Zaman asimi / hatali cevap o ornegi yanlis sayar ve devam eder.
        - Sunucuya hic ulasilamiyorsa (ProviderUnavailableError) hemen durur.
        - Ctrl+C ile kesilirse o ana kadarki rapor korunur.
        Her durumda ornegin sonucu, varsa result_writer'a aninda yazilir.
        """
        report = EvaluationReport()
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
        return report

    def _score(self, stem: str, expected: dict, found: dict, report: EvaluationReport) -> dict[str, bool]:
        """Her raporlanan alani karsilastirir; alan -> dogru mu sozlugu dondurur.
        items tek alan olarak puanlanir: satir sayisi ve her satirin tum alt alanlari eslesirse dogru."""
        results: dict[str, bool] = {}
        for name in REPORTED_FIELDS:
            is_correct = self.comparator.equal(expected[name], found.get(name))
            report.record(name, stem, expected[name], found.get(name), is_correct)
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
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"--limit 1 veya daha buyuk olmali, verilen: {value}")
    return number


def main() -> None:
    args = build_arg_parser().parse_args()
    samples = GoldenDataset(args.golden_dir).samples()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"Golden gorsel bulunamadi: {args.golden_dir / 'images'}")

    provider = get_llm_provider()
    model_name = getattr(provider, "model", type(provider).__name__)
    print(f"Provider: {type(provider).__name__} | model: {model_name} | gorsel: {len(samples)}", flush=True)
    print(f"Anlik sonuclar: {args.results_path}", flush=True)

    with JsonlResultWriter(args.results_path) as writer:
        report = VisionAgentEvaluator(VisionAgent(provider), result_writer=writer).evaluate(samples)

    report.print_table(REPORTED_FIELDS, len(samples))
    report.write_json(args.report_path, REPORTED_FIELDS, model_name)
    print(f"Rapor: {args.report_path}")


if __name__ == "__main__":
    main()

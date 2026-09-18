"""
convert_kaggle_labels.py

data/real_world_test/'teki Kaggle CSV'lerini (batch1_1.csv, batch1_2.csv, batch1_3.csv -
"Json Data" sutunu) bizim golden-set formatina (images/ + labels/*.json, bkz.
scripts/evaluate_vision_agent.py'nin GoldenDataset/GoldenSample'i) cevirir.

Sadece CSV'de referans verilen VE data/real_world_test/ altinda gercekten mevcut olan
gorseller icin cikti uretir - gorseli henuz eklenmemis satirlar sessizce atlanir (hata
degil); ileride daha fazla gorsel eklendiginde bu script'e hic dokunmadan otomatik
islenir. Zaten etiketi uretilmis gorseller ICIN TEKRAR CALISTIRILMAZ (idempotent ama
gereksiz isi atlar) - sadece eksik olanlar icin yeni etiket uretilir. images/ altinda
olup CSV'nin HICBIR satirinda karsiligi olmayan gorseller (varsa) ayrica raporlanir -
bunlar bu veri kaynagindan asla etiketlenemez.

Sayi formati (canlica incelendi - 1414 kaydin tamami, ~14000 sayisal alan uzerinde):
bosluk HER ZAMAN binlik grup ayraci ("34 320.00", "3 135,10"); virgul VE nokta HER
IKISI DE, ne zaman kullanilirlarsa kullanilsinlar, HER ZAMAN tam 2 haneli ondalik
suffix'i tasiyor (canlica dogrulandi: 3209/3209 virgul, 10819/10819 nokta - istisnasiz).
Hicbir kayitta virgul ve nokta birlikte gorulmedi (0/1414). Kural: once bosluklari sil,
sonra kalan virgulu (varsa) noktaya cevir, float() ile parse et - bu, tum veri kumesinde
belirsiz/cakisan bir durum olmadan calisir.

Alan eslemesi (ilk halinde "subtotal<-subtotal.total, grand_total<-subtotal+vat_total"
talimati harfiyen uygulanmisti; 21 orneklik canli evaluate_vision_agent.py kosusunda
Vision Agent'in bulduğu grand_total'in TAM OLARAK bizim o zamanki "subtotal"a esit
ciktigi gorulunce - yani Kaggle'in "subtotal.total"i gercekte ZATEN KDV dahil nihai
toplam - mapping asagidaki gibi duzeltildi, kullanicinin onayiyla):
    invoice_no       <- invoice.invoice_number
    invoice_date     <- invoice.invoice_date
    seller_name      <- invoice.seller_name
    buyer_name       <- invoice.client_name
    items[].description/quantity/line_total <- items[].description/quantity/total_price
        (DIKKAT: Kaggle'in total_price'i KDV DAHIL/gross; bizim semamizdaki line_total
        her yerde KDV HARIC/net anlaminda kullanilir - burada donusturulmeden, gross
        olarak birakildi, cunku per-item KDV orani bu veri setinde yok, guvenilir bir
        gross->net cevrimi yapilamaz. "items_gross": true ile isaretlenir.)
    items[].unit_price / vat_rate / vat_amount -> None (bu veri setinde yok)
    grand_total      <- subtotal.total (DOGRUDAN kaynaktan - zaten nihai toplam)
    vat_total        <- subtotal.tax
    subtotal         <- subtotal.total - vat_total (HESAPLANMIS, KDV haric), "subtotal_computed": true
    seller_tax_no    -> etikete hic eklenmez (bu alan Kaggle veri setinde yok; bkz.
                        evaluate_vision_agent.py'deki FIELDS_MAY_BE_UNKNOWN - bu yuzden
                        etiket dogrulamasinda eksik sayilmiyor)

NOT: Bu duzeltmeyle grand_total/subtotal, Vision Agent'in gercekte okudugu degerlerle
anlamli sekilde karsilastirilabilir hale geldi (evaluate_vision_agent.py ile canlica
dogrulandi). items alani hala gross/net uyusmazligi yuzunden dusuk cikabilir - bu
bilinen, "items_gross" bayragiyla belgelenen bir sinirlama, cozulmedi (cozumu per-item
KDV orani gerektirir, bu veri setinde yok).

Kullanim:
    python scripts/convert_kaggle_labels.py
    python scripts/convert_kaggle_labels.py --source-dir data/real_world_test
"""

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_CSV_NAMES = ("batch1_1.csv", "batch1_2.csv", "batch1_3.csv")
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


class ConversionError(ValueError):
    """Bir CSV satiri golden formata cevrilemedi (bozuk JSON, eksik/parse edilemeyen sayisal alan)."""


def _parse_number(raw: str) -> float:
    """'34 320.00' -> 34320.0, '3 135,10' -> 3135.1, '46.55' -> 46.55, '14,82' -> 14.82.

    Bosluk her zaman binlik ayracidir (silinir); kalan virgul (varsa) ondalik ayracidir
    (noktaya cevrilir) - bkz. modul docstring'indeki canli dogrulama notu."""
    text = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not text:
        raise ConversionError(f"bos sayisal deger: {raw!r}")
    try:
        return float(text)
    except ValueError as exc:
        raise ConversionError(f"sayi parse edilemedi: {raw!r}") from exc


def _convert_item(raw_item: dict) -> dict:
    return {
        "description": raw_item.get("description") or "",
        "quantity": _parse_number(raw_item.get("quantity", "")),
        "unit_price": None,
        "vat_rate": None,
        "line_total": _parse_number(raw_item.get("total_price", "")),
        "vat_amount": None,
    }


def convert_record(json_text: str) -> dict:
    """Bir CSV satirinin 'Json Data' alanini golden-set etiket semasina cevirir."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"gecersiz JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConversionError(f"JSON nesnesi degil: {type(data).__name__}")

    invoice = data.get("invoice") or {}
    items_raw = data.get("items") or []
    sub = data.get("subtotal") or {}

    if not items_raw:
        raise ConversionError("items listesi bos")

    items = [_convert_item(item) for item in items_raw]
    # Kaggle'in "subtotal.total"i dogrudan kaynaktan gelen NIHAI (KDV dahil) toplamdir
    # (canlica dogrulandi); bizim grand_total'imiz buna esittir, subtotal'imiz ise
    # buradan KDV'nin cikarilmasiyla HESAPLANIR.
    grand_total = _parse_number(sub.get("total", ""))
    vat_total = _parse_number(sub.get("tax", ""))
    subtotal = round(grand_total - vat_total, 2)

    return {
        "invoice_no": invoice.get("invoice_number") or "",
        "invoice_date": invoice.get("invoice_date") or "",
        "seller_name": invoice.get("seller_name") or "",
        "buyer_name": invoice.get("client_name") or "",
        "items": items,
        "items_gross": True,  # items[].line_total KDV dahil (Kaggle'in total_price'i) - net degil
        "subtotal": subtotal,
        "subtotal_computed": True,  # grand_total - vat_total olarak hesaplandi, kaynakta yok
        "vat_total": vat_total,
        "grand_total": grand_total,
        "grand_total_computed": False,  # dogrudan kaynaktan (subtotal.total)
    }


def _iter_csv_rows(source_dir: Path, csv_names: tuple[str, ...]):
    for csv_name in csv_names:
        path = source_dir / csv_name
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield csv_name, row


@dataclass
class ConversionSummary:
    """convert_all()'un tek seferlik kosu ozeti; main() bunu terminale yazdirir."""

    converted: int = 0
    already_labeled: int = 0
    skipped_no_image: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    no_csv_match: list[str] = field(default_factory=list)


def convert_all(source_dir: Path, csv_names: tuple[str, ...]) -> ConversionSummary:
    """CSV'lerdeki tum satirlari tarar; images/ altinda karsiligi olan VE henuz etiketi
    uretilmemis her gorsel icin bir etiket JSON'u yazar. Zaten etiketli olanlar tekrar
    islenmez (idempotent, gereksiz isi atlar). images/ altinda olup CSV'nin hicbir
    satirinda karsiligi olmayan gorseller summary.no_csv_match'te raporlanir."""
    images_out = source_dir / "images"
    labels_out = source_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    summary = ConversionSummary()
    csv_file_names: set[str] = set()

    for csv_name, row in _iter_csv_rows(source_dir, csv_names):
        file_name = (row.get("File Name") or "").strip()
        if not file_name:
            continue
        csv_file_names.add(file_name)

        label_path = labels_out / f"{Path(file_name).stem}.json"
        if label_path.is_file():
            summary.already_labeled += 1
            continue

        # Gorsel iki yerden biri gelebilir: dogrudan images/ altina konmus olabilir
        # (bu durumda kopyalamaya gerek yok), ya da henuz islenmemis, source_dir'in
        # kokunde ham .jpg olarak durabilir (eski/ilk yerlesim tarzi).
        already_placed = images_out / file_name
        raw_source = source_dir / file_name
        if already_placed.is_file():
            image_path, needs_copy = already_placed, False
        elif raw_source.is_file():
            image_path, needs_copy = raw_source, True
        else:
            summary.skipped_no_image += 1
            continue

        try:
            label = convert_record(row.get("Json Data") or "")
        except ConversionError as exc:
            summary.errors.append((file_name, str(exc)))
            if not needs_copy:
                # Gorsel zaten images/ altindaydi ama etiketi cikarilamadi (bozuk
                # kaynak veri) - GoldenDataset artik etiketsiz gorselleri sessizce
                # atladigi icin bu artik zorunlu degil, ama images/ klasorunu "hepsi
                # gecerli/etiketli" olarak temiz tutmak icin yine de disari tasiyoruz.
                shutil.move(image_path, source_dir / image_path.name)
            continue

        if needs_copy:
            shutil.copy(image_path, images_out / image_path.name)
        label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.converted += 1

    summary.no_csv_match = sorted(
        p.name for pattern in IMAGE_EXTENSIONS for p in images_out.glob(pattern)
        if p.name not in csv_file_names
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=Path("data/real_world_test"),
                         help="CSV'lerin ve mevcut .jpg gorsellerin bulundugu klasor")
    parser.add_argument("--csv-names", nargs="+", default=list(DEFAULT_CSV_NAMES))
    return parser


def main() -> None:
    # Windows'ta stdout, gercek bir konsola bagli degilse sistemin varsayilan ANSI kod
    # sayfasina duser, UTF-8'e degil; Turkce karakterleri sessizce bozar (bkz. diger
    # script'lerdeki ayni not).
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()

    summary = convert_all(args.source_dir, tuple(args.csv_names))

    print(f"Yeni donusturuldu: {summary.converted}")
    print(f"Zaten etiketliydi (atlandi, tekrar uretilmedi): {summary.already_labeled}")
    print(f"Gorseli henuz eklenmemis (atlandi, hata degil): {summary.skipped_no_image}")
    print(
        "CSV'de hic karsiligi olmayan gorsel (images/ icinde, hep etiketsiz kalacak): "
        f"{len(summary.no_csv_match)}"
    )
    for name in summary.no_csv_match:
        print(f"  {name}")
    print(f"Donusum hatasi: {len(summary.errors)}")
    for file_name, message in summary.errors:
        print(f"  {file_name}: {message}")


if __name__ == "__main__":
    main()

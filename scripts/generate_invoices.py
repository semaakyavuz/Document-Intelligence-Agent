"""
generate_invoices.py

Sentetik Turkce fatura goruntuleri + ground-truth (dogru cevap) JSON dosyalari uretir.
Amac: Vision Agent'i test etmek icin "sinav kagidi" gorevi gorecek veri seti olusturmak.
Model egitimi YAPMIYORUZ - sadece agent'in dogru okuyup okumadigini olcecegiz.

Kullanim:
    python generate_invoices.py --count 30 --out data/synthetic
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

fake = Faker("tr_TR")

# Faker'in tr_TR adres/sehir uretimi gercekci degil (Ingilizce sablonlari Turkce
# kelimeyle degistiriyor), o yuzden gercek il/ilce listesiyle kendimiz kuruyoruz.
CITIES = [
    ("İstanbul", "Kadıköy"), ("İstanbul", "Beşiktaş"), ("İzmit", "Merkez"),
    ("Ankara", "Çankaya"), ("İzmir", "Bornova"), ("Bursa", "Nilüfer"),
    ("Kocaeli", "Gebze"), ("Antalya", "Muratpaşa"),
]
STREET_TYPES = ["Caddesi", "Sokak", "Bulvarı"]
PRODUCTS = [
    "Ofis Kağıdı A4", "Toner Kartuşu", "Laptop Standı", "USB Kablo",
    "Klavye", "Mouse", "Monitör Kolu", "Yazıcı Mürekkebi", "Klasör",
    "Masaüstü Organizer", "Kırtasiye Seti", "Etiket Yazıcı Rulosu",
]
VAT_RATES = [0.01, 0.10, 0.20]  # Türkiye'deki KDV oranları


def fake_address() -> str:
    il, ilce = random.choice(CITIES)
    sokak = f"{fake.first_name()} {random.choice(STREET_TYPES)}"
    return f"{sokak} No:{random.randint(1, 120)} Kat:{random.randint(1, 8)}, {ilce}/{il}"


def fake_tax_number() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(10))


@dataclass
class InvoiceItem:
    description: str
    quantity: int
    unit_price: float
    vat_rate: float

    @property
    def line_total(self) -> float:
        return round(self.quantity * self.unit_price, 2)

    @property
    def vat_amount(self) -> float:
        return round(self.line_total * self.vat_rate, 2)


@dataclass
class Invoice:
    invoice_no: str
    invoice_date: str
    seller_name: str
    seller_address: str
    seller_tax_no: str
    buyer_name: str
    buyer_address: str
    items: list = field(default_factory=list)

    @property
    def subtotal(self) -> float:
        return round(sum(i.line_total for i in self.items), 2)

    @property
    def vat_total(self) -> float:
        return round(sum(i.vat_amount for i in self.items), 2)

    @property
    def grand_total(self) -> float:
        return round(self.subtotal + self.vat_total, 2)

    def to_ground_truth(self) -> dict:
        """Vision Agent'in cikardigi veriyle karsilastiracagimiz 'dogru cevap'."""
        return {
            "invoice_no": self.invoice_no,
            "invoice_date": self.invoice_date,
            "seller_name": self.seller_name,
            "seller_tax_no": self.seller_tax_no,
            "buyer_name": self.buyer_name,
            "items": [asdict(i) | {"line_total": i.line_total, "vat_amount": i.vat_amount} for i in self.items],
            "subtotal": self.subtotal,
            "vat_total": self.vat_total,
            "grand_total": self.grand_total,
        }


class InvoiceFactory:
    """Rastgele ama gecerli (matematigi tutarli) bir fatura uretir."""

    def generate(self) -> Invoice:
        n_items = random.randint(1, 5)
        items = [
            InvoiceItem(
                description=random.choice(PRODUCTS),
                quantity=random.randint(1, 10),
                unit_price=round(random.uniform(50, 2500), 2),
                vat_rate=random.choice(VAT_RATES),
            )
            for _ in range(n_items)
        ]
        invoice_date = date.today() - timedelta(days=random.randint(0, 365))
        return Invoice(
            invoice_no=f"{invoice_date.year}{random.randint(100000, 999999)}",
            invoice_date=invoice_date.strftime("%d.%m.%Y"),
            seller_name=fake.company(),
            seller_address=fake_address(),
            seller_tax_no=fake_tax_number(),
            buyer_name=fake.company(),
            buyer_address=fake_address(),
            items=items,
        )


class InvoiceRenderer:
    """Invoice objesini PNG goruntuye ciziyor. Birden fazla sablon = 'render_classic', 'render_minimal'."""

    WIDTH, HEIGHT = 1240, 1754  # A4, 150dpi civari

    # Kolon basliklari uc sablonda da kullaniliyor; render_compact'ta ayrica
    # values sozlugunun ANAHTARI olarak da geciyor. Tek kaynak olmasi, baslik
    # listesiyle sozluk anahtarlarinin sessizce ayrisip KeyError vermesini engelliyor.
    COL_DESC = "Açıklama"
    COL_QTY = "Miktar"
    COL_UNIT_PRICE = "Birim Fiyat"
    COL_VAT = "KDV%"
    COL_TOTAL = "Tutar"

    def __init__(self, font_dir: Path | None = None):
        self._font_reg_path, self._font_bold_path = self._resolve_font_paths(font_dir)
        self.font_regular = self._font(22, bold=False)
        self.font_bold = self._font(26, bold=True)

    @staticmethod
    def _resolve_font_paths(font_dir: Path | None) -> tuple[Path | None, Path | None]:
        candidates = []
        if font_dir:
            candidates.append((font_dir / "DejaVuSans.ttf", font_dir / "DejaVuSans-Bold.ttf"))
        # Linux'ta genelde hazir gelir.
        candidates.append((Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")))
        # Windows'ta Turkce karakterleri destekleyen Arial'a duser.
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates.append((windir / "Fonts" / "arial.ttf", windir / "Fonts" / "arialbd.ttf"))
        # macOS'ta Arial genelde bulunur.
        candidates.append((Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")))
        for reg, bold in candidates:
            if reg.exists() and bold.exists():
                return reg, bold
        print("UYARI: DejaVuSans.ttf bulunamadi, Turkce karakterler duzgun gorunmeyebilir. "
              "assets/fonts/ klasorune DejaVuSans.ttf ve DejaVuSans-Bold.ttf ekleyin.")
        return None, None

    def _font(self, size: int, bold: bool = False):
        """Sablonlar farkli boyutta yazi tipi kullanabilsin diye (letterhead'in buyuk
        basligi, compact'in kucuk yazisi gibi) yolu bir kere cozup istenen boyutta yeniden acar."""
        path = self._font_bold_path if bold else self._font_reg_path
        if path is None:
            return ImageFont.load_default(size=size)
        return ImageFont.truetype(str(path), size)

    def render_classic(self, inv: Invoice) -> Image.Image:
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), "white")
        d = ImageDraw.Draw(img)
        y = 60

        d.text((60, y), inv.seller_name, font=self.font_bold, fill="black"); y += 34
        d.text((60, y), inv.seller_address, font=self.font_regular, fill="black"); y += 28
        d.text((60, y), f"Vergi No: {inv.seller_tax_no}", font=self.font_regular, fill="black"); y += 50

        d.text((self.WIDTH // 2 - 60, y), "FATURA", font=self.font_bold, fill="black"); y += 50
        d.text((60, y), f"Fatura No: {inv.invoice_no}", font=self.font_regular, fill="black")
        d.text((600, y), f"Tarih: {inv.invoice_date}", font=self.font_regular, fill="black"); y += 40

        d.text((60, y), "Alıcı:", font=self.font_bold, fill="black"); y += 30
        d.text((60, y), inv.buyer_name, font=self.font_regular, fill="black"); y += 28
        d.text((60, y), inv.buyer_address, font=self.font_regular, fill="black"); y += 50

        headers = [self.COL_DESC, self.COL_QTY, self.COL_UNIT_PRICE,
                   self.COL_VAT, self.COL_TOTAL]
        col_x = [60, 560, 700, 920, 1040]
        for h, x in zip(headers, col_x):
            d.text((x, y), h, font=self.font_bold, fill="black")
        y += 30
        d.line((60, y, self.WIDTH - 60, y), fill="black", width=2); y += 15

        for item in inv.items:
            d.text((col_x[0], y), item.description, font=self.font_regular, fill="black")
            d.text((col_x[1], y), str(item.quantity), font=self.font_regular, fill="black")
            d.text((col_x[2], y), f"{item.unit_price:.2f}", font=self.font_regular, fill="black")
            d.text((col_x[3], y), f"%{int(item.vat_rate * 100)}", font=self.font_regular, fill="black")
            d.text((col_x[4], y), f"{item.line_total:.2f}", font=self.font_regular, fill="black")
            y += 34

        y += 20
        d.line((700, y, self.WIDTH - 60, y), fill="black", width=1); y += 20
        d.text((700, y), f"Ara Toplam: {inv.subtotal:.2f} TL", font=self.font_regular, fill="black"); y += 30
        d.text((700, y), f"KDV Toplam: {inv.vat_total:.2f} TL", font=self.font_regular, fill="black"); y += 30
        d.text((700, y), f"Genel Toplam: {inv.grand_total:.2f} TL", font=self.font_bold, fill="black")

        return img

    def render_compact(self, inv: Invoice) -> Image.Image:
        """Daha sikisik bir duzen: ust bilgi iki sutunda yan yana, tablo basliklari
        ters sirada (Tutar en solda, Aciklama en sagda), daha kucuk yazi tipi."""
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), "white")
        d = ImageDraw.Draw(img)
        reg, bold = self._font(16, bold=False), self._font(18, bold=True)
        left_x, right_x = 60, self.WIDTH // 2 + 20
        y = 50

        d.text((self.WIDTH // 2 - 40, y), "FATURA", font=bold, fill="black")
        y += 32
        d.text((left_x, y), f"No: {inv.invoice_no}", font=reg, fill="black")
        d.text((right_x, y), f"Tarih: {inv.invoice_date}", font=reg, fill="black")
        y += 36

        d.text((left_x, y), "Satıcı", font=bold, fill="black")
        d.text((right_x, y), "Alıcı", font=bold, fill="black")
        y += 22
        d.text((left_x, y), inv.seller_name, font=reg, fill="black")
        d.text((right_x, y), inv.buyer_name, font=reg, fill="black")
        y += 20
        d.text((left_x, y), inv.seller_address, font=reg, fill="black")
        d.text((right_x, y), inv.buyer_address, font=reg, fill="black")
        y += 20
        d.text((left_x, y), f"Vergi No: {inv.seller_tax_no}", font=reg, fill="black")
        y += 36

        # Basliklar ters sirada: Tutar en solda, Aciklama en sagda.
        headers = [self.COL_TOTAL, self.COL_VAT, self.COL_UNIT_PRICE,
                   self.COL_QTY, self.COL_DESC]
        col_x = [60, 220, 340, 520, 660]
        for h, x in zip(headers, col_x):
            d.text((x, y), h, font=bold, fill="black")
        y += 22
        d.line((60, y, self.WIDTH - 60, y), fill="black", width=1)
        y += 10

        for item in inv.items:
            values = {
                self.COL_DESC: item.description,
                self.COL_QTY: str(item.quantity),
                self.COL_UNIT_PRICE: f"{item.unit_price:.2f}",
                self.COL_VAT: f"%{int(item.vat_rate * 100)}",
                self.COL_TOTAL: f"{item.line_total:.2f}",
            }
            for h, x in zip(headers, col_x):
                d.text((x, y), values[h], font=reg, fill="black")
            y += 24

        y += 14
        d.line((500, y, self.WIDTH - 60, y), fill="black", width=1)
        y += 14
        d.text((500, y), f"Ara Toplam: {inv.subtotal:.2f} TL", font=reg, fill="black"); y += 22
        d.text((500, y), f"KDV Toplam: {inv.vat_total:.2f} TL", font=reg, fill="black"); y += 22
        d.text((500, y), f"Genel Toplam: {inv.grand_total:.2f} TL", font=bold, fill="black")

        return img

    def render_letterhead(self, inv: Invoice) -> Image.Image:
        """Buyuk renkli sirket antetli kagidi hissi: satici adi buyuk/koyu mavi,
        altinda ince cizgi, FATURA basligi sag ustte, tablo hucreleri cerceveli."""
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), "white")
        d = ImageDraw.Draw(img)
        letterhead_font = self._font(40, bold=True)
        LETTERHEAD_COLOR = (30, 58, 95)  # siyah yerine koyu mavi-gri
        y = 50

        d.text((60, y), inv.seller_name, font=letterhead_font, fill=LETTERHEAD_COLOR)
        d.text((self.WIDTH - 200, y + 10), "FATURA", font=self.font_bold, fill="black")
        y += 60
        d.line((60, y, self.WIDTH - 60, y), fill=LETTERHEAD_COLOR, width=3)
        y += 20

        d.text((60, y), inv.seller_address, font=self.font_regular, fill="black")
        d.text((self.WIDTH - 320, y), f"Fatura No: {inv.invoice_no}", font=self.font_regular, fill="black")
        y += 28
        d.text((60, y), f"Vergi No: {inv.seller_tax_no}", font=self.font_regular, fill="black")
        d.text((self.WIDTH - 320, y), f"Tarih: {inv.invoice_date}", font=self.font_regular, fill="black")
        y += 50

        d.text((60, y), "Alıcı:", font=self.font_bold, fill="black"); y += 30
        d.text((60, y), inv.buyer_name, font=self.font_regular, fill="black"); y += 28
        d.text((60, y), inv.buyer_address, font=self.font_regular, fill="black"); y += 50

        headers = [self.COL_DESC, self.COL_QTY, self.COL_UNIT_PRICE,
                   self.COL_VAT, self.COL_TOTAL]
        col_edges = [60, 560, 700, 920, 1040, self.WIDTH - 60]
        row_height = 36

        for x_start, x_end, h in zip(col_edges, col_edges[1:], headers):
            d.rectangle((x_start, y, x_end, y + row_height), outline="black", width=1)
            d.text((x_start + 6, y + 6), h, font=self.font_bold, fill="black")
        y += row_height

        for item in inv.items:
            values = [
                item.description, str(item.quantity), f"{item.unit_price:.2f}",
                f"%{int(item.vat_rate * 100)}", f"{item.line_total:.2f}",
            ]
            for x_start, x_end, val in zip(col_edges, col_edges[1:], values):
                d.rectangle((x_start, y, x_end, y + row_height), outline="black", width=1)
                d.text((x_start + 6, y + 6), val, font=self.font_regular, fill="black")
            y += row_height

        y += 20
        d.text((700, y), f"Ara Toplam: {inv.subtotal:.2f} TL", font=self.font_regular, fill="black"); y += 30
        d.text((700, y), f"KDV Toplam: {inv.vat_total:.2f} TL", font=self.font_regular, fill="black"); y += 30
        d.text((700, y), f"Genel Toplam: {inv.grand_total:.2f} TL", font=self.font_bold, fill="black")

        return img


TEMPLATES = {
    "classic": InvoiceRenderer.render_classic,
    "compact": InvoiceRenderer.render_compact,
    "letterhead": InvoiceRenderer.render_letterhead,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--template",
        choices=["classic", "compact", "letterhead", "random"],
        default="random",
        help="Fatura gorsel sablonu; 'random' her fatura icin uctan birini rastgele secer.",
    )
    return parser


def main():
    # Windows'ta stdout gercek bir konsola bagli degilse (yonlendirme/boru hatti) sistemin
    # varsayilan ANSI kod sayfasina duser, UTF-8'e degil; bu da asagidaki "uretildi" mesajindaki
    # Turkce karakteri sessizce bozabilir (bkz. run_pipeline_manual.py'de canlica yasanan hata).
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        Faker.seed(args.seed)

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    factory = InvoiceFactory()
    renderer = InvoiceRenderer(font_dir=Path("assets/fonts"))

    for i in range(1, args.count + 1):
        invoice = factory.generate()
        template_name = args.template if args.template != "random" else random.choice(list(TEMPLATES))
        img = TEMPLATES[template_name](renderer, invoice)

        img_path = out_dir / "images" / f"invoice_{i:04d}.png"
        label_path = out_dir / "labels" / f"invoice_{i:04d}.json"

        img.save(img_path)
        label_path.write_text(
            json.dumps(invoice.to_ground_truth(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"{args.count} sentetik fatura üretildi -> {out_dir}/images ve {out_dir}/labels")


if __name__ == "__main__":
    main()

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

    def __init__(self, font_dir: Path | None = None):
        self.font_regular, self.font_bold = self._load_fonts(font_dir)

    @staticmethod
    def _load_fonts(font_dir: Path | None):
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
                return ImageFont.truetype(str(reg), 22), ImageFont.truetype(str(bold), 26)
        print("UYARI: DejaVuSans.ttf bulunamadi, Turkce karakterler duzgun gorunmeyebilir. "
              "assets/fonts/ klasorune DejaVuSans.ttf ve DejaVuSans-Bold.ttf ekleyin.")
        default = ImageFont.load_default()
        return default, default

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

        headers = ["Açıklama", "Miktar", "Birim Fiyat", "KDV%", "Tutar"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

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
        img = renderer.render_classic(invoice)

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

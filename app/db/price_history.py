"""
price_history.py

InvoiceRecord.full_report'ta saklanan gecmis fatura kalemlerinden (bkz. ReportAgent'in
final_report'a ekledigi "items" alani), ayni urun aciklamasina sahip onceki birim
fiyatlari bulur. Basit bir metin eslestirmesi kullanir (kucuk/buyuk harf ve bosluk
normalize edilerek TAM esitlik) - bulanik/benzerlik eslestirmesi kapsam disi.

Bu ozellik yalnizca ILERIYE DONUK calisir: eski kayitlarin full_report'unda "items"
alani olmadigi icin (bu alan bu degisiklikle eklendi), gecmis kayitlar hicbir zaman
eslesme uretmez - bu bir hata degil, sadece verinin henuz birikmemis olmasidir.
"""

from sqlalchemy.orm import Session

from app.db.models import InvoiceRecord

MAX_RECORDS_SCANNED = 200
MAX_PRICES_PER_DESCRIPTION = 5


def _normalize(description: str) -> str:
    return description.strip().casefold()


def _record_item_price(item: object, prices_by_key: dict[str, list[float]]) -> None:
    """Tek kalemin birim fiyatini, ilgilendigimiz bir urunse prices_by_key'e YERINDE ekler.

    Ilgilenmedigimiz urun, okunamayan aciklama/fiyat ya da dolmus kota (bkz.
    MAX_PRICES_PER_DESCRIPTION) durumunda sessizce hicbir sey yapmaz."""
    if not isinstance(item, dict):
        return
    description = item.get("description")
    if not isinstance(description, str):
        return
    prices = prices_by_key.get(_normalize(description))
    if prices is None or len(prices) >= MAX_PRICES_PER_DESCRIPTION:
        return
    unit_price = item.get("unit_price")
    if isinstance(unit_price, (int, float)) and not isinstance(unit_price, bool):
        prices.append(float(unit_price))


def get_price_history(session: Session, descriptions: list[str]) -> list[dict]:
    """Verilen urun aciklamalari icin, veritabanindaki en yeni MAX_RECORDS_SCANNED
    kayitta gorulen birim fiyatlari toplar. Bir aciklama icin hic eslesme yoksa
    sonuca dahil edilmez (hata degil, sadece o urun icin gecmis veri yok demektir)."""
    targets = {_normalize(d): d for d in descriptions if isinstance(d, str) and d.strip()}
    if not targets:
        return []

    prices_by_key: dict[str, list[float]] = {key: [] for key in targets}

    rows = (
        session.query(InvoiceRecord.full_report)
        .order_by(InvoiceRecord.id.desc())
        .limit(MAX_RECORDS_SCANNED)
        .all()
    )
    for (full_report,) in rows:
        if not full_report:
            continue
        for item in full_report.get("items") or []:
            _record_item_price(item, prices_by_key)

    return [
        {"description": targets[key], "previous_prices": prices}
        for key, prices in prices_by_key.items()
        if prices
    ]

"""
save_pipeline_result.py

Bir fatura gorselini gercek LangGraph pipeline'i (app/graph.py) ile isler ve sonucu
(final_report) InvoiceRecord olarak veritabanina yazar. FastAPI katmani gelmeden once
veritabani katmanini kendi basina (izole) dogrulayabilmek icin: bu script'in FastAPI'ye
hic ihtiyaci yok.

Kullanim:
    python scripts/save_pipeline_result.py data/golden_mixed/images/invoice_0003_aug04.png
"""

import argparse
import asyncio
import sys
from pathlib import Path

# scripts/ klasorunden calistirildiginda app paketinin bulunabilmesi icin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import InvoiceRecord  # noqa: E402
from app.db.session import get_session_factory, init_db  # noqa: E402
from app.graph import build_pipeline_graph  # noqa: E402
from app.state import PipelineState  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Islenip veritabanina kaydedilecek fatura gorselinin yolu")
    return parser


async def _run_pipeline(image_path: str) -> dict:
    graph = build_pipeline_graph()
    raw_result = await graph.ainvoke(PipelineState(image_path=image_path))
    # graph.ainvoke() duz bir dict dondurur; None kalan alanlar dusebilir
    # (bkz. run_pipeline_graph.py'deki ayni not); PipelineState(**raw_result) geri tamamlar.
    final_state = PipelineState(**raw_result)
    return final_state.final_report


def _to_invoice_record(image_path: str, final_report: dict) -> InvoiceRecord:
    return InvoiceRecord(
        image_filename=Path(image_path).name,
        invoice_no=final_report.get("invoice_no"),
        seller_name=final_report.get("seller_name"),
        grand_total=final_report.get("grand_total"),
        is_valid=final_report["is_valid"],
        anomaly_count=final_report["anomaly_count"],
        anomalies=final_report["anomalies"],
        full_report=final_report,
    )


def main() -> None:
    # Windows'ta stdout, gercek bir konsola bagli degilse (yonlendirme/boru hatti) sistemin
    # varsayilan ANSI kod sayfasina duser, UTF-8'e degil; bu da Turkce karakterleri sessizce
    # bozar. Bkz. run_pipeline_manual.py'deki ayni not.
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()

    final_report = asyncio.run(_run_pipeline(args.image))
    print(f"Pipeline tamamlandi: {final_report['summary']}")

    init_db()
    session_factory = get_session_factory()
    with session_factory() as session:
        record = _to_invoice_record(args.image, final_report)
        session.add(record)
        session.commit()
        session.refresh(record)
        print(f"Veritabanina yazildi: InvoiceRecord(id={record.id}, image_filename={record.image_filename!r})")


if __name__ == "__main__":
    main()

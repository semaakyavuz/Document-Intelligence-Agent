"""
run_pipeline_graph.py

app/graph.py'deki gercek LangGraph StateGraph'ini .ainvoke() ile calistirir.
run_pipeline_manual.py ile ayni CLI arayuzu ve ayni final_report ciktisini
verir; farkla, dort agent'i duz Python yerine derlenmis (compile edilmis)
bir graph uzerinden zincirler.

Async: RAGAgent artik bir MCP client'i oldugu ve rag_node bunun icin async
tanimlandigi icin (bkz. app/graph.py), graph .invoke() ile degil .ainvoke()
ile calistirilmali (LangGraph, tek bir async node bile olsa senkron invoke'u
reddediyor).

Kullanim:
    python scripts/run_pipeline_graph.py data/golden_mixed/images/invoice_0003_aug04.png
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# scripts/ klasorunden calistirildiginda app paketinin bulunabilmesi icin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph import build_pipeline_graph  # noqa: E402
from app.state import PipelineState  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Degerlendirilecek fatura gorselinin yolu")
    return parser


async def _run(image_path: str) -> None:
    initial_state = PipelineState(image_path=image_path)
    print(f"Baslangic: image_path = {initial_state.image_path}")

    graph = build_pipeline_graph()
    raw_result = await graph.ainvoke(initial_state)
    # graph.ainvoke() duz bir dict dondurur (PipelineState degil) ve degeri None kalan
    # alanlar bu dict'ten tamamen dusuyor (canlica dogrulanmis langgraph davranisi);
    # PipelineState(**raw_result) alanlari kendi varsayilanlarina gore geri tamamlar.
    final_state = PipelineState(**raw_result)

    print("Graph calisti: vision -> rag -> validation -> report")
    print("\n=== FINAL REPORT ===")
    print(json.dumps(final_state.final_report, ensure_ascii=False, indent=2))


def main() -> None:
    # Windows'ta stdout, gercek bir konsola bagli degilse (yonlendirme/boru hatti) sistemin
    # varsayilan ANSI kod sayfasina (ornegin Turkce Windows'ta cp1254) duser, UTF-8'e degil;
    # bu da Turkce karakterleri sessizce bozar. Bkz. run_pipeline_manual.py'deki ayni not.
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()
    asyncio.run(_run(args.image))


if __name__ == "__main__":
    main()

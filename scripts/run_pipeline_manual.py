"""
run_pipeline_manual.py

LangGraph'a gecmeden once dort agent'i (Vision, RAG, Validation, Report) duz Python'da
sirayla zincirleyen manuel bir dogrulama script'i. LangGraph kullanmaz; tek amaci
pipeline'in uctan uca gercekten calistigini goz onunde dogrulamaktir.

Kullanim:
    python scripts/run_pipeline_manual.py data/golden_mixed/images/invoice_0003_aug04.png
"""

import argparse
import json
import sys
from pathlib import Path

# scripts/ klasorunden calistirildiginda app paketinin bulunabilmesi icin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.rag_agent import RAGAgent  # noqa: E402
from app.agents.report_agent import ReportAgent  # noqa: E402
from app.agents.validation_agent import ValidationAgent  # noqa: E402
from app.agents.vision_agent import VisionAgent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.providers.factory import get_llm_provider  # noqa: E402
from app.state import PipelineState  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Degerlendirilecek fatura gorselinin yolu")
    return parser


def main() -> None:
    # Windows'ta stdout, gercek bir konsola bagli degilse (yonlendirme/boru hatti) sistemin
    # varsayilan ANSI kod sayfasina (ornegin Turkce Windows'ta cp1254) duser, UTF-8'e degil;
    # bu da Turkce karakterleri (ornegin 's،ş' yerine tek baytlik bozuk bir karakter) sessizce
    # bozar. Ciktinin nereye yonlendirildiginden bagimsiz dogru kalmasi icin acikca UTF-8'e sabitlenir.
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()
    settings = Settings()

    state = PipelineState(image_path=args.image)
    print(f"Baslangic: image_path = {state.image_path}")

    vision = VisionAgent(get_llm_provider(settings))
    state = vision.run(state)
    if state.raw_extraction is not None:
        print(f"[1/4] VisionAgent     -> raw_extraction dolduruldu ({len(state.raw_extraction)} alan).")
    else:
        print(f"[1/4] VisionAgent     -> raw_extraction BOS (parse hatasi): {state.validation_errors}")

    rag = RAGAgent(top_k=settings.RAG_TOP_K)
    state = rag.run(state)
    print(f"[2/4] RAGAgent        -> retrieved_rules dolduruldu ({len(state.retrieved_rules)} kural).")

    validation = ValidationAgent()
    state = validation.run(state)
    print(f"[3/4] ValidationAgent -> anomalies dolduruldu ({len(state.anomalies)} anomali), is_valid={state.is_valid}.")

    report = ReportAgent()
    state = report.run(state)
    print("[4/4] ReportAgent     -> final_report dolduruldu.")

    print("\n=== FINAL REPORT ===")
    print(json.dumps(state.final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
debug_single_extraction.py

Tek bir gorsel icin VisionAgent'in kullandigi PROMPT ile provider.generate()'i
dogrudan cagirir ve donen HAM metni (hic kirpmadan) terminale yazar.

Neden: eval_results.jsonl'deki "errors" alani sadece ilk ~100 karakteri tutuyor
(evaluate_vision_agent.py'deki hata mesaji onizlemesi yuzunden). Bu script gecici bir
teshis araci; kalici olmasi gerekmiyor, elle calistirilir.

Kullanim:
    python scripts/debug_single_extraction.py
    python scripts/debug_single_extraction.py --image data/golden/images/invoice_0005_aug05.png

Provider .env'den gelir (LLM_PROVIDER, GEMINI_API_KEY/GEMINI_VISION_MODEL vb.).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.vision_agent import VisionAgent  # noqa: E402
from app.providers.factory import get_llm_provider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("data/golden/images/invoice_0003_aug04.png"))
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Gorsel bulunamadi: {args.image}")

    provider = get_llm_provider()
    print(f"Provider: {type(provider).__name__} | model: {getattr(provider, 'model', '?')}")
    print(f"Gorsel: {args.image}")
    print("-" * 70, flush=True)

    raw_text = provider.generate(
        VisionAgent.EXTRACTION_PROMPT, image_path=str(args.image), max_tokens=VisionAgent.MAX_OUTPUT_TOKENS
    )

    print(f"HAM CEVAP ({len(raw_text)} karakter):")
    print("=" * 70)
    print(raw_text)
    print("=" * 70)

    # Gemini icin ek teshis: cevap kisa/kirpilmis gorunuyorsa nedeni (finish_reason,
    # guvenlik filtresi, token kullanimi) provider.generate()'in dondurdugu duz metinde yok;
    # ham response nesnesinde var. Bunun icin ayni istegi ikinci kez atmak gerekiyor (LLMProvider
    # ABC'si yalnizca str dondurur); provider._build_generation_config() kullanarak yukaridaki
    # cagriyla BIREBIR ayni ayarlarla (temperature, thinking_config, response_mime_type) gidilir,
    # yoksa bu teshis bilgisi gercek davranistan sapabilir. include_thinking, yukaridaki gercek
    # generate() cagrisinin ogrendigi degeri (provider._thinking_supported) kullanir; boylece
    # model thinking_config'i reddediyorsa burada da (bosuna) tekrar denenmez.
    client = getattr(provider, "_client", None)
    if client is not None:
        include_thinking = provider._thinking_supported is not False
        response = client.models.generate_content(
            model=provider.model,
            contents=[VisionAgent.EXTRACTION_PROMPT, provider._load_image(str(args.image))],
            config=provider._build_generation_config(VisionAgent.MAX_OUTPUT_TOKENS, include_thinking=include_thinking),
        )
        print("\nTESHIS BILGISI (Gemini):")
        if response.candidates:
            candidate = response.candidates[0]
            print(f"  finish_reason  : {candidate.finish_reason}")
            print(f"  finish_message : {candidate.finish_message}")
            print(f"  safety_ratings : {candidate.safety_ratings}")
        else:
            print(f"  candidates bos. prompt_feedback: {response.prompt_feedback}")
        print(f"  usage_metadata : {response.usage_metadata}")


if __name__ == "__main__":
    main()

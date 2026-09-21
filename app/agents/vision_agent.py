"""
vision_agent.py

Fatura gorselini gorsel destekli LLM'e gonderir ve alanlari JSON olarak cikarir.

Cikti semasi scripts/generate_invoices.py'nin urettigi ground-truth JSON ile
birebir aynidir; boylece evaluate_vision_agent.py ikisini dogrudan karsilastirir.
"""

import json
import re

from app.agents.base import BaseAgent
from app.providers.base import LLMProvider
from app.state import PipelineState


class JsonResponseParser:
    """LLM cevabindan JSON nesnesini ayiklar.

    Modeller JSON'u bazen ```json ... ``` icine, bazen aciklama cumlelerinin
    arasina koyar. Sirayla: cit icerigi -> metnin tamami -> ilk '{' ile son '}' arasi.
    """

    # Bastaki/sondaki boslugu regex yerine .strip() ile kirpiyoruz. Onceki desen
    # `\s*(.*?)\s*` idi: bosluklar hem `\s*` hem `.*?` tarafindan eslesebildigi icin,
    # kapanis citi olmayan uzun bir bosluk dizisinde motor super-lineer geri izlemeye
    # giriyordu (olculdu: 800 bosluk -> 579 ms, 2000 bosluk -> pratikte hic bitmiyor).
    # Model cevabi disaridan geldigi icin bu bir ReDoS yoluydu. Bu desen lineer ve
    # 15 girdilik karsilastirmada eskisiyle birebir ayni sonucu veriyor.
    _FENCE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)

    def parse(self, text: str) -> dict:
        candidates = [text.strip()]
        fenced = self._FENCE.search(text)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
            raise ValueError(f"JSON bir nesne (dict) degil: {type(data).__name__}")
        raise ValueError(f"Cevapta gecerli JSON bulunamadi: {text[:120]!r}")


class VisionAgent(BaseAgent):
    """Gorselden fatura alanlarini cikarir ve state.raw_extraction'a yazar."""

    # Dikkat: sema icine ornek DEGER yazilmaz. Kucuk modeller "e.g. 2025864544" gibi ornekleri
    # ve "(Satici / seller)" gibi ipuclarini okumak yerine aynen kopyaliyor (golden eval'de gozlendi).
    EXTRACTION_PROMPT = """You are reading a Turkish invoice image. Extract the fields below and return ONLY a JSON object, no explanation, no markdown.

Schema (keys and value types must match exactly; every value must be read from the image, never copied from this schema):
{
  "invoice_no": string,
  "invoice_date": string,
  "seller_name": string,
  "seller_tax_no": string,
  "buyer_name": string,
  "items": [
    {
      "description": string,
      "quantity": number,
      "unit_price": number,
      "vat_rate": number,
      "line_total": number,
      "vat_amount": number
    }
  ],
  "subtotal": number,
  "vat_total": number,
  "grand_total": number
}

Field guide:
- invoice_no: the invoice number printed on the document (digits only).
- invoice_date: the invoice date in DD.MM.YYYY form.
- seller_name / buyer_name: the company names printed under the seller and buyer headings, not the headings themselves.
- seller_tax_no: the seller's tax number, exactly 10 digits.
- items: one entry per line of the product table, in order.
- vat_rate: a fraction, so a 20 percent VAT column is written as 0.2.
- subtotal, vat_total, grand_total: the totals printed at the bottom of the invoice.

Numbers must be plain JSON numbers with a dot decimal separator. Keep Turkish characters in names exactly as they appear on the invoice. If a value is unreadable, use null.

Türkçe özel karakterleri (ç, ğ, ı, ö, ş, ü ve büyük halleri Ç, Ğ, İ, Ö, Ş, Ü) gördüğün şekliyle birebir koru, ASCII karşılığına (c, g, i, o, s, u) çevirme."""

    # Gecerli bir fatura JSON'u 500 token'i gecmez; model donguye girerse burada kesilir,
    # zaman asimina kadar bosuna beklenmez.
    MAX_OUTPUT_TOKENS = 1024

    def __init__(self, llm_provider: LLMProvider, parser: JsonResponseParser | None = None):
        super().__init__(llm_provider)
        self._parser = parser or JsonResponseParser()

    def run(self, state: PipelineState) -> PipelineState:
        # Baglanti hatalari burada yutulmaz: "model yanlis cevap verdi" degil "altyapi yok" demektir.
        raw_text = self.llm_provider.generate(
            self.EXTRACTION_PROMPT, image_path=state.image_path, max_tokens=self.MAX_OUTPUT_TOKENS
        )
        try:
            extraction = self._parser.parse(raw_text)
        except ValueError as exc:
            return state.model_copy(
                update={"validation_errors": [*state.validation_errors, f"VisionAgent: {exc}"]}
            )
        return state.model_copy(update={"raw_extraction": extraction})

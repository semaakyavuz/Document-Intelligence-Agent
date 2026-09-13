"""
groq_provider.py

Groq bulut API'si uzerinden LLM. Groq embedding servisi sunmadigi icin
yalnizca LLMProvider implemente edilir. Gercek cagri 4. fazda eklenecek.
"""

from app.providers.base import LLMProvider


class GroqLLMProvider(LLMProvider):
    """Groq chat completions API'si uzerinden metin/gorsel uretimi."""

    def __init__(self, api_key: str | None):
        if not api_key:
            raise ValueError("GROQ_API_KEY tanimli degil; .env dosyasina ekleyin")
        self.api_key = api_key

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        # TODO(faz 4): groq SDK ile chat.completions; image_path varsa vision modeli ve image_url icerigi kullan.
        raise NotImplementedError("GroqLLMProvider.generate 4. fazda eklenecek")

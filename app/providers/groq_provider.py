"""
groq_provider.py

Groq bulut API'si uzerinden LLM. Groq embedding servisi sunmadigi icin
yalnizca LLMProvider implemente edilir. Gercek cagri henuz eklenmedi.
"""

from app.providers.base import LLMProvider


class GroqLLMProvider(LLMProvider):
    """Groq chat completions API'si uzerinden metin/gorsel uretimi."""

    def __init__(self, api_key: str | None, temperature: float = 0.0):
        if not api_key:
            raise ValueError("GROQ_API_KEY tanimli degil; .env dosyasina ekleyin")
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        # TODO: groq SDK ile chat.completions (temperature=self.temperature); image_path varsa
        # vision modeli ve image_url icerigi kullan.
        raise NotImplementedError("GroqLLMProvider.generate henuz eklenmedi")

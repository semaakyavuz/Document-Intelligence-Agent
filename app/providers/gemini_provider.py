"""
gemini_provider.py

Google Gemini API'si uzerinden LLM ve embedding. Gercek cagrilar 4. fazda eklenecek.
"""

from app.providers.base import EmbeddingProvider, LLMProvider


def _require_api_key(api_key: str | None) -> str:
    if not api_key:
        raise ValueError("GEMINI_API_KEY tanimli degil; .env dosyasina ekleyin")
    return api_key


class GeminiLLMProvider(LLMProvider):
    """Gemini generateContent ucu uzerinden metin/gorsel uretimi."""

    def __init__(self, api_key: str | None):
        self.api_key = _require_api_key(api_key)

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        # TODO(faz 4): google-genai SDK ile generate_content; image_path varsa gorseli part olarak ekle.
        raise NotImplementedError("GeminiLLMProvider.generate 4. fazda eklenecek")


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedContent ucu uzerinden vektor uretimi."""

    def __init__(self, api_key: str | None):
        self.api_key = _require_api_key(api_key)

    def embed(self, text: str) -> list[float]:
        # TODO(faz 4): google-genai SDK ile embed_content
        raise NotImplementedError("GeminiEmbeddingProvider.embed 4. fazda eklenecek")

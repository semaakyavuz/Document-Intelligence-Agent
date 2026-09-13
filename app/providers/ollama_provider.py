"""
ollama_provider.py

Yerel Ollama sunucusu uzerinden LLM ve embedding.
Gercek HTTP cagrilari 4. fazda agent'larla birlikte eklenecek; simdilik iskelet.
"""

from app.providers.base import EmbeddingProvider, LLMProvider


class OllamaLLMProvider(LLMProvider):
    """Ollama /api/generate ucu uzerinden metin ve gorsel destekli uretim."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        # TODO(faz 4): POST {base_url}/api/generate; image_path varsa base64'e cevirip "images" alanina ekle.
        raise NotImplementedError("OllamaLLMProvider.generate 4. fazda eklenecek")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama /api/embeddings ucu uzerinden vektor uretimi."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        # TODO(faz 4): POST {base_url}/api/embeddings
        raise NotImplementedError("OllamaEmbeddingProvider.embed 4. fazda eklenecek")

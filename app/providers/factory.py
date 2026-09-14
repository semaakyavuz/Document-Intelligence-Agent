"""
factory.py

.env'deki LLM_PROVIDER / EMBEDDING_PROVIDER degerine gore dogru provider nesnesini kurar.
Agent'lar somut provider siniflarini import etmez, buradan alir.
"""

from collections.abc import Callable

from app.config import Settings
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.gemini_provider import GeminiEmbeddingProvider, GeminiLLMProvider
from app.providers.groq_provider import GroqLLMProvider
from app.providers.ollama_provider import OllamaEmbeddingProvider, OllamaLLMProvider

_LLM_BUILDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "ollama": lambda s: OllamaLLMProvider(
        base_url=s.OLLAMA_BASE_URL,
        model=s.OLLAMA_VISION_MODEL,
        timeout_s=s.OLLAMA_TIMEOUT_S,
        temperature=s.LLM_TEMPERATURE,
    ),
    "groq": lambda s: GroqLLMProvider(api_key=s.GROQ_API_KEY, temperature=s.LLM_TEMPERATURE),
    "gemini": lambda s: GeminiLLMProvider(
        api_key=s.GEMINI_API_KEY, model=s.GEMINI_VISION_MODEL, temperature=s.LLM_TEMPERATURE
    ),
}

_EMBEDDING_BUILDERS: dict[str, Callable[[Settings], EmbeddingProvider]] = {
    "ollama": lambda s: OllamaEmbeddingProvider(base_url=s.OLLAMA_BASE_URL),
    "gemini": lambda s: GeminiEmbeddingProvider(api_key=s.GEMINI_API_KEY),
}


def _build(builders: dict, name: str, settings: Settings, kind: str):
    builder = builders.get(name)
    if builder is None:
        raise ValueError(f"{kind} icin desteklenmeyen saglayici: '{name}'. Gecerli secenekler: {sorted(builders)}")
    return builder(settings)


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """settings verilmezse .env'den okunur; testlerde Settings(_env_file=None) ile beslenebilir."""
    settings = settings or Settings()
    return _build(_LLM_BUILDERS, settings.LLM_PROVIDER, settings, kind="LLM")


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or Settings()
    return _build(_EMBEDDING_BUILDERS, settings.EMBEDDING_PROVIDER, settings, kind="Embedding")

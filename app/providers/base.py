"""
base.py

Saglayici (provider) soyutlamalari. Agent'lar Ollama/Groq/Gemini gibi somut
siniflari degil yalnizca bu arayuzleri tanir; saglayici degisince agent kodu degismez.
"""

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """Tum saglayici hatalarinin atasi. Agent'lar somut Ollama/Groq hatalarini degil bunu yakalar."""


class ProviderUnavailableError(ProviderError):
    """Saglayiciya hic ulasilamiyor (sunucu kapali, yanlis adres). Tekrar denemek anlamsiz."""


class ProviderTimeoutError(ProviderError):
    """Saglayici sure sinirinda cevap vermedi. Diger orneklerle devam edilebilir."""


class ProviderResponseError(ProviderError):
    """Saglayici ulasildi ama hatali/eksik cevap dondu (model yok, gecersiz govde...)."""


class LLMProvider(ABC):
    """Metin (ve istege bagli gorsel) alip metin ureten dil modeli arayuzu."""

    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        """image_path verilirse gorsel destekli (vision) cagri yapilir.

        max_tokens uretilecek token sayisina ust sinir koyar (Ollama: num_predict,
        Gemini: max_output_tokens, Groq: max_tokens). None ise saglayicinin varsayilani gecerlidir.
        """


class EmbeddingProvider(ABC):
    """Metni vektore ceviren arayuz (RAG icin)."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Verilen metnin embedding vektorunu dondurur."""

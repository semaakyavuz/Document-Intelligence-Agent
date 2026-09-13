"""
base.py

Saglayici (provider) soyutlamalari. Agent'lar Ollama/Groq/Gemini gibi somut
siniflari degil yalnizca bu arayuzleri tanir; saglayici degisince agent kodu degismez.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Metin (ve istege bagli gorsel) alip metin ureten dil modeli arayuzu."""

    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None) -> str:
        """image_path verilirse gorsel destekli (vision) cagri yapilir."""


class EmbeddingProvider(ABC):
    """Metni vektore ceviren arayuz (RAG icin)."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Verilen metnin embedding vektorunu dondurur."""

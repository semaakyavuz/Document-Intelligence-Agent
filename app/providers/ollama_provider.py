"""
ollama_provider.py

Yerel Ollama sunucusu uzerinden LLM ve embedding.

OllamaLLMProvider gercek HTTP cagrisi yapar (/api/generate). Gorsel verilirse
dosya base64'e cevrilip "images" alaninda gonderilir; llava gibi gorsel destekli
modeller bunu okur. OllamaEmbeddingProvider /api/embeddings ucunu kullanir (RAG Agent icin).
"""

import base64
import logging
from pathlib import Path

import requests

from app.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaConnectionError(ProviderUnavailableError):
    """Ollama sunucusuna hic ulasilamadi (kapali, yanlis port)."""


class OllamaTimeoutError(ProviderTimeoutError):
    """Ollama sure sinirinda cevap vermedi (CPU'da buyuk model, uzun cikti)."""


class OllamaResponseError(ProviderResponseError):
    """Sunucu ulasildi ama hata dondu (ornegin model yuklu degil)."""


class OllamaLLMProvider(LLMProvider):
    """Ollama /api/generate ucu uzerinden metin ve gorsel destekli uretim."""

    GENERATE_PATH = "/api/generate"

    def __init__(self, base_url: str, model: str, timeout_s: float = 600.0, temperature: float = 0.0):
        # CPU'da llava: gorsel kodlama ~80 sn + 2-3 token/sn uretim; tam bir fatura JSON'u 5 dakikayi asabilir.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        options: dict = {"temperature": self.temperature}
        if max_tokens is not None:
            # Model tekrar dongusune girerse zaman asimina kadar CPU yakmasin; num_predict uretimi keser.
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if image_path is not None:
            payload["images"] = [self._encode_image(image_path)]

        try:
            response = requests.post(self.base_url + self.GENERATE_PATH, json=payload, timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Ollama sunucusuna ulasilamadi ({self.base_url}). "
                "Ollama calisiyor mu? 'ollama serve' ile baslattin mi?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(
                f"Ollama {self.timeout_s:.0f} sn icinde cevap vermedi ({self.base_url}). "
                "CPU'da buyuk model yavas olabilir; .env'de OLLAMA_TIMEOUT_S degerini artirin."
            ) from exc

        if not response.ok:
            raise OllamaResponseError(
                f"Ollama HTTP {response.status_code} dondu: {self._error_text(response)}. "
                f"Model yuklu mu? 'ollama pull {self.model}' ile kontrol edin."
            )

        return self._extract_text(response)

    @classmethod
    def _extract_text(cls, response: requests.Response) -> str:
        """Govdeyi dogrular: JSON olmali, 'response' alani string olmali. Aksi halde OllamaResponseError."""
        try:
            body = response.json()
        except ValueError as exc:  # requests.JSONDecodeError, ValueError'in alt sinifi
            raise OllamaResponseError(f"Ollama JSON olmayan govde dondu: {response.text[:200]!r}") from exc
        if not isinstance(body, dict):
            raise OllamaResponseError(f"Ollama beklenmeyen govde tipi dondu: {type(body).__name__}")
        text = body.get("response")
        if not isinstance(text, str):
            raise OllamaResponseError(
                f"Ollama cevabinda 'response' alani yok ya da string degil: {cls._preview(body)}"
            )
        if not body.get("done", True) or body.get("done_reason") == "length":
            # Model tekrar dongusune girince Ollama uretimi sessizce keser (done=false) ya da
            # num_predict sinirina takilir (done_reason=length). Metin yine doner ama JSON yarim
            # kalmis olabilir. Kaydi tutalim ki degerlendirmede iz surulebilsin.
            logger.warning(
                "Ollama cevabi tamamlanmadan kesildi (done=%s, done_reason=%s, %d karakter)",
                body.get("done"), body.get("done_reason"), len(text),
            )
        return text

    @staticmethod
    def _preview(body: dict) -> str:
        return ", ".join(f"{k}={type(v).__name__}" for k, v in body.items()) or "(bos govde)"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Gorsel bulunamadi: {path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    @staticmethod
    def _error_text(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return body["error"]
        return response.text[:200]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama /api/embeddings ucu uzerinden vektor uretimi."""

    EMBEDDINGS_PATH = "/api/embeddings"
    DEFAULT_TIMEOUT_S = 60.0  # embedding, vision uretiminden cok daha hizli; kisa metinler icin yeterli

    def __init__(self, base_url: str, model: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}

        try:
            response = requests.post(self.base_url + self.EMBEDDINGS_PATH, json=payload, timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Ollama sunucusuna ulasilamadi ({self.base_url}). "
                "Ollama calisiyor mu? 'ollama serve' ile baslattin mi?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(
                f"Ollama {self.timeout_s:.0f} sn icinde cevap vermedi ({self.base_url})."
            ) from exc

        if not response.ok:
            raise OllamaResponseError(
                f"Ollama HTTP {response.status_code} dondu: {self._error_text(response)}. "
                f"Model yuklu mu? 'ollama pull {self.model}' ile kontrol edin."
            )

        return self._extract_embedding(response)

    @classmethod
    def _extract_embedding(cls, response: requests.Response) -> list[float]:
        """Govdeyi dogrular: JSON olmali, 'embedding' alani sayilardan olusan bir liste olmali."""
        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaResponseError(f"Ollama JSON olmayan govde dondu: {response.text[:200]!r}") from exc
        if not isinstance(body, dict):
            raise OllamaResponseError(f"Ollama beklenmeyen govde tipi dondu: {type(body).__name__}")
        embedding = body.get("embedding")
        if not isinstance(embedding, list) or not embedding or not all(isinstance(x, (int, float)) for x in embedding):
            raise OllamaResponseError(
                f"Ollama cevabinda 'embedding' alani yok ya da gecersiz: {OllamaLLMProvider._preview(body)}"
            )
        return embedding

    @staticmethod
    def _error_text(response: requests.Response) -> str:
        return OllamaLLMProvider._error_text(response)

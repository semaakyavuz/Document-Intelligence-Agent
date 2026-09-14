"""
Factory'nin ortam degiskenlerine gore dogru provider sinifini sectigini test eder.

Gercek .env dosyasi okunmaz (_env_file=None); degiskenler monkeypatch ile
sahte olarak verilir. Ollama testlerinde requests.post monkeypatch'lenir,
diger provider'lar stub oldugu icin hicbir test agla cikmaz.
"""

import base64

import pytest
import requests
from pydantic import ValidationError

from app.config import Settings
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.providers.gemini_provider import GeminiEmbeddingProvider, GeminiLLMProvider
from app.providers.groq_provider import GroqLLMProvider
from app.providers.ollama_provider import (
    OllamaConnectionError,
    OllamaEmbeddingProvider,
    OllamaLLMProvider,
    OllamaResponseError,
)


def _settings_from_env(monkeypatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


@pytest.mark.parametrize(
    "provider_name, expected_cls, extra_env",
    [
        ("ollama", OllamaLLMProvider, {}),
        ("groq", GroqLLMProvider, {"GROQ_API_KEY": "test-key"}),
        ("gemini", GeminiLLMProvider, {"GEMINI_API_KEY": "test-key"}),
    ],
)
def test_get_llm_provider_selects_class_from_env(monkeypatch, provider_name, expected_cls, extra_env):
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER=provider_name, **extra_env)
    assert isinstance(get_llm_provider(settings), expected_cls)


@pytest.mark.parametrize(
    "provider_name, expected_cls, extra_env",
    [
        ("ollama", OllamaEmbeddingProvider, {}),
        ("gemini", GeminiEmbeddingProvider, {"GEMINI_API_KEY": "test-key"}),
    ],
)
def test_get_embedding_provider_selects_class_from_env(monkeypatch, provider_name, expected_cls, extra_env):
    settings = _settings_from_env(monkeypatch, EMBEDDING_PROVIDER=provider_name, **extra_env)
    assert isinstance(get_embedding_provider(settings), expected_cls)


def test_groq_has_no_embedding_provider(monkeypatch):
    settings = _settings_from_env(monkeypatch, EMBEDDING_PROVIDER="groq")
    with pytest.raises(ValueError, match="groq"):
        get_embedding_provider(settings)


def test_unknown_provider_name_is_rejected_by_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_cloud_provider_requires_api_key(monkeypatch):
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER="groq", GROQ_API_KEY="")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm_provider(settings)


def test_stub_provider_raises_instead_of_calling_network(monkeypatch):
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key")
    with pytest.raises(NotImplementedError):
        get_llm_provider(settings).generate("merhaba")


def test_ollama_provider_reads_model_from_settings(monkeypatch):
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_VISION_MODEL="llava:13b")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OllamaLLMProvider)
    assert provider.model == "llava:13b"


def test_ollama_connection_error_gives_actionable_message(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaLLMProvider(base_url="http://localhost:1", model="llava")
    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        provider.generate("merhaba")


class FakeResponse:
    """requests.Response'un testlerde ihtiyac duyulan kucuk bir yerine gecicisi."""

    def __init__(self, ok=True, status_code=200, body=None, raw_text="", raises_on_json=False):
        self.ok = ok
        self.status_code = status_code
        self.text = raw_text
        self._body = body
        self._raises_on_json = raises_on_json

    def json(self):
        if self._raises_on_json:
            raise requests.exceptions.JSONDecodeError("beklenmeyen govde", self.text, 0)
        return self._body


def test_ollama_sends_base64_image_and_model(monkeypatch, tmp_path):
    image = tmp_path / "fatura.png"
    image.write_bytes(b"\x89PNG fake")
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse(body={"response": '{"invoice_no": "1"}'})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaLLMProvider(base_url="http://localhost:11434/", model="llava")
    text = provider.generate("oku", image_path=str(image))

    assert text == '{"invoice_no": "1"}'
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"]["model"] == "llava"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["images"] == [base64.b64encode(b"\x89PNG fake").decode("ascii")]
    assert "num_predict" not in captured["payload"]["options"]  # max_tokens verilmedi


def test_ollama_max_tokens_becomes_num_predict(monkeypatch):
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured.update(payload=json)
        return FakeResponse(body={"response": "{}"})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    provider.generate("oku", max_tokens=1024)

    assert captured["payload"]["options"]["num_predict"] == 1024


def test_ollama_non_json_body_raises_response_error(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: FakeResponse(raises_on_json=True, raw_text="<html>502 Bad Gateway</html>"),
    )
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    with pytest.raises(OllamaResponseError, match="JSON olmayan"):
        provider.generate("oku")


def test_ollama_body_not_a_dict_raises_response_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body=["beklenmedik", "liste"]))
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    with pytest.raises(OllamaResponseError, match="beklenmeyen govde tipi"):
        provider.generate("oku")


def test_ollama_missing_response_field_raises_response_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body={"done": True}))
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    with pytest.raises(OllamaResponseError, match="'response' alani"):
        provider.generate("oku")


def test_ollama_non_string_response_field_raises_response_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body={"response": None}))
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    with pytest.raises(OllamaResponseError, match="'response' alani"):
        provider.generate("oku")


def test_ollama_http_error_status_raises_response_error(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: FakeResponse(ok=False, status_code=404, body={"error": "model 'llava' not found"}),
    )
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    with pytest.raises(OllamaResponseError, match="not found"):
        provider.generate("oku")


def test_ollama_truncated_response_logs_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: FakeResponse(body={"response": '{"invoice_no": "111', "done": False}),
    )
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="llava")
    with caplog.at_level("WARNING"):
        text = provider.generate("oku")

    assert text == '{"invoice_no": "111'  # yarim da olsa metin kaybedilmez, ust katman (parser) karar verir
    assert "kesildi" in caplog.text

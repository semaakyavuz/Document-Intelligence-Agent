"""
Factory'nin ortam degiskenlerine gore dogru provider sinifini sectigini test eder.

Gercek .env dosyasi okunmaz (_env_file=None); degiskenler monkeypatch ile
sahte olarak verilir. Stub provider'lar NotImplementedError firlattigi icin
hicbir test agla cikmaz.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.providers.gemini_provider import GeminiEmbeddingProvider, GeminiLLMProvider
from app.providers.groq_provider import GroqLLMProvider
from app.providers.ollama_provider import OllamaEmbeddingProvider, OllamaLLMProvider


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
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER="ollama")
    with pytest.raises(NotImplementedError):
        get_llm_provider(settings).generate("merhaba")

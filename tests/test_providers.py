"""
Factory'nin ortam degiskenlerine gore dogru provider sinifini sectigini test eder.

Gercek .env dosyasi okunmaz (_env_file=None); degiskenler monkeypatch ile
sahte olarak verilir. Ollama testlerinde requests.post, Gemini testlerinde
provider._client (google-genai Client'in yerine gecen sahte nesne) monkeypatch'lenir;
Groq hala stub oldugu icin o testler NotImplementedError bekler. Hicbir test agla cikmaz.
"""

import base64

import httpx
import pytest
import requests
from google.genai import errors as genai_errors
from PIL import Image as PILImage
from pydantic import ValidationError

from app.config import Settings
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.providers.gemini_provider import (
    GeminiConnectionError,
    GeminiEmbeddingProvider,
    GeminiLLMProvider,
    GeminiRateLimitError,
    GeminiResponseError,
    GeminiTimeoutError,
)
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
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER="groq", GROQ_API_KEY="test-key")
    with pytest.raises(NotImplementedError):
        get_llm_provider(settings).generate("merhaba")


def test_ollama_provider_reads_model_from_settings(monkeypatch):
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_VISION_MODEL="llava:13b")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OllamaLLMProvider)
    assert provider.model == "llava:13b"


@pytest.mark.parametrize(
    "provider_name, extra_env",
    [
        ("ollama", {}),
        ("groq", {"GROQ_API_KEY": "test-key"}),
        ("gemini", {"GEMINI_API_KEY": "test-key"}),
    ],
)
def test_llm_temperature_flows_from_settings_through_factory_to_every_provider(monkeypatch, provider_name, extra_env):
    """temperature artik hicbir provider'a gomulu degil; ucu de factory'den, tek bir
    Settings.LLM_TEMPERATURE alanindan aliyor (provider Settings'i kendisi okumuyor)."""
    settings = _settings_from_env(monkeypatch, LLM_PROVIDER=provider_name, LLM_TEMPERATURE="0.42", **extra_env)
    assert get_llm_provider(settings).temperature == pytest.approx(0.42)


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


def test_ollama_temperature_is_configurable(monkeypatch):
    """temperature artik provider'a gomulu sabit degil; Settings.LLM_TEMPERATURE'dan gelir."""
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured.update(payload=json)
        return FakeResponse(body={"response": "{}"})

    monkeypatch.setattr(requests, "post", fake_post)
    OllamaLLMProvider(base_url="http://localhost:11434", model="llava").generate("oku")
    assert captured["payload"]["options"]["temperature"] == 0.0  # varsayilan

    OllamaLLMProvider(base_url="http://localhost:11434", model="llava", temperature=0.7).generate("oku")
    assert captured["payload"]["options"]["temperature"] == 0.7


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


# --- OllamaEmbeddingProvider --------------------------------------------------

def test_ollama_embedding_provider_reads_model_from_settings(monkeypatch):
    settings = _settings_from_env(monkeypatch, EMBEDDING_PROVIDER="ollama", OLLAMA_EMBEDDING_MODEL="nomic-embed-text")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model == "nomic-embed-text"


def test_ollama_embed_sends_model_and_prompt_returns_vector(monkeypatch):
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, payload=json)
        return FakeResponse(body={"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
    vector = provider.embed("kural metni")

    assert vector == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://localhost:11434/api/embeddings"
    assert captured["payload"] == {"model": "nomic-embed-text", "prompt": "kural metni"}


def test_ollama_embed_connection_error_gives_actionable_message(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError()))
    provider = OllamaEmbeddingProvider(base_url="http://localhost:1", model="nomic-embed-text")
    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        provider.embed("kural metni")


def test_ollama_embed_non_json_body_raises_response_error(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: FakeResponse(raises_on_json=True, raw_text="<html>502 Bad Gateway</html>"),
    )
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
    with pytest.raises(OllamaResponseError, match="JSON olmayan"):
        provider.embed("kural metni")


@pytest.mark.parametrize(
    "body",
    [
        {"done": True},  # 'embedding' alani hic yok
        {"embedding": []},  # bos liste
        {"embedding": "bozuk"},  # liste degil
        {"embedding": [0.1, "iki", 0.3]},  # sayisal olmayan eleman
    ],
)
def test_ollama_embed_invalid_embedding_field_raises_response_error(monkeypatch, body):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body=body))
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
    with pytest.raises(OllamaResponseError, match="embedding"):
        provider.embed("kural metni")


def test_ollama_embed_http_error_status_raises_response_error(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: FakeResponse(ok=False, status_code=404, body={"error": "model 'nomic-embed-text' not found"}),
    )
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
    with pytest.raises(OllamaResponseError, match="not found"):
        provider.embed("kural metni")


# --- GeminiLLMProvider -------------------------------------------------------
#
# google-genai'nin gercek Client'ina baglanilmaz; provider._client sahte bir
# nesneyle degistirilir. Bu, requests.post monkeypatch'iyle ayni fikri
# (saglayicinin cagirdigi son sinira mudahale et) Gemini'nin istemci-nesnesi
# tabanli SDK'sina uyarlar.

class FakeGeminiModels:
    """Client().models yerine gecer; behavior bir Exception ise firlatir, degilse dondurur."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text
        self.prompt_feedback = None


@pytest.fixture(scope="module")
def gemini_provider() -> GeminiLLMProvider:
    """genai.Client(...) kurulumu tek basina ~1.5 sn suruyor (google-genai'nin kendi maliyeti,
    agla ilgisi yok). Testler arasinda tek bir GeminiLLMProvider paylasilir; her test yalnizca
    provider._client'i kendi sahte davranisiyla degistirir, provider'i yeniden kurmaz."""
    return GeminiLLMProvider(api_key="test-key", model="gemini-3.6-flash")


def _use_fake_client(provider: GeminiLLMProvider, behavior) -> FakeGeminiModels:
    fake_models = FakeGeminiModels(behavior)
    provider._client = type("FakeClient", (), {"models": fake_models})()
    return fake_models


def test_gemini_sends_prompt_and_image(gemini_provider, tmp_path):
    image = tmp_path / "fatura.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\nIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    fake_models = _use_fake_client(gemini_provider, FakeGeminiResponse(text='{"invoice_no": "1"}'))

    text = gemini_provider.generate("faturayi oku", image_path=str(image))

    assert text == '{"invoice_no": "1"}'
    call = fake_models.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["contents"][0] == "faturayi oku"
    assert isinstance(call["contents"][1], PILImage.Image)  # gorsel gercekten acilip eklendi
    assert call["config"].temperature == 0
    assert call["config"].max_output_tokens is None  # max_tokens verilmedi


def test_gemini_max_tokens_becomes_max_output_tokens(gemini_provider):
    fake_models = _use_fake_client(gemini_provider, FakeGeminiResponse(text="{}"))
    gemini_provider.generate("oku", max_tokens=1024)
    assert fake_models.calls[0]["config"].max_output_tokens == 1024


def test_gemini_missing_image_raises_before_calling_api(gemini_provider, tmp_path):
    fake_models = _use_fake_client(gemini_provider, FakeGeminiResponse(text="{}"))
    with pytest.raises(FileNotFoundError, match="olmayan"):
        gemini_provider.generate("oku", image_path=str(tmp_path / "olmayan.png"))
    assert fake_models.calls == []  # dosya yoksa API'ye hic gidilmez


def test_gemini_corrupted_image_raises_response_error_not_raw_exception(gemini_provider, tmp_path):
    """Dosya var ama gorsel olarak acilamiyor (bozuk/yarim indirilmis/yanlis format).
    FileNotFoundError'dan farkli: bu durumda ProviderError ailesine sarilmali ki
    VisionAgentEvaluator tek bir bozuk gorselde tum kosuyu cokertmesin."""
    bad_image = tmp_path / "bozuk.png"
    bad_image.write_bytes(b"bu bir gorsel dosyasi degil")
    fake_models = _use_fake_client(gemini_provider, FakeGeminiResponse(text="{}"))

    with pytest.raises(GeminiResponseError, match="acilamadi"):
        gemini_provider.generate("oku", image_path=str(bad_image))
    assert fake_models.calls == []  # gorsel acilamadiysa API'ye hic gidilmez


def test_gemini_rate_limit_raises_specific_error(gemini_provider):
    error = genai_errors.ClientError(
        code=429, response_json={"error": {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}
    )
    _use_fake_client(gemini_provider, error)
    with pytest.raises(GeminiRateLimitError, match="429"):
        gemini_provider.generate("oku")


def test_gemini_other_api_error_raises_response_error_not_rate_limit(gemini_provider):
    error = genai_errors.ServerError(code=500, response_json={"error": {"message": "internal", "status": "INTERNAL"}})
    _use_fake_client(gemini_provider, error)
    with pytest.raises(GeminiResponseError) as excinfo:
        gemini_provider.generate("oku")
    assert not isinstance(excinfo.value, GeminiRateLimitError)


def test_gemini_api_error_with_no_code_is_not_mistaken_for_rate_limit(gemini_provider):
    """exc.code None donebilir (SDK durumu response_json'dan cikaramazsa). 'exc.code == 429'
    kontrolu bu durumda sessizce False olmali, genel GeminiResponseError'a dusmeli."""
    error = genai_errors.ServerError(code=None, response_json={"error": {"message": "bilinmiyor"}})
    assert error.code is None  # varsayim gecerli mi, once dogrula
    _use_fake_client(gemini_provider, error)
    with pytest.raises(GeminiResponseError) as excinfo:
        gemini_provider.generate("oku")
    assert not isinstance(excinfo.value, GeminiRateLimitError)


def test_gemini_connect_error_raises_connection_error(gemini_provider):
    _use_fake_client(gemini_provider, httpx.ConnectError("refused"))
    with pytest.raises(GeminiConnectionError):
        gemini_provider.generate("oku")


def test_gemini_timeout_raises_timeout_error(gemini_provider):
    _use_fake_client(gemini_provider, httpx.ReadTimeout("timed out"))
    with pytest.raises(GeminiTimeoutError, match="120"):
        gemini_provider.generate("oku")


def test_gemini_empty_text_response_raises_response_error(gemini_provider):
    _use_fake_client(gemini_provider, FakeGeminiResponse(text=None))
    with pytest.raises(GeminiResponseError, match="bos"):
        gemini_provider.generate("oku")


# --- GeminiLLMProvider: thinking_config fallback ------------------------------
#
# gemini-3.6-flash thinking_level=MINIMAL'i kabul ediyordu ama thinking_budget=0'i HTTP 400
# ile reddediyordu (canli dogrulandi). Farkli bir GEMINI_VISION_MODEL, thinking_config'in
# KENDISINI de reddedebilir; bu durumda _call_with_thinking_fallback thinking olmadan bir
# kez daha dener ve ogrendigini (provider._thinking_supported) hatirlar.

class SequencedFakeGeminiModels:
    """Client().models yerine gecer; her cagrida behaviors listesinden bir sonrakini kullanir
    (ilk cagri farkli, fallback sonrasi ikinci cagri farkli davransin diye)."""

    def __init__(self, behaviors: list):
        self.behaviors = list(behaviors)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def _use_fake_client_sequence(provider: GeminiLLMProvider, behaviors: list) -> SequencedFakeGeminiModels:
    fake_models = SequencedFakeGeminiModels(behaviors)
    provider._client = type("FakeClient", (), {"models": fake_models})()
    return fake_models


def _thinking_rejected_error() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code=400, response_json={"error": {"message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}}
    )


def test_gemini_falls_back_when_model_rejects_thinking_config(gemini_provider):
    gemini_provider._thinking_supported = None  # bilinmeyen durumdan basla (test izolasyonu)
    fake_models = _use_fake_client_sequence(gemini_provider, [_thinking_rejected_error(), FakeGeminiResponse(text="{}")])

    text = gemini_provider.generate("oku")

    assert text == "{}"
    assert len(fake_models.calls) == 2
    assert fake_models.calls[0]["config"].thinking_config is not None  # ilk deneme thinking ile
    assert fake_models.calls[1]["config"].thinking_config is None  # fallback thinking'siz
    assert gemini_provider._thinking_supported is False


def test_gemini_remembers_thinking_unsupported_and_skips_retry_next_call(gemini_provider):
    gemini_provider._thinking_supported = False  # onceki bir cagridan ogrenilmis gibi
    fake_models = _use_fake_client_sequence(gemini_provider, [FakeGeminiResponse(text="{}")])

    text = gemini_provider.generate("oku")

    assert text == "{}"
    assert len(fake_models.calls) == 1  # thinking'li ilk deneme hic yapilmadi, bosuna cagri yok
    assert fake_models.calls[0]["config"].thinking_config is None


def test_gemini_remembers_thinking_supported(gemini_provider):
    gemini_provider._thinking_supported = None
    fake_models = _use_fake_client_sequence(gemini_provider, [FakeGeminiResponse(text="{}")])

    gemini_provider.generate("oku")

    assert gemini_provider._thinking_supported is True
    assert fake_models.calls[0]["config"].thinking_config is not None


def test_gemini_rate_limit_on_first_attempt_is_not_treated_as_thinking_issue(gemini_provider):
    gemini_provider._thinking_supported = None
    error = genai_errors.ClientError(code=429, response_json={"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    fake_models = _use_fake_client_sequence(gemini_provider, [error])

    with pytest.raises(GeminiRateLimitError):
        gemini_provider.generate("oku")

    assert len(fake_models.calls) == 1  # fallback denenmedi, dogrudan rate-limit hatasi verildi
    assert gemini_provider._thinking_supported is None  # ogrenilen bir sey yok, sebep 400 degildi


def test_gemini_provider_reads_model_and_timeout_from_settings(monkeypatch):
    settings = _settings_from_env(
        monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key", GEMINI_VISION_MODEL="gemini-3.7-flash"
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.model == "gemini-3.7-flash"


def test_gemini_embedding_provider_is_still_a_stub(monkeypatch):
    settings = _settings_from_env(monkeypatch, EMBEDDING_PROVIDER="gemini", GEMINI_API_KEY="test-key")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, GeminiEmbeddingProvider)
    with pytest.raises(NotImplementedError):
        provider.embed("merhaba")

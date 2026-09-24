"""
gemini_provider.py

Google Gemini API'si uzerinden LLM ve embedding.

GeminiLLMProvider gercek API cagrisi yapar (google-genai SDK). google-generativeai
paketi artik legacy/deprecated oldugu icin (Google'in kendi tavsiyesiyle) onun
yerine aktif gelistirilen google-genai kullanilir.

GeminiEmbeddingProvider bulut dagitimi (Render) icin gerekli: orada Ollama yok,
embedding de Gemini'den geliyor. Toplu cagriyi destekler - bilgi tabaninin tamami
tek istekte embed edilir.
"""

import logging
import math
from contextlib import contextmanager
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types
from PIL import Image

logger = logging.getLogger(__name__)

from app.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def _require_api_key(api_key: str | None) -> str:
    if not api_key:
        raise ValueError("GEMINI_API_KEY tanimli degil; .env dosyasina ekleyin")
    return api_key


class GeminiConnectionError(ProviderUnavailableError):
    """Gemini sunucusuna hic ulasilamadi (ag yok, DNS, TCP reddedildi)."""


class GeminiTimeoutError(ProviderTimeoutError):
    """Gemini sure sinirinda cevap vermedi."""


class GeminiResponseError(ProviderResponseError):
    """Sunucu ulasildi ama hata dondu ya da cevap beklenmeyen sekildeydi."""


class GeminiRateLimitError(GeminiResponseError):
    """API kotasi asildi (HTTP 429). GeminiResponseError'un ozel bir durumu; ayri
    yakalanabilir ki ileride burada bekle-ve-tekrar-dene (backoff) eklenebilsin."""


@contextmanager
def _mapped_gemini_errors(model: str, timeout_s: float, rate_limit_hint: str):
    """google-genai / httpx hatalarini ProviderError ailesine cevirir.

    generate() ve embed() ayni eslemeyi paylassin diye tek yerde toplandi -
    boylece iki ucun hata davranisi kendiliginden simetrik kaliyor (Ollama
    tarafindaki _post_json'in ustlendigi role denk)."""
    try:
        yield
    except httpx.TimeoutException as exc:
        raise GeminiTimeoutError(
            f"Gemini {timeout_s:.0f} sn icinde cevap vermedi (model={model})."
        ) from exc
    except httpx.TransportError as exc:
        raise GeminiConnectionError(
            f"Gemini API'sine ulasilamadi: {exc}. Internet baglantisini kontrol edin."
        ) from exc
    except errors.APIError as exc:
        if exc.code == 429:
            raise GeminiRateLimitError(
                f"Gemini API rate limit'e takildi (429): {exc.message}. {rate_limit_hint}"
            ) from exc
        raise GeminiResponseError(f"Gemini HTTP {exc.code} ({exc.status}): {exc.message}") from exc


class GeminiLLMProvider(LLMProvider):
    """Gemini generateContent ucu uzerinden metin/gorsel destekli uretim."""

    # Bulut API'si oldugu icin yerel Ollama'dan cok daha hizli yanit verir;
    # 120 sn buyuk bir gorsel + uzun bir JSON cikti icin dahi yeterli olmali.
    DEFAULT_TIMEOUT_S = 120.0

    def __init__(
        self, api_key: str | None, model: str, timeout_s: float = DEFAULT_TIMEOUT_S, temperature: float = 0.0
    ):
        self.api_key = _require_api_key(api_key)
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        # None: henuz bilinmiyor (ilk cagrida ogrenilir). Bazi modeller thinking_config'i
        # tanimiyor/reddediyor (bkz. _call_with_thinking_fallback); ogrenilince burada tutulur
        # ki her generate() cagrisinda bosuna basarisiz bir deneme yapilmasin.
        self._thinking_supported: bool | None = None
        # Client tek seferde kurulur (baglanti/kimlik dogrulama her generate() cagrisinda tekrarlanmaz).
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # ms bekliyor
        )

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        contents: list = [prompt]
        if image_path is not None:
            contents.append(self._load_image(image_path))

        with _mapped_gemini_errors(
            self.model, self.timeout_s,
            "Bir sure bekleyip tekrar deneyin ya da GEMINI_VISION_MODEL'i degistirin.",
        ):
            response = self._call_with_thinking_fallback(contents, max_tokens)

        return self._extract_text(response)

    def _call_with_thinking_fallback(self, contents: list, max_tokens: int | None):
        """thinking_config icerecek sekilde cagirir; model bunu taniyip taramiyorsa (HTTP 400)
        thinking_config'siz tekrar dener ve sonucu bu instance icin hatirlar.

        Neden gerekli: gemini-3.6-flash'ta thinking_budget=0 acikca reddediliyordu,
        thinking_level=MINIMAL ise calisiyordu (bkz. modul ust bilgisi). Ama her Gemini
        modelinin thinking_config'i ayni sekilde destekleyecegi garanti degil; farkli bir
        GEMINI_VISION_MODEL'e gecilince thinking_config'in kendisi de reddedilebilir. Bu
        durumda tek seferlik bir HTTP 400'u tolere edip thinking olmadan devam ederiz.
        """
        if self._thinking_supported is not False:
            try:
                config = self._build_generation_config(max_tokens, include_thinking=True)
                response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                self._thinking_supported = True
                return response
            except errors.ClientError as exc:
                if exc.code != 400:
                    raise  # 429 (rate limit) vb. burada ele alinmaz, generate()'e yukari cikar
                logger.warning(
                    "%s modeli thinking_config'i reddetti (HTTP 400); thinking olmadan tekrar deneniyor.",
                    self.model,
                )
                self._thinking_supported = False

        config = self._build_generation_config(max_tokens, include_thinking=False)
        return self._client.models.generate_content(model=self.model, contents=contents, config=config)

    def _build_generation_config(self, max_tokens: int | None, include_thinking: bool) -> types.GenerateContentConfig:
        """Bu saglayicinin her cagrida kullandigi sabit uretim ayarlari (debug script'i de
        aynen bunu cagirir; ayarlarin iki yerde ayri ayri tutulup birbirinden sapmasini onler)."""
        thinking_config = None
        if include_thinking:
            # Yapisal veri cikarma karmasik muhakeme gerektirmiyor; "thinking" butcesi
            # max_output_tokens'i gorunmeyen ara-dusunce token'lariyla paylasip goruntu
            # cevabini erken kesiyordu (bkz. scripts/debug_single_extraction.py bulgusu:
            # finish_reason=MAX_TOKENS, thoughts_token_count=982).
            #
            # thinking_budget=0 (SDK dokumantasyonunda "0 is DISABLED" der) hem gemini-3.6-flash
            # hem gemini-3.5-flash-lite'ta canlica denendi ve ikisinde de HTTP 400 INVALID_ARGUMENT
            # ile reddedildi; thinking_level=MINIMAL ise ikisinde de calisiyor ve
            # thoughts_token_count=None (olcumsuz/en az) sonucunu veriyor. Yine de her model
            # thinking_config'i taniyacak diye bir garanti yok; taramayan bir model icin
            # _call_with_thinking_fallback bu ayari otomatik olarak atlar.
            thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_tokens,
            thinking_config=thinking_config,
            # Modelin metin etrafina aciklama/kod citi eklemeden dogrudan JSON dondurmesini zorunlu kilar.
            response_mime_type="application/json",
        )

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        text = response.text
        if not isinstance(text, str) or not text:
            # response.text; guvenlik filtresi tarafindan engellenen ya da yalnizca
            # metin-disi parca iceren cevaplarda None doner.
            reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
            raise GeminiResponseError(f"Gemini bos/metin olmayan cevap dondu (block_reason={reason})")
        return text

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Gorsel bulunamadi: {path}")
        try:
            return Image.open(path)
        except OSError as exc:
            # PIL.UnidentifiedImageError (OSError'un alt sinifi) dahil: dosya var ama
            # gorsel olarak acilamiyor (bozuk/yarim indirilmis/yanlis format). Bu, tek bir
            # gorselin degerlendirme kosusunun tamamini cokertmemesi icin ProviderError
            # ailesine sarilir; VisionAgentEvaluator bunu diger saglayici hatalari gibi yakalar.
            raise GeminiResponseError(f"Gorsel acilamadi (bozuk/gecersiz dosya): {path} ({exc})") from exc


def _l2_normalize(vector: list[float], model: str) -> list[float]:
    """Vektoru birim uzunluga getirir.

    Gerekli, cunku Gemini yalnizca varsayilan 3072 boyutlu cikti icin normalize
    vektor donduruyor; output_dimensionality ile kisilan ciktilar normalize DEGIL
    (olculdu: 3072 -> |v|=1.0000, 768 -> |v|=0.5839). Normalize etmezsek kosinus
    benzerligi bozulur ve RAG alakasiz kurallar getirir."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise GeminiResponseError(f"Gemini sifir uzunlukta embedding vektoru dondurdu (model={model}).")
    return [x / norm for x in vector]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedContent ucu uzerinden vektor uretimi."""

    # Embedding, vision uretiminden cok daha hizli (olcum: ~0.5 sn), 60 sn fazlasiyla yeterli.
    DEFAULT_TIMEOUT_S = 60.0
    RATE_LIMIT_HINT = "Bir sure bekleyip tekrar deneyin ya da GEMINI_EMBEDDING_MODEL'i degistirin."

    def __init__(
        self, api_key: str | None, model: str, output_dim: int, timeout_s: float = DEFAULT_TIMEOUT_S
    ):
        self.api_key = _require_api_key(api_key)
        self.model = model
        self.output_dim = output_dim
        self.timeout_s = timeout_s
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # ms bekliyor
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Tum metinleri TEK API cagrisinda embed eder.

        Render gibi diski gecici olan ortamlarda uygulama her uyandiginda bilgi
        tabani yeniden indeksleniyor; toplu cagri bunun kota maliyetini dokuman
        sayisi kadar (12 dokuman -> 12 istek yerine 1 istek) azaltiyor."""
        if not texts:
            return []

        with _mapped_gemini_errors(self.model, self.timeout_s, self.RATE_LIMIT_HINT):
            response = self._client.models.embed_content(
                model=self.model,
                contents=list(texts),
                config=types.EmbedContentConfig(output_dimensionality=self.output_dim),
            )

        embeddings = getattr(response, "embeddings", None)
        # Uzunluk kontrolu sart: bazi modeller (orn. gemini-embedding-2) coklu metin
        # verildiginde sessizce TEK vektor donduruyor. Bu kontrol olmasa tum bilgi
        # tabani yanlis indekslenir ve hicbir hata gorunmezdi.
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            got = len(embeddings) if isinstance(embeddings, list) else type(embeddings).__name__
            raise GeminiResponseError(
                f"Gemini {len(texts)} metin icin {got} embedding dondurdu (model={self.model})."
            )
        return [_l2_normalize(self._vector_of(e), self.model) for e in embeddings]

    def _vector_of(self, embedding) -> list[float]:
        values = getattr(embedding, "values", None)
        if not isinstance(values, list) or not values or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in values
        ):
            raise GeminiResponseError(
                f"Gemini gecersiz embedding vektoru dondurdu (model={self.model}): {str(values)[:120]}"
            )
        if len(values) != self.output_dim:
            raise GeminiResponseError(
                f"Gemini {self.output_dim} boyut istendi ama {len(values)} boyut dondurdu (model={self.model})."
            )
        return values

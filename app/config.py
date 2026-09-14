"""
config.py

Uygulama ayarlarini .env dosyasindan ve ortam degiskenlerinden okur.
Kodun baska hicbir yerinde os.environ okunmaz; her ayar Settings uzerinden gelir.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "groq", "gemini"]


class Settings(BaseSettings):
    """Ortam degiskenleri. .env'de olmayan alanlar varsayilan degeriyle gelir."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    LLM_PROVIDER: ProviderName = "ollama"
    EMBEDDING_PROVIDER: ProviderName = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_VISION_MODEL: str = "llava"  # gorsel okuyabilen bir model olmali (ollama list ile kontrol edin)
    OLLAMA_TIMEOUT_S: float = 600  # CPU'da gorsel model bir fatura icin 5 dakikayi asabilir
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GEMINI_VISION_MODEL: str = "gemini-3.6-flash"  # goruntu destekli Gemini modeli
    LLM_TEMPERATURE: float = 0.0  # tum saglayicilar icin ortak: 0 = ayni girdi -> ayni cikti (tekrarlanabilir sonuc)
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"  # RAG icin embedding modeli (ollama pull nomic-embed-text)
    CHROMA_DB_PATH: str = "data/chroma_db"  # yerel, persist edilen vektor veritabani klasoru
    CHROMA_COLLECTION_NAME: str = "invoice_rules"
    RAG_TOP_K: int = 3  # RAGAgent'in getirecegi en alakali kural sayisi
    DATABASE_URL: str = "sqlite:///./data/app.db"

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
    # goruntu destekli Gemini modeli. gemini-3.6-flash gunde sadece 20 istek veriyordu
    # (ucretsiz katman kotasi); flash-lite ~500/gun veriyor ve ayri bir kota havuzu kullaniyor.
    GEMINI_VISION_MODEL: str = "gemini-3.5-flash-lite"
    LLM_TEMPERATURE: float = 0.0  # tum saglayicilar icin ortak: 0 = ayni girdi -> ayni cikti (tekrarlanabilir sonuc)
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"  # RAG icin embedding modeli (ollama pull nomic-embed-text)
    CHROMA_DB_PATH: str = "data/chroma_db"  # yerel, persist edilen vektor veritabani klasoru
    CHROMA_COLLECTION_NAME: str = "invoice_rules"
    RAG_TOP_K: int = 3  # RAGAgent'in getirecegi en alakali kural sayisi

    # docker-compose.yml'deki postgres servisiyle eslesir (bkz. o dosyadaki container_name:
    # document-intelligence-postgres). Host portu 5434: bu makinede zaten baska bir projeden
    # kalma "my-postgres" adli container 5433'u kullaniyor, catismamasi icin 5434 secildi.
    POSTGRES_USER: str = "invoice_app"
    POSTGRES_PASSWORD: str = "invoice_app"
    POSTGRES_DB: str = "invoice_agent"
    # Not: .env'deki DATABASE_URL, ${POSTGRES_USER} vb. ile yukaridaki uc alana referans
    # verir (python-dotenv interpolasyonu, canlica dogrulandi) - sifre tek yerde tutulur.
    # Buradaki Python varsayilani interpolasyona girmez (sadece .env dosyasindan okunan
    # degerler interpolate edilir); bu yuzden .env hic yoksa diye kendi basina gecerli,
    # yukaridaki varsayilanlarla eslesen duz bir URL olarak yazildi.
    DATABASE_URL: str = "postgresql+psycopg://invoice_app:invoice_app@localhost:5434/invoice_agent"

"""
config.py

Uygulama ayarlarini .env dosyasindan ve ortam degiskenlerinden okur.
Kodun baska hicbir yerinde os.environ okunmaz; her ayar Settings uzerinden gelir.
"""

from typing import Literal

from pydantic import Field
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

    # Bulutta (Render) Ollama yok, embedding Gemini'den geliyor.
    # gemini-embedding-001 bilincli secim: API'de embedContent destekleyen uc modeli de
    # denedim, ucu de calisiyor - ama gemini-embedding-2 coklu metin verildiginde TEK
    # vektor donduruyor (12 dokuman -> 1 vektor), yani toplu indeksleme sessizce bozuk
    # olurdu. 001 ise 12 dokuman icin 12 vektor donduruyor.
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    # 768: Ollama nomic-embed-text ile AYNI boyut, boylece iki saglayici ayni
    # ChromaDB koleksiyon semasini paylasir (boyut uyusmazligi hatasi olmaz).
    # Not: Gemini 3072'nin altina kisilan vektorleri normalize ETMIYOR (olculdu:
    # 3072 -> |v|=1.0000, 768 -> |v|=0.5839); provider kendisi normalize ediyor.
    GEMINI_EMBEDDING_DIM: int = 768
    CHROMA_DB_PATH: str = "data/chroma_db"  # yerel, persist edilen vektor veritabani klasoru
    CHROMA_COLLECTION_NAME: str = "invoice_rules"
    RAG_TOP_K: int = 3  # RAGAgent'in getirecegi en alakali kural sayisi

    # ValidationAgent.FormatCheck'in invoice_no/seller_tax_no'yu nasil denetleyecegi:
    # strict_tr: tam 10 haneli sayisal format zorunlu (sadece Turk formati sentetik veride).
    # lenient: sadece alan bos/anlamsiz degil mi bakar, format dayatmaz (varsayilan -
    # gercek dunya/karma faturalarda ve public demo'da yabanci formatlari yanlislikla
    # anomali diye isaretlememek icin).
    FORMAT_PROFILE: Literal["strict_tr", "lenient"] = "lenient"

    # Ayni anda kac fatura pipeline'i calisabilir (app/api/main.py'deki semafor).
    # Varsayilan 2, Render'in 512 MB plani icin olculdu: temel kullanim ~159 MiB,
    # ucus halindeki her istek MCP server'i AYRI bir surec olarak baslatiyor
    # (bkz. rag_agent.py) ve ~119 MB daha getiriyor. Olculen tepe degerler:
    #   sinir 2 -> 3 es zamanli istek 283 MiB, 4 es zamanli 386 MiB, OOM yok
    #   sinirsiz -> 3 es zamanli 477 MiB (kil payi), 4 es zamanli OOMKilled
    # Sinirin ustundeki istekler reddedilmez, kuyrukta bekler ve SSE'de "queued"
    # olayi alir. Daha buyuk bir plana gecerken bu degeri artirmak yeterli.
    # ge=1: 0 verilse her istek sonsuza kadar beklerdi; .env bir dis girdi oldugu
    # icin deger burada, sunucu acilmadan once dogrulaniyor.
    MAX_CONCURRENT_PIPELINES: int = Field(default=2, ge=1)

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

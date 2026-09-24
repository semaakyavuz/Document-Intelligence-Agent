"""
rag_index.py

Bilgi tabanini (data/knowledge_base/*.txt) embed edip ChromaDB'ye yazar; ayrica
mevcut koleksiyonun SU ANKI embedding saglayicisiyla uyumlu olup olmadigini denetler.

Neden app/ altinda (scripts/ altinda degil): bulut imaji yalnizca runtime
bagimliliklarini kuruyor ve uygulama acilista bu modulu cagirmak zorunda - Render'in
diski gecici oldugu icin koleksiyon her uyanista yeniden kurulmali.

NEDEN SADECE "koleksiyon bos mu" YETMIYOR:
Ollama (nomic-embed-text) ve Gemini (gemini-embedding-001, 768'e kisilmis) ayni
BOYUTTA vektor uretiyor, yani saglayici degisince ChromaDB hata VERMEZ - ama iki
model ayni anlam uzayinda degil. Ollama ile indekslenmis koleksiyonu Gemini ile
sorgulamak sessizce alakasiz kurallar dondururdu. Bu yuzden koleksiyona hangi
saglayici/model ile olusturuldugu damgalaniyor ve acilista karsilastiriliyor.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from app.config import Settings
from app.providers.base import EmbeddingProvider
from app.providers.factory import get_embedding_provider

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")

# Hangi ayarin o saglayicinin model adini tuttugu - tek yerde, cunku hem damgalama
# hem karsilastirma ayni esleme uzerinden yapilmali.
_MODEL_SETTING_BY_PROVIDER = {
    "ollama": "OLLAMA_EMBEDDING_MODEL",
    "gemini": "GEMINI_EMBEDDING_MODEL",
}

# Koleksiyon metadata'sinda karsilastirilan anahtarlar. Chroma metadata'ya kendi
# ayarlarini da ekliyor, o yuzden butun sozlugu degil yalnizca bunlari kiyasliyoruz.
IDENTITY_KEYS = ("embedding_provider", "embedding_model")


def embedding_identity(settings: Settings) -> dict[str, str]:
    """Koleksiyona damgalanacak/karsilastirilacak saglayici kimligi."""
    provider = settings.EMBEDDING_PROVIDER
    setting_name = _MODEL_SETTING_BY_PROVIDER.get(provider)
    model = getattr(settings, setting_name) if setting_name else ""
    return {"embedding_provider": provider, "embedding_model": model}


class KnowledgeBaseLoader:
    """data/knowledge_base/ altindaki .txt dosyalarini (dosya adi -> metin) okur."""

    def __init__(self, directory: Path):
        self.directory = directory

    def load(self) -> dict[str, str]:
        paths = sorted(self.directory.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(f"Kural belgesi bulunamadi: {self.directory}/*.txt")
        return {path.stem: path.read_text(encoding="utf-8").strip() for path in paths}


class KnowledgeBaseIndexer:
    """Belgeleri embed edip verilen ChromaDB koleksiyonuna yazar."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chroma_client: chromadb.ClientAPI,
        collection_name: str,
        identity: dict[str, str],
    ):
        self.embedding_provider = embedding_provider
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.identity = identity

    def reindex(self, documents: dict[str, str]) -> int:
        """Koleksiyonu sifirdan olusturur (varsa once siler) ve tum belgeleri ekler.

        Idempotent: diskteki .txt dosyalari degisse/silinse bile koleksiyon her zaman
        onlarla birebir eslesir, eski belgeler geride kalmaz. Eklenen belge sayisini dondurur."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except NotFoundError:
            pass  # ilk calistirma: koleksiyon henuz yok, sorun degil

        ids = list(documents.keys())
        texts = list(documents.values())
        # embed_many: toplu cagriyi destekleyen saglayicilarda (Gemini) bu tek istek
        # eder - Render her uyandiginda yeniden indeksledigi icin kota acisindan onemli.
        embeddings = self.embedding_provider.embed_many(texts)

        metadata = dict(self.identity)
        if embeddings:
            metadata["dim"] = len(embeddings[0])

        # embedding_function=None: vektorleri kendimiz uretip veriyoruz (Chroma'nin
        # varsayilan/indirilen embedding modelini kullanmasini istemiyoruz).
        collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name, embedding_function=None, metadata=metadata
        )
        collection.add(ids=ids, embeddings=embeddings, documents=texts)
        return collection.count()


def reindex_reason(
    chroma_client: chromadb.ClientAPI, collection_name: str, identity: dict[str, str]
) -> str | None:
    """Yeniden indeksleme gerekiyorsa nedenini (loglanabilir metin), gerekmiyorsa None dondurur."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except NotFoundError:
        return "koleksiyon yok"

    if collection.count() == 0:
        return "koleksiyon bos"

    stamped = collection.metadata or {}
    missing = [key for key in IDENTITY_KEYS if key not in stamped]
    if missing:
        # Damga oncesi olusturulmus eski koleksiyon: hangi modelle kuruldugu bilinmiyor,
        # guvenli taraf yeniden indekslemek.
        return f"saglayici damgasi yok ({', '.join(missing)})"

    differences = [
        f"{key}: {stamped[key]!r} -> {identity[key]!r}"
        for key in IDENTITY_KEYS
        if stamped[key] != identity[key]
    ]
    if differences:
        return "saglayici/model degismis (" + "; ".join(differences) + ")"
    return None


def ensure_indexed(
    settings: Settings | None = None,
    knowledge_base_dir: Path | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> str:
    """Gerekiyorsa bilgi tabanini yeniden indeksler; ne yapildigini anlatan metin dondurur.

    Gerekli olma kosullari: koleksiyon yok, bos, damgasiz ya da baska bir
    saglayici/model ile olusturulmus (bkz. reindex_reason).

    Parametreler testlerde sahtelenebilsin diye opsiyonel; verilmezse gercekleri kurulur."""
    settings = settings or Settings()
    knowledge_base_dir = knowledge_base_dir or DEFAULT_KNOWLEDGE_BASE_DIR
    chroma_client = chroma_client or chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

    identity = embedding_identity(settings)
    reason = reindex_reason(chroma_client, settings.CHROMA_COLLECTION_NAME, identity)
    if reason is None:
        logger.info(
            "Bilgi tabani guncel (%s / %s), yeniden indekslenmedi.",
            identity["embedding_provider"], identity["embedding_model"],
        )
        return "guncel"

    logger.info("Bilgi tabani yeniden indeksleniyor - neden: %s", reason)
    embedding_provider = embedding_provider or get_embedding_provider(settings)
    documents = KnowledgeBaseLoader(knowledge_base_dir).load()
    indexer = KnowledgeBaseIndexer(
        embedding_provider, chroma_client, settings.CHROMA_COLLECTION_NAME, identity
    )
    count = indexer.reindex(documents)
    logger.info(
        "Bilgi tabani indekslendi: %d belge (%s / %s).",
        count, identity["embedding_provider"], identity["embedding_model"],
    )
    return f"indekslendi: {count} belge ({reason})"

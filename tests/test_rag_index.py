"""
app/rag_index.py'yi test eder.

Gercek dis servise cikmaz: embedding provider sahte (sabit vektor uretir).
ChromaDB gercek ama yerel/gecici (tmp_path) - projedeki diger Chroma testleriyle
ayni yaklasim, ag gerektirmiyor.

Asil onemli senaryo: Ollama ile indekslenmis bir koleksiyonu Gemini saglayicisiyla
kullanmak. Ikisi de 768 boyut uretiyor, yani ChromaDB HATA VERMEZ; damga olmasa
sessizce alakasiz kurallar donerdi. Bu yuzden damga karsilastirmasi test ediliyor.
"""

import chromadb
import pytest

from app.config import Settings
from app.providers.base import EmbeddingProvider
from app.rag_index import (
    KnowledgeBaseIndexer,
    KnowledgeBaseLoader,
    embedding_identity,
    ensure_indexed,
    reindex_reason,
)

COLLECTION = "invoice_rules"


class FakeEmbeddingProvider(EmbeddingProvider):
    """Aga cikmaz. Metin uzunluguna bagli, sabit boyutlu bir vektor uretir."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.embed_calls = 0
        self.embed_many_calls = 0

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [float(len(text) % 7) + 1.0] + [0.0] * (self.dim - 1)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.embed_many_calls += 1
        return [self.embed(t) for t in texts]


@pytest.fixture
def kb_dir(tmp_path):
    directory = tmp_path / "knowledge_base"
    directory.mkdir()
    (directory / "kdv_oranlari.txt").write_text("KDV oranlari %1, %10, %20.", encoding="utf-8")
    (directory / "birim_fiyat_klavye.txt").write_text("Klavye genellikle 300-2500 TL.", encoding="utf-8")
    return directory


def _settings(tmp_path, provider="ollama", **extra) -> Settings:
    return Settings(
        _env_file=None,
        EMBEDDING_PROVIDER=provider,
        CHROMA_DB_PATH=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME=COLLECTION,
        GEMINI_API_KEY="test-key",
        **extra,
    )


def _client(tmp_path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


# --- KnowledgeBaseLoader -------------------------------------------------------

def test_loader_reads_txt_files_keyed_by_stem(kb_dir):
    documents = KnowledgeBaseLoader(kb_dir).load()
    assert set(documents) == {"kdv_oranlari", "birim_fiyat_klavye"}
    assert documents["kdv_oranlari"] == "KDV oranlari %1, %10, %20."


def test_loader_raises_when_no_documents(tmp_path):
    empty = tmp_path / "bos"
    empty.mkdir()
    loader = KnowledgeBaseLoader(empty)
    with pytest.raises(FileNotFoundError, match="Kural belgesi bulunamadi"):
        loader.load()


# --- damga (identity) ----------------------------------------------------------

def test_embedding_identity_uses_the_model_setting_of_the_active_provider(tmp_path):
    ollama = embedding_identity(_settings(tmp_path, "ollama"))
    gemini = embedding_identity(_settings(tmp_path, "gemini"))

    assert ollama == {"embedding_provider": "ollama", "embedding_model": "nomic-embed-text"}
    assert gemini["embedding_provider"] == "gemini"
    assert gemini["embedding_model"] == "gemini-embedding-001"


def test_reindex_stamps_provider_model_and_dimension(tmp_path, kb_dir):
    settings = _settings(tmp_path, "ollama")
    client = _client(tmp_path)
    identity = embedding_identity(settings)

    KnowledgeBaseIndexer(FakeEmbeddingProvider(), client, COLLECTION, identity).reindex(
        KnowledgeBaseLoader(kb_dir).load()
    )

    metadata = client.get_collection(COLLECTION).metadata
    assert metadata["embedding_provider"] == "ollama"
    assert metadata["embedding_model"] == "nomic-embed-text"
    assert metadata["dim"] == 768


def test_reindex_uses_embed_many_not_one_call_per_document(tmp_path, kb_dir):
    """Toplu cagri kota maliyetini dusuruyor; indexer bunu kullanmali."""
    provider = FakeEmbeddingProvider()
    identity = embedding_identity(_settings(tmp_path, "ollama"))

    KnowledgeBaseIndexer(provider, _client(tmp_path), COLLECTION, identity).reindex(
        KnowledgeBaseLoader(kb_dir).load()
    )

    assert provider.embed_many_calls == 1


# --- reindex_reason ------------------------------------------------------------

def test_reason_is_collection_missing_when_never_indexed(tmp_path):
    reason = reindex_reason(_client(tmp_path), COLLECTION, {"embedding_provider": "ollama", "embedding_model": "x"})
    assert reason == "koleksiyon yok"


def test_reason_is_empty_when_collection_has_no_documents(tmp_path):
    client = _client(tmp_path)
    client.get_or_create_collection(name=COLLECTION, embedding_function=None)
    reason = reindex_reason(client, COLLECTION, {"embedding_provider": "ollama", "embedding_model": "x"})
    assert reason == "koleksiyon bos"


def test_reason_is_none_when_stamp_matches(tmp_path, kb_dir):
    settings = _settings(tmp_path, "ollama")
    client = _client(tmp_path)
    identity = embedding_identity(settings)
    KnowledgeBaseIndexer(FakeEmbeddingProvider(), client, COLLECTION, identity).reindex(
        KnowledgeBaseLoader(kb_dir).load()
    )

    assert reindex_reason(client, COLLECTION, identity) is None


def test_ollama_indexed_collection_must_be_reindexed_for_gemini(tmp_path, kb_dir):
    """ASIL SENARYO: koleksiyon Ollama ile kurulmus, saglayici Gemini'ye cevrilmis.

    Iki model de 768 boyut uretiyor, yani ChromaDB hic hata vermez - damga olmasa
    sorgular sessizce alakasiz kurallar dondururdu."""
    client = _client(tmp_path)
    ollama_identity = embedding_identity(_settings(tmp_path, "ollama"))
    KnowledgeBaseIndexer(FakeEmbeddingProvider(), client, COLLECTION, ollama_identity).reindex(
        KnowledgeBaseLoader(kb_dir).load()
    )

    gemini_identity = embedding_identity(_settings(tmp_path, "gemini"))
    reason = reindex_reason(client, COLLECTION, gemini_identity)

    assert reason is not None
    assert "saglayici/model degismis" in reason
    assert "'ollama'" in reason and "'gemini'" in reason


def test_same_provider_but_different_model_also_triggers_reindex(tmp_path, kb_dir):
    client = _client(tmp_path)
    identity = embedding_identity(_settings(tmp_path, "ollama"))
    KnowledgeBaseIndexer(FakeEmbeddingProvider(), client, COLLECTION, identity).reindex(
        KnowledgeBaseLoader(kb_dir).load()
    )

    other_model = embedding_identity(_settings(tmp_path, "ollama", OLLAMA_EMBEDDING_MODEL="baska-model"))
    reason = reindex_reason(client, COLLECTION, other_model)

    assert reason is not None
    assert "embedding_model" in reason


def test_unstamped_legacy_collection_triggers_reindex(tmp_path, kb_dir):
    """Damga eklenmeden once olusturulmus koleksiyonlar: hangi modelle kuruldugu
    bilinmedigi icin guvenli taraf yeniden indekslemek."""
    client = _client(tmp_path)
    collection = client.get_or_create_collection(name=COLLECTION, embedding_function=None)
    collection.add(ids=["a"], embeddings=[[1.0] + [0.0] * 767], documents=["eski kayit"])

    reason = reindex_reason(client, COLLECTION, {"embedding_provider": "gemini", "embedding_model": "m"})

    assert reason is not None
    assert "damgasi yok" in reason


# --- ensure_indexed ------------------------------------------------------------

def test_ensure_indexed_builds_collection_on_first_run(tmp_path, kb_dir):
    settings = _settings(tmp_path, "ollama")
    provider = FakeEmbeddingProvider()

    result = ensure_indexed(
        settings=settings, knowledge_base_dir=kb_dir,
        chroma_client=_client(tmp_path), embedding_provider=provider,
    )

    assert "indekslendi: 2 belge" in result
    assert "koleksiyon yok" in result


def test_ensure_indexed_is_a_noop_when_already_current(tmp_path, kb_dir):
    settings = _settings(tmp_path, "ollama")
    client = _client(tmp_path)
    ensure_indexed(settings=settings, knowledge_base_dir=kb_dir, chroma_client=client,
                   embedding_provider=FakeEmbeddingProvider())

    second_provider = FakeEmbeddingProvider()
    result = ensure_indexed(settings=settings, knowledge_base_dir=kb_dir, chroma_client=client,
                            embedding_provider=second_provider)

    assert result == "guncel"
    assert second_provider.embed_many_calls == 0  # bosa embedding cagrisi yapilmadi


def test_ensure_indexed_rebuilds_when_provider_changed(tmp_path, kb_dir):
    """Ollama ile kur, sonra Gemini ayarlariyla cagir: yeniden indekslenmeli."""
    client = _client(tmp_path)
    ensure_indexed(settings=_settings(tmp_path, "ollama"), knowledge_base_dir=kb_dir,
                   chroma_client=client, embedding_provider=FakeEmbeddingProvider())

    gemini_provider = FakeEmbeddingProvider()
    result = ensure_indexed(settings=_settings(tmp_path, "gemini"), knowledge_base_dir=kb_dir,
                            chroma_client=client, embedding_provider=gemini_provider)

    assert "indekslendi" in result
    assert "saglayici/model degismis" in result
    assert gemini_provider.embed_many_calls == 1
    assert client.get_collection(COLLECTION).metadata["embedding_provider"] == "gemini"

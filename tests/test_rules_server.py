"""
app/mcp/rules_server.py'yi test eder.

_query_rules_impl birim testleri protokolden tamamen bagimsizdir (dogrudan Python
fonksiyonu olarak cagrilir, MCP/subprocess yok) ve eski RAGAgent'in embed+ChromaDB
mantiginin (bkz. tests/test_rag_agent.py'nin eski surumundeki ayni senaryolar:
tek en alakali kural, top_k kadar kural, bos koleksiyon hatasi, saglayici hatasi)
BIREBIR AYNI sonuclari urettigini kanitlar - bu, RAGAgent'tan silinmeden once
davranisin korundugunun kanitidir.

Sondaki test gercek bir stdio subprocess'i (python -m app.mcp.rules_server) baslatip
gercek Ollama embedding + gercek (izole, tmp_path'teki) ChromaDB ile uctan uca calisir;
yavas oldugu icin @pytest.mark.slow ile isaretlenmistir.
"""

import asyncio
import sys
from pathlib import Path

import chromadb
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings
from app.mcp.rules_server import (
    KnowledgeBaseEmptyError,
    _query_rules_impl,
    build_server,
)
from app.providers.base import EmbeddingProvider, ProviderUnavailableError
from app.providers.factory import get_embedding_provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingProvider(EmbeddingProvider):
    """Aga cikmaz. VOCAB'daki her kelime metinde varsa 1, yoksa 0 olan sabit uzunlukta vektor uretir."""

    VOCAB = ["klavye", "mouse", "kağıt", "kdv", "vergi", "fatura", "toner"]

    def __init__(self):
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        lowered = text.lower()
        return [1.0 if word in lowered else 0.0 for word in self.VOCAB]


class FailingEmbeddingProvider(EmbeddingProvider):
    """embed() her zaman ProviderUnavailableError firlatir (baglanti yokmus gibi)."""

    def embed(self, text: str) -> list[float]:
        raise ProviderUnavailableError("Ollama sunucusuna ulasilamadi")


def _seed_collection(chroma_path, collection_name: str, docs: dict[str, str], provider=None) -> None:
    provider = provider or FakeEmbeddingProvider()
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=None)
    collection.add(
        ids=list(docs.keys()),
        embeddings=[provider.embed(text) for text in docs.values()],
        documents=list(docs.values()),
    )


def _collection(chroma_path, collection_name: str):
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(name=collection_name, embedding_function=None)


# --- _query_rules_impl: protokolden bagimsiz, eski RAGAgent davranisiyla ayni ------------

def test_query_rules_impl_returns_single_most_relevant_rule(tmp_path):
    _seed_collection(tmp_path, "rules", {
        "d_klavye": "Klavye ile ilgili kural metni.",
        "d_kagit": "Kağıt ile ilgili kural metni.",
        "d_toner": "Toner ile ilgili kural metni.",
    })
    collection = _collection(tmp_path, "rules")

    result = _query_rules_impl("Klavye", top_k=1, embedding_provider=FakeEmbeddingProvider(), collection=collection)

    assert result == ["Klavye ile ilgili kural metni."]


def test_query_rules_impl_returns_up_to_top_k_rules(tmp_path):
    _seed_collection(tmp_path, "rules", {
        "d_klavye": "Klavye ile ilgili kural metni.",
        "d_kagit": "Kağıt ile ilgili kural metni.",
        "d_toner": "Toner ile ilgili kural metni.",
    })
    collection = _collection(tmp_path, "rules")

    result = _query_rules_impl("Klavye", top_k=3, embedding_provider=FakeEmbeddingProvider(), collection=collection)

    assert len(result) == 3
    assert "Klavye ile ilgili kural metni." in result


def test_query_rules_impl_raises_when_collection_empty(tmp_path):
    collection = _collection(tmp_path, "rules")  # hic .add() edilmedi

    with pytest.raises(KnowledgeBaseEmptyError, match="index_knowledge_base"):
        _query_rules_impl("Klavye", top_k=3, embedding_provider=FakeEmbeddingProvider(), collection=collection)


def test_query_rules_impl_does_not_swallow_provider_errors(tmp_path):
    _seed_collection(tmp_path, "rules", {"d1": "herhangi bir kural metni."})
    collection = _collection(tmp_path, "rules")

    with pytest.raises(ProviderUnavailableError):
        _query_rules_impl("Klavye", top_k=1, embedding_provider=FailingEmbeddingProvider(), collection=collection)


# --- build_server(): tool dogru bagimliliklarla kayitli mi -------------------------------

def test_build_server_registers_query_rules_tool(tmp_path):
    settings = Settings(_env_file=None, CHROMA_DB_PATH=str(tmp_path), CHROMA_COLLECTION_NAME="rules")
    server = build_server(settings=settings, embedding_provider=FakeEmbeddingProvider())

    assert server.name == "invoice-rules"


def test_build_server_tool_uses_injected_embedding_provider_and_chroma_path(tmp_path):
    _seed_collection(tmp_path, "rules", {"d_klavye": "Klavye ile ilgili kural metni."})
    settings = Settings(_env_file=None, CHROMA_DB_PATH=str(tmp_path), CHROMA_COLLECTION_NAME="rules")
    server = build_server(settings=settings, embedding_provider=FakeEmbeddingProvider())

    result = asyncio.run(server.call_tool("query_rules", {"query": "Klavye", "top_k": 1}))

    assert result.structured_content["result"] == ["Klavye ile ilgili kural metni."]


# --- gercek stdio subprocess ucu ucu -------------------------------------------------------

@pytest.mark.slow
def test_real_stdio_subprocess_round_trip_with_real_ollama_embedding(tmp_path):
    """python -m app.mcp.rules_server'i gercek bir alt surec olarak baslatir; gercek Ollama
    embedding kullanir (Ollama'nin calisiyor ve nomic-embed-text'in cekili olmasi gerekir).
    ChromaDB izole bir tmp_path'te (gercek projenin data/chroma_db'sine dokunulmaz)."""
    real_embedding_provider = get_embedding_provider(Settings(_env_file=None))
    _seed_collection(tmp_path, "rules", {
        "d_klavye": "Klavye ile ilgili kural metni.",
        "d_kagit": "Kağıt ile ilgili kural metni.",
    }, provider=real_embedding_provider)

    async def run() -> list[str]:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.rules_server"],
            cwd=str(PROJECT_ROOT),
            env={"CHROMA_DB_PATH": str(tmp_path), "CHROMA_COLLECTION_NAME": "rules"},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("query_rules", {"query": "Klavye", "top_k": 1})
        return result.structured_content["result"]

    rules = asyncio.run(run())
    assert rules == ["Klavye ile ilgili kural metni."]

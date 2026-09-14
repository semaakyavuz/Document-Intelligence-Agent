"""
RAGAgent'in sorgu kurma ve getirme (retrieval) mantigini test eder.

Gercek Ollama'ya baglanmaz (FakeEmbeddingProvider). ChromaDB gercek ve yereldir
(tmp_path'e persist eder) -- bu "dis servis" sayilmaz, tamamen yerel/embedded bir
veritabani; boylece RAGAgent'in .query() cagrisini dogru yapip yapmadigi da
gercekten sinanmis olur, sadece mock edilmez.

FakeEmbeddingProvider basit bir "bag of keyword" vektoru uretir: sabit bir kelime
listesindeki her kelime metinde geciyorsa 1, gecmiyorsa 0. Boylece hangi belgenin
sorguya en yakin oldugu onceden kesin olarak bilinir (ties olmadan).
"""

import pytest

from app.agents.rag_agent import KnowledgeBaseEmptyError, RAGAgent, RuleQueryBuilder
from app.providers.base import EmbeddingProvider, ProviderUnavailableError
from app.state import PipelineState

VALID_EXTRACTION = {
    "invoice_no": "2025123456",
    "seller_tax_no": "1234567890",
    "items": [
        {"description": "Klavye", "vat_rate": 0.20},
        {"description": "Mouse", "vat_rate": 0.01},
    ],
}


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


def _seed_collection(chroma_path, collection_name: str, docs: dict[str, str]) -> None:
    """docs'u FakeEmbeddingProvider ile embed edip belirtilen yola/koleksiyona yazar."""
    import chromadb

    provider = FakeEmbeddingProvider()
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=None)
    collection.add(
        ids=list(docs.keys()),
        embeddings=[provider.embed(text) for text in docs.values()],
        documents=list(docs.values()),
    )


# --- RuleQueryBuilder --------------------------------------------------------

def test_query_builder_combines_items_vat_and_ids():
    query = RuleQueryBuilder().build(VALID_EXTRACTION)
    assert "Klavye" in query and "Mouse" in query
    assert "%20" in query and "%1" in query
    assert "2025123456" in query
    assert "1234567890" in query


def test_query_builder_skips_malformed_items_without_crashing():
    broken = {
        "items": [
            "bu bir dict degil",
            {"description": 123},  # description string degil
            {"description": "Klavye", "vat_rate": "gecersiz"},  # vat_rate sayiya cevrilemiyor
            {"description": "Mouse", "vat_rate": 0.01},  # bu gecerli
        ],
    }
    query = RuleQueryBuilder().build(broken)
    assert "Mouse" in query
    assert "%1" in query
    assert "Klavye" in query  # description gecerli, sadece vat_rate atlanir


def test_query_builder_returns_empty_string_for_unusable_extraction():
    assert RuleQueryBuilder().build({}) == ""
    assert RuleQueryBuilder().build({"foo": "bar"}) == ""
    assert RuleQueryBuilder().build({"items": []}) == ""


# --- RAGAgent.run() -----------------------------------------------------------

def test_returns_empty_rules_when_raw_extraction_is_none(tmp_path):
    provider = FakeEmbeddingProvider()
    agent = RAGAgent(provider, chroma_path=str(tmp_path), collection_name="rules", top_k=3)

    result = agent.run(PipelineState(image_path="x.png", raw_extraction=None))

    assert result.retrieved_rules == []
    assert provider.calls == []  # embed() hic cagrilmadi


def test_returns_empty_rules_when_query_text_is_empty(tmp_path):
    provider = FakeEmbeddingProvider()
    agent = RAGAgent(provider, chroma_path=str(tmp_path), collection_name="rules", top_k=3)

    result = agent.run(PipelineState(image_path="x.png", raw_extraction={"foo": "bar"}))

    assert result.retrieved_rules == []
    assert provider.calls == []


def test_raises_when_knowledge_base_not_indexed(tmp_path):
    provider = FakeEmbeddingProvider()
    agent = RAGAgent(provider, chroma_path=str(tmp_path), collection_name="rules", top_k=3)

    with pytest.raises(KnowledgeBaseEmptyError, match="index_knowledge_base"):
        agent.run(PipelineState(image_path="x.png", raw_extraction=VALID_EXTRACTION))


def test_retrieves_the_single_most_relevant_rule(tmp_path):
    _seed_collection(tmp_path, "rules", {
        "d_klavye": "Klavye ile ilgili kural metni.",
        "d_kagit": "Kağıt ile ilgili kural metni.",
        "d_toner": "Toner ile ilgili kural metni.",
    })
    agent = RAGAgent(FakeEmbeddingProvider(), chroma_path=str(tmp_path), collection_name="rules", top_k=1)

    result = agent.run(PipelineState(image_path="x.png", raw_extraction={"items": [{"description": "Klavye"}]}))

    assert result.retrieved_rules == ["Klavye ile ilgili kural metni."]


def test_retrieves_up_to_top_k_rules_and_does_not_mutate_input_state(tmp_path):
    _seed_collection(tmp_path, "rules", {
        "d_klavye": "Klavye ile ilgili kural metni.",
        "d_kagit": "Kağıt ile ilgili kural metni.",
        "d_toner": "Toner ile ilgili kural metni.",
    })
    agent = RAGAgent(FakeEmbeddingProvider(), chroma_path=str(tmp_path), collection_name="rules", top_k=3)
    original = PipelineState(image_path="x.png", raw_extraction={"items": [{"description": "Klavye"}]})

    result = agent.run(original)

    assert len(result.retrieved_rules) == 3
    assert "Klavye ile ilgili kural metni." in result.retrieved_rules
    assert original.retrieved_rules == []  # model_copy: girdi degismedi


def test_provider_error_is_not_swallowed(tmp_path):
    _seed_collection(tmp_path, "rules", {"d1": "herhangi bir kural metni."})
    agent = RAGAgent(FailingEmbeddingProvider(), chroma_path=str(tmp_path), collection_name="rules", top_k=1)

    with pytest.raises(ProviderUnavailableError):
        agent.run(PipelineState(image_path="x.png", raw_extraction=VALID_EXTRACTION))

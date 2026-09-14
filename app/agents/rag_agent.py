"""
rag_agent.py

Vision Agent'in cikardigi verilerden bir sorgu metni kurar, data/knowledge_base/
altindaki kural belgelerinin ChromaDB'de indekslenmis halinden en alakali olanlari
getirir ve state.retrieved_rules'a yazar.

RAGAgent bir LLMProvider degil bir EmbeddingProvider + yerel bir vektor veritabani
kullanir; bu yuzden BaseAgent.__init__'i (llm_provider bekler) miras almiyor, kendi
imzasini tanimliyor. BaseAgent.__init__ soyut olmadigi icin bu gecerli bir ABC kullanimi.
"""

import chromadb

from app.agents.base import BaseAgent
from app.providers.base import EmbeddingProvider
from app.state import PipelineState


class KnowledgeBaseEmptyError(RuntimeError):
    """Koleksiyon hic indekslenmemis (once scripts/index_knowledge_base.py calistirilmali)."""


class RuleQueryBuilder:
    """Vision Agent ciktisindan (raw_extraction) kisa bir Turkce sorgu metni kurar.

    Girdi bir LLM ciktisi oldugu icin sema garantisi yoktur (Validation Agent henuz
    calismadi); eksik/bozuk alanlar sessizce atlanir, hicbir zaman hata firlatilmaz.
    """

    def build(self, raw_extraction: dict) -> str:
        parts: list[str] = []

        items = raw_extraction.get("items")
        if isinstance(items, list):
            descriptions = [
                item["description"] for item in items
                if isinstance(item, dict) and isinstance(item.get("description"), str) and item["description"]
            ]
            if descriptions:
                parts.append("Ürünler: " + ", ".join(descriptions) + ".")

            vat_percentages = [self._as_percentage(item.get("vat_rate")) for item in items if isinstance(item, dict)]
            vat_percentages = [p for p in vat_percentages if p is not None]
            if vat_percentages:
                parts.append("KDV oranları: " + ", ".join(vat_percentages) + ".")

        invoice_no = raw_extraction.get("invoice_no")
        if isinstance(invoice_no, str) and invoice_no:
            parts.append(f"Fatura numarası: {invoice_no}.")

        seller_tax_no = raw_extraction.get("seller_tax_no")
        if isinstance(seller_tax_no, str) and seller_tax_no:
            parts.append(f"Vergi numarası: {seller_tax_no}.")

        return " ".join(parts)

    @staticmethod
    def _as_percentage(vat_rate) -> str | None:
        try:
            return f"%{float(vat_rate) * 100:.0f}"
        except (TypeError, ValueError):
            return None


class RAGAgent(BaseAgent):
    """Cikarilan fatura verisiyle ilgili kural belgelerini yerel bilgi tabanindan getirir."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chroma_path: str,
        collection_name: str,
        top_k: int = 3,
        query_builder: RuleQueryBuilder | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self._query_builder = query_builder or RuleQueryBuilder()
        # Client/koleksiyon tek seferde acilir; her run() cagrisinda yeniden baglanilmaz.
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._client.get_or_create_collection(name=collection_name, embedding_function=None)

    def run(self, state: PipelineState) -> PipelineState:
        if not state.raw_extraction:
            # Vision Agent basarisiz olmus ya da henuz calismamis: sorgulanacak bir sey yok,
            # bu bir hata degil; bos liste ile devam edilir.
            return state.model_copy(update={"retrieved_rules": []})

        query_text = self._query_builder.build(state.raw_extraction)
        if not query_text:
            return state.model_copy(update={"retrieved_rules": []})

        if self._collection.count() == 0:
            raise KnowledgeBaseEmptyError(
                "Bilgi tabani bos. Once 'python scripts/index_knowledge_base.py' calistirin."
            )

        # Baglanti hatalari burada yutulmaz: VisionAgent'taki ayni kuralin aynisi.
        query_embedding = self.embedding_provider.embed(query_text)
        result = self._collection.query(query_embeddings=[query_embedding], n_results=self.top_k)

        retrieved_rules = result["documents"][0] if result["documents"] else []
        return state.model_copy(update={"retrieved_rules": retrieved_rules})

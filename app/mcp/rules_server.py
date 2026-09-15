"""
rules_server.py

RAG kural sorgulamasini (embed + ChromaDB query) bagimsiz bir MCP server olarak
sunar. Tek arac: query_rules(query, top_k) -> list[str].

_query_rules_impl, RAGAgent'ta eskiden dogrudan calisan embed+query mantiginin
BIREBIR AYNISIdir (kopyalanmistir); protokolden tamamen bagimsiz, saf bir
fonksiyondur - hem burada tool olarak sarmalanir, hem tests/test_rules_server.py'de
dogrudan cagrilip eski RAGAgent davranisiyla karsilastirilir.

Bagimsiz calistirma (stdio):
    python -m app.mcp.rules_server

Not: FastMCP, guncel "mcp" SDK'sinda MCPServer olarak yeniden adlandirildi
(pip'teki mcp>=2 surumu; eski FastMCP ismi artik yok, import hatasi dogrudan
buraya yonlendiriyor). Ayni "tool" dekoratoru / stdio calistirma API'si gecerli.
"""

import chromadb
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.config import Settings
from app.providers.base import EmbeddingProvider, ProviderError
from app.providers.factory import get_embedding_provider


class KnowledgeBaseEmptyError(RuntimeError):
    """Koleksiyon hic indekslenmemis (once scripts/index_knowledge_base.py calistirilmali)."""


def _query_rules_impl(
    query: str, top_k: int, embedding_provider: EmbeddingProvider, collection: chromadb.Collection
) -> list[str]:
    """RAGAgent'ta eskiden dogrudan calisan embed+query mantiginin birebir aynisi."""
    if collection.count() == 0:
        raise KnowledgeBaseEmptyError(
            "Bilgi tabani bos. Once 'python scripts/index_knowledge_base.py' calistirin."
        )

    # Baglanti hatalari burada yutulmaz: VisionAgent/eski RAGAgent'taki ayni kuralin aynisi.
    query_embedding = embedding_provider.embed(query)
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return result["documents"][0] if result["documents"] else []


def build_server(settings: Settings | None = None, embedding_provider: EmbeddingProvider | None = None) -> MCPServer:
    """Derlenmis (configure edilmis) MCP server'i kurar.

    embedding_provider verilmezse factory'den (.env'deki ayarlara gore) kurulur;
    testlerde gercek Ollama'ya baglanmadan sahte bir provider dogrudan enjekte edilebilir."""
    settings = settings or Settings()
    embedding_provider = embedding_provider or get_embedding_provider(settings)

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME, embedding_function=None)

    server = MCPServer("invoice-rules")

    @server.tool()
    def query_rules(query: str, top_k: int = 3) -> list[str]:
        """Verilen sorgu metnine en alakali en fazla top_k kural belgesini dondurur."""
        try:
            return _query_rules_impl(query, top_k, embedding_provider, collection)
        except (KnowledgeBaseEmptyError, ProviderError) as exc:
            # Ham exception firlatilirsa mesaji client'a ulasmiyor (MCP SDK'si "beklenmeyen"
            # hatalari guvenlik icin generic bir mesaja indirgiyor - canlica dogrulandi).
            # ToolError ise mesaji oldugu gibi client'a tasiyor; RAGAgent bunu okuyup
            # kendi tarafinda anlamli bir hata (RuleQueryError) olarak yeniden firlatiyor.
            raise ToolError(str(exc)) from exc

    return server


def main() -> None:
    build_server().run()  # varsayilan transport zaten stdio


if __name__ == "__main__":
    main()

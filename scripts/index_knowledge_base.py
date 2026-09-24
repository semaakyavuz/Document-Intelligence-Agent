"""
index_knowledge_base.py

data/knowledge_base/ altindaki kural belgelerini embed edip ChromaDB'ye yazar.

Asil mantik app/rag_index.py'de: uygulama acilista da (bulutta disk gecici oldugu
icin) ayni kodu calistirmak zorunda, bu yuzden burada degil app/ altinda duruyor.
Bu script yalnizca CLI sarmalayicisi.

Tekrar calistirildiginda mevcut koleksiyonu siler ve sifirdan yeniden olusturur
(idempotent): belgeler degisse/eklense/cikarilsa bile koleksiyon her zaman diskteki
.txt dosyalariyla birebir eslesir, eski/silinmis belgeler koleksiyonda kalmaz.

Yalnizca yerel calisir: embedding provider .env'den EMBEDDING_PROVIDER=ollama olmasini
BEKLER ve degilse acik bir hata ile durur (Gemini/cloud'a yanlislikla gidilmesin diye).

Kullanim:
    python scripts/index_knowledge_base.py
    python scripts/index_knowledge_base.py --knowledge-base-dir data/knowledge_base --chroma-path data/chroma_db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402

from app.config import Settings  # noqa: E402
from app.providers.factory import get_embedding_provider  # noqa: E402
from app.providers.ollama_provider import OllamaEmbeddingProvider  # noqa: E402
from app.rag_index import KnowledgeBaseIndexer, KnowledgeBaseLoader, embedding_identity  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kural belgelerini embed edip ChromaDB'ye indeksler.")
    parser.add_argument("--knowledge-base-dir", type=Path, default=Path("data/knowledge_base"))
    parser.add_argument("--chroma-path", type=Path, default=None, help="Varsayilan: Settings.CHROMA_DB_PATH")
    parser.add_argument("--collection-name", type=str, default=None, help="Varsayilan: Settings.CHROMA_COLLECTION_NAME")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = Settings()

    if settings.EMBEDDING_PROVIDER != "ollama":
        raise SystemExit(
            f"Bu script yalnizca yerel calisir ama .env'de EMBEDDING_PROVIDER={settings.EMBEDDING_PROVIDER!r}. "
            "Bulut saglayiciya gitmemesi icin durduruldu; .env'de EMBEDDING_PROVIDER=ollama yapin."
        )
    embedding_provider = get_embedding_provider(settings)
    if not isinstance(embedding_provider, OllamaEmbeddingProvider):
        raise SystemExit(f"Beklenmeyen embedding provider turu: {type(embedding_provider).__name__} (Ollama olmali)")

    chroma_path = args.chroma_path or Path(settings.CHROMA_DB_PATH)
    collection_name = args.collection_name or settings.CHROMA_COLLECTION_NAME

    documents = KnowledgeBaseLoader(args.knowledge_base_dir).load()
    print(f"{len(documents)} kural belgesi bulundu: {args.knowledge_base_dir}")

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    identity = embedding_identity(settings)
    indexer = KnowledgeBaseIndexer(embedding_provider, chroma_client, collection_name, identity)
    count = indexer.reindex(documents)

    print(
        f"Koleksiyon '{collection_name}' yeniden olusturuldu -> {chroma_path} "
        f"({count} belge, damga: {identity['embedding_provider']}/{identity['embedding_model']})"
    )


if __name__ == "__main__":
    main()

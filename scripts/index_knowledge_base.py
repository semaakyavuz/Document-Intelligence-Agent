"""
index_knowledge_base.py

data/knowledge_base/ altindaki kural belgelerini embed edip ChromaDB'ye yazar.

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
from chromadb.errors import NotFoundError  # noqa: E402

from app.config import Settings  # noqa: E402
from app.providers.factory import get_embedding_provider  # noqa: E402
from app.providers.ollama_provider import OllamaEmbeddingProvider  # noqa: E402
from app.providers.base import EmbeddingProvider  # noqa: E402


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

    def __init__(self, embedding_provider: EmbeddingProvider, chroma_client: chromadb.ClientAPI, collection_name: str):
        self.embedding_provider = embedding_provider
        self.chroma_client = chroma_client
        self.collection_name = collection_name

    def reindex(self, documents: dict[str, str]) -> int:
        """Koleksiyonu sifirdan olusturur (varsa once siler) ve tum belgeleri ekler. Eklenen belge sayisini dondurur."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except NotFoundError:
            pass  # ilk calistirma: koleksiyon henuz yok, sorun degil

        # embedding_function=None: vektorleri kendimiz uretip veriyoruz (Chroma'nin
        # varsayilan/indirilen embedding modelini kullanmasini istemiyoruz).
        collection = self.chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=None)

        ids = list(documents.keys())
        texts = list(documents.values())
        embeddings = [self.embedding_provider.embed(text) for text in texts]

        collection.add(ids=ids, embeddings=embeddings, documents=texts)
        return collection.count()


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
    indexer = KnowledgeBaseIndexer(embedding_provider, chroma_client, collection_name)
    count = indexer.reindex(documents)

    print(f"Koleksiyon '{collection_name}' yeniden olusturuldu -> {chroma_path} ({count} belge)")


if __name__ == "__main__":
    main()

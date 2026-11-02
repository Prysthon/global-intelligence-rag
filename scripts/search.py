import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.config import Settings
from global_intelligence.database.connection import Database
from global_intelligence.database.repository import ChunkRepository
from global_intelligence.embeddings.service import EmbeddingService
from global_intelligence.retrieval.pipeline import RetrievalPipeline


def main():
    settings = Settings()
    embedding_service = EmbeddingService(
        settings.dense_model, settings.sparse_model, settings.colbert_model
    )
    database = Database(
        database_url=settings.database_url,
        dense_dimension=settings.dense_dimension,
        table_name=settings.table_name,
    )
    repository = ChunkRepository(database.engine, database.table)
    pipeline = RetrievalPipeline(embedding_service, repository)
    query_text = "what are the main financial risks?"
    response = pipeline.search(query_text)

    for result in response.results:
        print(f"Empresa: {result.metadata.company_name} ({result.metadata.ticker})")
        print(f"Relatório: {result.metadata.form_type}")
        print(f"Fonte: {result.metadata.source_url}")
        print(f"ColBERT Score: {result.colbert_score:.6f}")
        print(f"RRF Score: {result.score:.6f}")
        print(f"Texto: {result.content[:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()

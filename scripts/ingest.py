import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.config import Settings
from global_intelligence.database.connection import Database
from global_intelligence.database.repository import ChunkRepository
from global_intelligence.embeddings.service import EmbeddingService
from global_intelligence.ingestion.chunker import SemanticChunker
from global_intelligence.ingestion.edgar_client import EdgarClient
from global_intelligence.ingestion.pipeline import IngestionPipeline


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
    database.setup()
    
    repository = ChunkRepository(database.engine, database.table)
    client = EdgarClient(settings.edgar_identity)
    chunker = SemanticChunker(model_name=settings.dense_model)
    pipeline = IngestionPipeline(client, chunker, embedding_service, repository)

    chunks = pipeline.run(settings.tickers, settings.form_types)
    inserted_count = repository.count_chunks([chunk.id for chunk in chunks])

    print(f"Ingestão concluída: {inserted_count} chunks confirmados no banco.")


if __name__ == "__main__":
    main()

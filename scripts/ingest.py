import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.config import EDGAR_IDENTITY, FORM_TYPES, TICKERS
from global_intelligence.database.connection import create_db_engine
from global_intelligence.database.repository import count_chunks
from global_intelligence.database.schema import setup_database
from global_intelligence.embeddings.encoder import EmbeddingEncoder
from global_intelligence.ingestion.chunker import SemanticChunker
from global_intelligence.ingestion.edgar_client import EdgarClient
from global_intelligence.ingestion.pipeline import ingest_filings


def main():
    encoder = EmbeddingEncoder()
    engine = create_db_engine()
    setup_database(engine)
    client = EdgarClient(EDGAR_IDENTITY)
    chunker = SemanticChunker()

    data = ingest_filings(TICKERS, FORM_TYPES, client, chunker, encoder, engine)

    inserted_count = count_chunks(engine, [chunk["id"] for chunk in data])

    print(f"Ingestão concluída: {inserted_count} chunks confirmados no banco.")


if __name__ == "__main__":
    main()

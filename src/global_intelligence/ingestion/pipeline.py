import uuid

from global_intelligence.database.repository import ChunkRepository
from global_intelligence.embeddings.service import EmbeddingService
from global_intelligence.ingestion.chunker import SemanticChunker
from global_intelligence.ingestion.edgar_client import EdgarClient
from global_intelligence.models import DocumentChunk


class IngestionPipeline:
    def __init__(
        self,
        client: EdgarClient,
        chunker: SemanticChunker,
        embedding_service: EmbeddingService,
        repository: ChunkRepository,
    ):
        self._client = client
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._repository = repository

    def run(self, tickers: list[str], form_types: list[str]) -> list[DocumentChunk]:
        chunks = []

        for ticker in tickers:
            for form_type in form_types:
                filing = self._client.fetch_filing_data(ticker, form_type)
                content = self._client.get_combined_text(filing)

                for text in self._chunker.create_chunks(content):
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            content=text,
                            metadata=filing.metadata,
                            embeddings=self._embedding_service.embed_passage(text),
                        )
                    )

        self._repository.insert_chunks(chunks)
        return chunks

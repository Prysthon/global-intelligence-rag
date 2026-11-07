from fastapi import APIRouter

from global_intelligence.config import Settings
from global_intelligence.database.connection import Database
from global_intelligence.database.repository import ChunkRepository
from global_intelligence.embeddings.service import EmbeddingService
from global_intelligence.models import SearchRequest, SearchResponse
from global_intelligence.retrieval.pipeline import RetrievalPipeline

router = APIRouter(tags=["search"])

settings = Settings()
database = Database(
    database_url=settings.database_url,
    dense_dimension=settings.dense_dimension,
    table_name=settings.table_name,
)
embedding_service = EmbeddingService(
    settings.dense_model, settings.sparse_model, settings.colbert_model
)
repository = ChunkRepository(database.engine, database.table)
pipeline = RetrievalPipeline(embedding_service, repository)


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    return pipeline.search(payload.query, k=payload.k)

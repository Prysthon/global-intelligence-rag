from global_intelligence.database.repository import ChunkRepository
from global_intelligence.embeddings.service import EmbeddingService
from global_intelligence.models import SearchResponse
from global_intelligence.retrieval.fusion import reciprocal_rank_fusion
from global_intelligence.retrieval.reranker import rerank
from global_intelligence.retrieval.sparse import rank_sparse


class RetrievalPipeline:
    def __init__(
        self, embedding_service: EmbeddingService, repository: ChunkRepository
    ):
        self._embedding_service = embedding_service
        self._repository = repository

    def search(self, query: str, k: int = 3) -> SearchResponse:
        embeddings = self._embedding_service.embed_query(query)
        dense_results = self._repository.search_dense(embeddings.dense)
        chunks = self._repository.get_chunks()
        sparse_results = rank_sparse(embeddings.sparse, chunks)

        results = reciprocal_rank_fusion(dense_results, sparse_results)[:20]
        final_results = rerank(embeddings.colbert, results)[:k]
        max_score = max((result.colbert_score for result in final_results), default=0.0)

        normalized_results = [
            result.model_copy(
                update={
                    "colbert_score": result.colbert_score / max_score
                    if max_score
                    else 0.0
                }
            )
            for result in final_results
        ]
        return SearchResponse(results=normalized_results)

import numpy as np

from global_intelligence.models import SearchResult


def colbert_score(query_embedding: np.ndarray, document_embedding: np.ndarray) -> float:
    similarity = query_embedding @ document_embedding.T
    max_similarities = similarity.max(axis=1)
    return float(max_similarities.sum())


def rerank(
    query_colbert: list[list[float]], results: list[SearchResult]
) -> list[SearchResult]:
    query_embedding = np.array(query_colbert, dtype=np.float32)
    reranked_results = []

    for result in results:
        document_embedding = np.array(result.colbert_embedding, dtype=np.float32)
        score = colbert_score(query_embedding, document_embedding)
        reranked_results.append(
            result.model_copy(update={"colbert_score": float(score)})
        )

    return sorted(
        reranked_results, key=lambda result: result.colbert_score, reverse=True
    )

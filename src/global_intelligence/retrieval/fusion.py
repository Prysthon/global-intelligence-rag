from global_intelligence.models import SearchResult


def reciprocal_rank_fusion(
    dense_results: list[SearchResult], sparse_results: list[SearchResult], k: int = 1
) -> list[SearchResult]:
    scores = {}
    documents = {}

    for results in (dense_results, sparse_results):
        for rank, result in enumerate(results, start=1):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank)
            documents[result.id] = result

    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return [
        documents[chunk_id].model_copy(
            update={"distance": None, "score": score, "colbert_score": None}
        )
        for chunk_id, score in ranked_results
    ]

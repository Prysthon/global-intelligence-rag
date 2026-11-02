from global_intelligence.models import DocumentChunk, SearchResult, SparseEmbedding


def _sparse_dot_product(
    query_sparse: SparseEmbedding, document_sparse: SparseEmbedding
) -> float:
    query_vector = dict(zip(query_sparse.indices, query_sparse.values))
    document_vector = dict(zip(document_sparse.indices, document_sparse.values))

    return sum(
        query_value * document_vector.get(index, 0)
        for index, query_value in query_vector.items()
    )


def rank_sparse(
    query_sparse: SparseEmbedding, chunks: list[DocumentChunk]
) -> list[SearchResult]:
    sparse_results = []

    for chunk in chunks:
        score = _sparse_dot_product(query_sparse, chunk.embeddings.sparse)
        sparse_results.append(
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
                colbert_embedding=chunk.embeddings.colbert,
                score=score,
            )
        )

    return sorted(sparse_results, key=lambda result: result.score, reverse=True)[:10]

from global_intelligence.database.repository import get_chunks, search_dense
from global_intelligence.retrieval.fusion import reciprocal_rank_fusion
from global_intelligence.retrieval.reranker import rerank
from global_intelligence.retrieval.sparse import rank_sparse


def search(query: str, encoder, engine):
    query_dense, query_sparse, query_colbert = encoder.embed_query(query)

    dense_results = search_dense(engine, query_dense)
    rows = get_chunks(engine)
    sparse_results = rank_sparse(query_sparse, rows)

    results = reciprocal_rank_fusion(dense_results, sparse_results)[:20]
    reranked_results = rerank(query_colbert, results)

    return reranked_results[:3]

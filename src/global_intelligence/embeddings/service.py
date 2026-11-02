from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)

from global_intelligence.models import Embeddings, SparseEmbedding


class EmbeddingService:
    def __init__(self, dense_model: str, sparse_model: str, colbert_model: str):
        self.dense_model = TextEmbedding(dense_model)
        self.sparse_model = SparseTextEmbedding(sparse_model)
        self.colbert_model = LateInteractionTextEmbedding(colbert_model)

    def embed_passage(self, chunk: str) -> Embeddings:
        dense_embedding = next(iter(self.dense_model.passage_embed([chunk]))).tolist()

        sparse = next(iter(self.sparse_model.passage_embed([chunk])))
        sparse_embedding = SparseEmbedding(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

        colbert_embedding = next(
            iter(self.colbert_model.passage_embed([chunk]))
        ).tolist()

        return Embeddings(
            dense=dense_embedding,
            sparse=sparse_embedding,
            colbert=colbert_embedding,
        )

    def embed_query(self, query: str) -> Embeddings:
        dense = next(iter(self.dense_model.query_embed([query]))).tolist()

        sparse_raw = next(iter(self.sparse_model.query_embed([query])))
        sparse = SparseEmbedding(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist(),
        )

        colbert = next(iter(self.colbert_model.query_embed(query)))

        return Embeddings(dense=dense, sparse=sparse, colbert=colbert.tolist())

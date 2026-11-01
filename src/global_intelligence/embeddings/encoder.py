from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)

from global_intelligence.config import COLBERT_MODEL, DENSE_MODEL, SPARSE_MODEL


class EmbeddingEncoder:
    def __init__(self):
        self.dense_model = TextEmbedding(DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(SPARSE_MODEL)
        self.colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

    def embed_passage(self, chunk: str) -> dict:
        dense_embedding = next(iter(self.dense_model.passage_embed([chunk]))).tolist()

        sparse = next(iter(self.sparse_model.passage_embed([chunk])))
        sparse_embedding = {
            "indices": sparse.indices.tolist(),
            "values": sparse.values.tolist(),
        }

        colbert_embedding = next(
            iter(self.colbert_model.passage_embed([chunk]))
        ).tolist()

        return {
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_embedding,
            "colbert_embedding": colbert_embedding,
        }

    def embed_query(self, query: str):
        dense = next(iter(self.dense_model.query_embed([query]))).tolist()

        sparse_raw = next(iter(self.sparse_model.query_embed([query])))
        sparse = {
            "indices": sparse_raw.indices.tolist(),
            "values": sparse_raw.values.tolist(),
        }

        colbert = next(iter(self.colbert_model.query_embed(query)))

        return dense, sparse, colbert

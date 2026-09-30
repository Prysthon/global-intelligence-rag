# %%
import os
import uuid

import numpy as np
from dotenv import load_dotenv
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

load_dotenv()

FILE_PATH = "../data/raw/AAPL_10-K_1A_temp.md"

dense_model = TextEmbedding(os.getenv("DENSE_MODEL"))
sparse_model = SparseTextEmbedding(os.getenv("SPARSE_MODEL"))
colbert_model = LateInteractionTextEmbedding(os.getenv("COLBERT_MODEL"))
# %%
engine = create_engine(os.getenv("DATABASE_URL"))

metadata = MetaData()

document_chunks = Table(
    "document_chunks",
    metadata,
    Column("id", String, primary_key=True),
    Column("content", String, nullable=False),
    Column(
        "dense_embedding",
        Vector(int(os.getenv("DENSE_DIMENSION"))),
        nullable=False,
    ),
    Column("sparse_embedding", JSONB, nullable=True),
    Column("colbert_embedding", JSONB, nullable=True),
    Column("source", String, nullable=False),
)


def setup_database():
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    metadata.drop_all(engine)
    metadata.create_all(engine)


setup_database()
# %%
with open(FILE_PATH, "r", encoding="UTF-8") as f:
    content = f.read()

paragraphs = content.split("\n")

chunks = [m.strip() for m in paragraphs if len(m.strip()) > 50]

# %%
data = []

for chunk in chunks:
    dense_embedding = next(iter(dense_model.passage_embed([chunk]))).tolist()

    sparse = next(iter(sparse_model.passage_embed([chunk])))

    sparse_embedding = {
        "indices": sparse.indices.tolist(),
        "values": sparse.values.tolist(),
    }

    colbert_embedding = next(iter(colbert_model.passage_embed([chunk]))).tolist()

    id = str(uuid.uuid4())

    data.append(
        {
            "id": id,
            "content": chunk,
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_embedding,
            "colbert_embedding": colbert_embedding,
            "source": FILE_PATH,
        }
    )
# %%
with engine.begin() as connection:
    connection.execute(
        insert(document_chunks),
        data,
    )

# %%
query_text = "what are the main financial risks?"

query_dense = next(iter(dense_model.query_embed([query_text]))).tolist()

query_sparse_raw = next(iter(sparse_model.query_embed([query_text])))

query_sparse = {
    "indices": query_sparse_raw.indices.tolist(),
    "values": query_sparse_raw.values.tolist(),
}

query_colbert = next(iter(colbert_model.query_embed(query_text)))

# %%
dense_distance = document_chunks.c.dense_embedding.cosine_distance(query_dense).label(
    "distance"
)

dense_query = (
    document_chunks.select()
    .add_columns(dense_distance)
    .order_by(dense_distance)
    .limit(10)
)

with engine.connect() as connection:
    dense_results = connection.execute(dense_query).mappings().all()

# %%
def sparse_dot_product(query_sparse, document_sparse):
    query_vector = dict(
        zip(
            query_sparse["indices"],
            query_sparse["values"],
        )
    )

    document_vector = dict(
        zip(
            document_sparse["indices"],
            document_sparse["values"],
        )
    )

    return sum(
        query_value * document_vector.get(index, 0)
        for index, query_value in query_vector.items()
    )


with engine.connect() as connection:
    rows = connection.execute(document_chunks.select()).mappings().all()

sparse_results = []

for row in rows:
    score = sparse_dot_product(
        query_sparse,
        row["sparse_embedding"],
    )

    sparse_results.append(
        {
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "colbert_embedding": row["colbert_embedding"],
            "score": score,
        }
    )

sparse_results = sorted(
    sparse_results,
    key=lambda x: x["score"],
    reverse=True,
)[:10]


# %%
def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=1,
):
    scores = {}
    documents = {}

    for rank, result in enumerate(dense_results, start=1):
        doc_id = result["id"]

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        documents[doc_id] = {
            "id": doc_id,
            "content": result["content"],
            "source": result["source"],
            "colbert_embedding": result["colbert_embedding"],
        }

    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result["id"]

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        documents[doc_id] = {
            "id": doc_id,
            "content": result["content"],
            "source": result["source"],
            "colbert_embedding": result["colbert_embedding"],
        }

    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            **documents[doc_id],
            "score": score,
        }
        for doc_id, score in ranked_results
    ]


# %%
results = reciprocal_rank_fusion(
    dense_results,
    sparse_results,
)

results = results[:20]


# %%
def colbert_score(query_embedding, document_embedding):
    similarity = query_embedding @ document_embedding.T

    max_similarities = similarity.max(axis=1)

    return max_similarities.sum()


reranked_results = []

for result in results:
    document_colbert = np.array(
        result["colbert_embedding"],
        dtype=np.float32,
    )

    score = colbert_score(
        query_colbert,
        document_colbert,
    )

    reranked_results.append(
        {
            **result,
            "colbert_score": float(score),
        }
    )

reranked_results = sorted(
    reranked_results,
    key=lambda x: x["colbert_score"],
    reverse=True,
)

# %%
final_results = reranked_results[:3]

for result in final_results:
    print(f"ColBERT Score: {result['colbert_score']:.2f}")
    print(f"RRF Score: {result['score']:.6f}")
    print(f"Texto: {result['content'][:200]}...")
    print("-" * 80)


# %%

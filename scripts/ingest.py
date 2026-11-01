# %%
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.config import EDGAR_IDENTITY, FORM_TYPES, TICKERS
from global_intelligence.database.connection import create_db_engine
from global_intelligence.database.schema import document_chunks, setup_database
from global_intelligence.embeddings.encoder import EmbeddingEncoder
from global_intelligence.ingestion.chunker import SemanticChunker
from global_intelligence.ingestion.edgar_client import EdgarClient
from global_intelligence.ingestion.pipeline import ingest_filings

encoder = EmbeddingEncoder()
dense_model = encoder.dense_model
sparse_model = encoder.sparse_model
colbert_model = encoder.colbert_model
# %%
engine = create_db_engine()
setup_database(engine)
# %%
client = EdgarClient(EDGAR_IDENTITY)
chunker = SemanticChunker()

# %%
data = ingest_filings(TICKERS, FORM_TYPES, client, chunker, encoder, engine)

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
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "form_type": row["form_type"],
            "report_date": row["report_date"],
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
            "ticker": result["ticker"],
            "company_name": result["company_name"],
            "form_type": result["form_type"],
            "report_date": result["report_date"],
            "source": result["source"],
            "colbert_embedding": result["colbert_embedding"],
        }

    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result["id"]

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        documents[doc_id] = {
            "id": doc_id,
            "content": result["content"],
            "ticker": result["ticker"],
            "company_name": result["company_name"],
            "form_type": result["form_type"],
            "report_date": result["report_date"],
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

max_score = max(result["colbert_score"] for result in final_results)

for result in final_results:
    print(f"Empresa: {result['company_name']} ({result['ticker']})")
    print(f"Relatório: {result['form_type']}")
    print(f"Fonte: {result['source']}")
    print(f"ColBERT Score: {result['colbert_score'] / max_score:.6f}")
    print(f"RRF Score: {result['score']:.6f}")
    print(f"Texto: {result['content'][:200]}...")
    print("-" * 80)
# %%

import numpy as np


def colbert_score(query_embedding, document_embedding):
    similarity = query_embedding @ document_embedding.T

    max_similarities = similarity.max(axis=1)

    return max_similarities.sum()


def rerank(query_colbert, results):
    reranked_results = []

    for result in results:
        document_colbert = np.array(
            result["colbert_embedding"],
            dtype=np.float32,
        )

        score = colbert_score(query_colbert, document_colbert)

        reranked_results.append(
            {
                **result,
                "colbert_score": float(score),
            }
        )

    return sorted(reranked_results, key=lambda x: x["colbert_score"], reverse=True)

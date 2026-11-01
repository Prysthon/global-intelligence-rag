def _sparse_dot_product(query_sparse, document_sparse):
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


def rank_sparse(query_sparse, rows):
    sparse_results = []

    for row in rows:
        score = _sparse_dot_product(query_sparse, row["sparse_embedding"])

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

    return sorted(sparse_results, key=lambda x: x["score"], reverse=True)[:10]

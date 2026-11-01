def reciprocal_rank_fusion(dense_results, sparse_results, k=1):
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

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            **documents[doc_id],
            "score": score,
        }
        for doc_id, score in ranked_results
    ]

import uuid
from datetime import date

from global_intelligence.database.repository import insert_chunks


def ingest_filings(tickers, form_types, client, chunker, encoder, engine):
    data = []

    for ticker in tickers:
        for form_type in form_types:
            filing_data = client.fetch_filing_data(ticker, form_type)
            filing_metadata = filing_data["metadata"]
            report_date = date.fromisoformat(filing_metadata["report_date"])
            content = client.get_combined_text(filing_data)
            chunks = chunker.create_chunks(content)

            for chunk in chunks:
                embeddings = encoder.embed_passage(chunk)

                data.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": chunk,
                        **embeddings,
                        "ticker": filing_metadata["ticker"],
                        "company_name": filing_metadata["company_name"],
                        "form_type": filing_metadata["form_type"],
                        "report_date": report_date,
                        "source": filing_metadata["source_url"],
                    }
                )

    insert_chunks(engine, data)
    return data

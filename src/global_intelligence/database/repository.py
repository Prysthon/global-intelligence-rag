from collections.abc import Mapping
from typing import Any

from sqlalchemy import Engine, Table, func, insert, select

from global_intelligence.models import (
    DocumentChunk,
    Embeddings,
    FilingMetadata,
    SearchResult,
)


class ChunkRepository:
    def __init__(self, engine: Engine, table: Table):
        self._engine = engine
        self._table = table

    def insert_chunks(self, chunks: list[DocumentChunk]) -> None:
        records = [self._to_record(chunk) for chunk in chunks]
        with self._engine.begin() as connection:
            connection.execute(insert(self._table), records)

    def count_chunks(self, chunk_ids: list[str]) -> int:
        query = (
            select(func.count())
            .select_from(self._table)
            .where(self._table.c.id.in_(chunk_ids))
        )

        with self._engine.connect() as connection:
            return connection.execute(query).scalar_one()

    def search_dense(self, query_dense: list[float]) -> list[SearchResult]:
        dense_distance = self._table.c.dense_embedding.cosine_distance(
            query_dense
        ).label("distance")
        query = (
            self._table.select()
            .add_columns(dense_distance)
            .order_by(dense_distance)
            .limit(10)
        )

        with self._engine.connect() as connection:
            rows = connection.execute(query).mappings().all()

        return [
            SearchResult(
                id=row["id"],
                content=row["content"],
                metadata=self._metadata_from_record(row),
                colbert_embedding=row["colbert_embedding"],
                distance=row["distance"],
            )
            for row in rows
        ]

    def get_chunks(self) -> list[DocumentChunk]:
        with self._engine.connect() as connection:
            rows = connection.execute(self._table.select()).mappings().all()

        return [
            DocumentChunk(
                id=row["id"],
                content=row["content"],
                metadata=self._metadata_from_record(row),
                embeddings=Embeddings(
                    dense=row["dense_embedding"],
                    sparse=row["sparse_embedding"],
                    colbert=row["colbert_embedding"],
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _to_record(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "content": chunk.content,
            "dense_embedding": chunk.embeddings.dense,
            "sparse_embedding": chunk.embeddings.sparse.model_dump(),
            "colbert_embedding": chunk.embeddings.colbert,
            "ticker": chunk.metadata.ticker,
            "company_name": chunk.metadata.company_name,
            "form_type": chunk.metadata.form_type,
            "report_date": chunk.metadata.report_date,
            "source": chunk.metadata.source_url,
        }

    @staticmethod
    def _metadata_from_record(row: Mapping[str, Any]) -> FilingMetadata:
        return FilingMetadata(
            ticker=row["ticker"],
            company_name=row["company_name"],
            form_type=row["form_type"],
            report_date=row["report_date"],
            source_url=row["source"],
        )

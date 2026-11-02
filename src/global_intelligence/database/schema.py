from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Date, Engine, MetaData, String, Table, text
from sqlalchemy.dialects.postgresql import JSONB


def create_document_chunks(
    metadata: MetaData, dense_dimension: int, table_name: str
) -> Table:
    return Table(
        table_name,
        metadata,
        Column("id", String, primary_key=True),
        Column("content", String, nullable=False),
        Column("dense_embedding", Vector(dense_dimension), nullable=False),
        Column("sparse_embedding", JSONB, nullable=True),
        Column("colbert_embedding", JSONB, nullable=True),
        Column("ticker", String, nullable=False),
        Column("company_name", String, nullable=False),
        Column("form_type", String, nullable=False),
        Column("report_date", Date, nullable=False),
        Column("source", String, nullable=False),
    )


def setup_schema(engine: Engine, metadata: MetaData) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    metadata.drop_all(engine)
    metadata.create_all(engine)

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Date, MetaData, String, Table, text
from sqlalchemy.dialects.postgresql import JSONB

from global_intelligence.config import DENSE_DIMENSION

metadata = MetaData()

document_chunks = Table(
    "document_chunks",
    metadata,
    Column("id", String, primary_key=True),
    Column("content", String, nullable=False),
    Column(
        "dense_embedding",
        Vector(int(DENSE_DIMENSION)),
        nullable=False,
    ),
    Column("sparse_embedding", JSONB, nullable=True),
    Column("colbert_embedding", JSONB, nullable=True),
    Column("ticker", String, nullable=False),
    Column("company_name", String, nullable=False),
    Column("form_type", String, nullable=False),
    Column("report_date", Date, nullable=False),
    Column("source", String, nullable=False),
)


def setup_database(engine):
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    metadata.drop_all(engine)
    metadata.create_all(engine)

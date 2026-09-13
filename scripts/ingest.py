# %%
import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding
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

load_dotenv()

FILE_PATH = "../data/raw/AAPL_10-K_1A_temp.md"

embedding_model = TextEmbedding(os.getenv("EMBEDDING_MODEL"))
# %%
engine = create_engine(os.getenv("DATABASE_URL"))

metadata = MetaData()

document_chunks = Table(
    "document_chunks",
    metadata,
    Column("id", String, primary_key=True),
    Column("content", String, nullable=False),
    Column("embedding", Vector(int(os.getenv("EMBEDDING_DIMENSION"))), nullable=False),
    Column("source", String, nullable=False),
)

# documents = Table(
#     "documents",
#     metadata,
#     Column("id", Integer, primary_key=True),
#     Column("player", String, nullable=False),
#     Column("title", String, nullable=False),
#     Column("source_url", String, nullable=False),
#     Column("published_at", Date, nullable=False),
#     Column("document_type", String, nullable=False),
# )


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
    embedding = next(iter(embedding_model.passage_embed([chunk]))).tolist()
    id = uuid.uuid4()

    data.append({"id": id, "content": chunk, "embedding": embedding, "source": FILE_PATH})
# %%
with engine.begin() as connection:
    connection.execute(
        insert(document_chunks),
        data,
    )
# %%

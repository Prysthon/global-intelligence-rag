from sqlalchemy import insert

from global_intelligence.database.schema import document_chunks


def insert_chunks(engine, data):
    with engine.begin() as connection:
        connection.execute(insert(document_chunks), data)

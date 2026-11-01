from sqlalchemy import func, insert, select

from global_intelligence.database.schema import document_chunks


def insert_chunks(engine, data):
    with engine.begin() as connection:
        connection.execute(insert(document_chunks), data)


def count_chunks(engine, chunk_ids):
    query = (
        select(func.count())
        .select_from(document_chunks)
        .where(document_chunks.c.id.in_(chunk_ids))
    )

    with engine.connect() as connection:
        return connection.execute(query).scalar_one()


def search_dense(engine, query_dense):
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
        return connection.execute(dense_query).mappings().all()


def get_chunks(engine):
    with engine.connect() as connection:
        return connection.execute(document_chunks.select()).mappings().all()

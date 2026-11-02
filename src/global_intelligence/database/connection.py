from sqlalchemy import MetaData, create_engine

from global_intelligence.database.schema import create_document_chunks, setup_schema


class Database:
    def __init__(self, database_url: str, dense_dimension: int, table_name: str):
        self.engine = create_engine(database_url)
        self.metadata = MetaData()
        self.table = create_document_chunks(self.metadata, dense_dimension, table_name)

    def setup(self) -> None:
        setup_schema(self.engine, self.metadata)

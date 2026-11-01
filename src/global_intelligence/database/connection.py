from sqlalchemy import create_engine

from global_intelligence.config import DATABASE_URL


def create_db_engine():
    return create_engine(DATABASE_URL)

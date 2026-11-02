import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.config import Settings
from global_intelligence.database.connection import Database


def main():
    settings = Settings()
    database = Database(
        database_url=settings.database_url,
        dense_dimension=settings.dense_dimension,
        table_name=settings.table_name,
    )
    database.setup()


if __name__ == "__main__":
    main()

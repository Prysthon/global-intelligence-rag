import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_intelligence.database.connection import create_db_engine
from global_intelligence.database.schema import setup_database

if __name__ == "__main__":
    engine = create_db_engine()
    setup_database(engine)

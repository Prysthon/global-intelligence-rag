import os

from dotenv import load_dotenv

load_dotenv()

TICKERS = ["AAPL"]
FORM_TYPES = ["10-K", "10-Q"]

DATABASE_URL = os.getenv("DATABASE_URL")
EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY")
DENSE_MODEL = os.getenv("DENSE_MODEL")
DENSE_DIMENSION = os.getenv("DENSE_DIMENSION")
SPARSE_MODEL = os.getenv("SPARSE_MODEL")
COLBERT_MODEL = os.getenv("COLBERT_MODEL")

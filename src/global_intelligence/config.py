from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    database_url: str = Field(
        description="URL de conexão SQLAlchemy do PostgreSQL.", repr=False
    )
    table_name: str = Field(
        default="document_chunks",
        description="Nome da tabela que armazena os chunks e seus embeddings.",
    )

    tickers: list[str] = Field(
        default_factory=lambda: ["AAPL"],
        description="Tickers das empresas consultadas na ingestão.",
    )
    form_types: list[str] = Field(
        default_factory=lambda: ["10-K", "10-Q"],
        description="Tipos de relatório consultados para cada empresa.",
    )
    edgar_identity: str = Field(description="Identificação enviada nas consultas ao EDGAR.")

    dense_model: str = Field(description="Nome do modelo de embeddings densos.")
    dense_dimension: int = Field(description="Dimensão dos vetores densos armazenados.")
    sparse_model: str = Field(description="Nome do modelo de embeddings esparsos.")
    colbert_model: str = Field(description="Nome do modelo ColBERT usado no reranking.")

    groq_api_key: str = Field(description="Chave de acesso à API da Groq.", repr=False)
    groq_model: str = Field(description="Nome do modelo de linguagem hospedado na Groq.")

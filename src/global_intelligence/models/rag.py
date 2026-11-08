from pydantic import BaseModel, Field

from global_intelligence.models.filing import FilingMetadata


class RagRequest(BaseModel):
    query: str = Field(min_length=1, description="Texto da consulta.")
    k: int = Field(default=3, ge=1, description="Limite de trechos usados como contexto.")


class RagResult(BaseModel):
    content: str = Field(description="Texto do trecho recuperado.")
    metadata: FilingMetadata = Field(description="Relatório que originou o trecho.")
    colbert_score: float | None = Field(
        default=None, description="Pontuação ColBERT normalizada do trecho."
    )


class RagResponse(BaseModel):
    query: str = Field(min_length=1, description="Texto da consulta.")
    answer: str = Field(min_length=1, description="Resposta gerada a partir dos documentos.")
    results: list[RagResult] = Field(
        description="Trechos recuperados e usados como contexto para a resposta."
    )

from pydantic import BaseModel, Field

from global_intelligence.models.filing import FilingMetadata


class SearchResult(BaseModel):
    id: str = Field(description="Identificador do chunk recuperado.")
    content: str = Field(description="Texto do chunk recuperado.")
    metadata: FilingMetadata = Field(description="Relatório que originou o resultado.")
    colbert_embedding: list[list[float]] = Field(description="Matriz usada no reranking.")
    distance: float | None = Field(default=None, description="Distância cosseno na busca densa.")
    score: float | None = Field(default=None, description="Pontuação da busca esparsa ou do RRF.")
    colbert_score: float | None = Field(
        default=None,
        description="Pontuação ColBERT, normalizada pelo maior score nos resultados finais.",
    )


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(
        description="Resultados finais da busca, ordenados por relevância."
    )

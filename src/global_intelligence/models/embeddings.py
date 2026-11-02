from pydantic import BaseModel, Field


class SparseEmbedding(BaseModel):
    indices: list[int] = Field(description="Índices das dimensões presentes no vetor esparso.")
    values: list[float] = Field(description="Pesos correspondentes aos índices do vetor esparso.")


class Embeddings(BaseModel):
    dense: list[float] = Field(description="Vetor denso do texto.")
    sparse: SparseEmbedding = Field(description="Vetor esparso do texto.")
    colbert: list[list[float]] = Field(description="Matriz ColBERT: um vetor por token.")

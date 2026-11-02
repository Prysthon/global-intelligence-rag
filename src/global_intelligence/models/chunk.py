from pydantic import BaseModel, Field

from global_intelligence.models.embeddings import Embeddings
from global_intelligence.models.filing import FilingMetadata


class DocumentChunk(BaseModel):
    id: str = Field(description="Identificador único do chunk.")
    content: str = Field(description="Texto do chunk.")
    metadata: FilingMetadata = Field(description="Relatório que originou o chunk.")
    embeddings: Embeddings = Field(description="Embeddings gerados para o conteúdo.")

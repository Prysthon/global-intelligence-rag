from datetime import date

from pydantic import BaseModel, Field


class FilingMetadata(BaseModel):
    ticker: str = Field(description="Ticker da empresa, como AAPL.")
    company_name: str = Field(description="Nome da empresa informado no relatório.")
    form_type: str = Field(description="Tipo de relatório, como 10-K ou 10-Q.")
    report_date: date = Field(description="Data de encerramento do período reportado.")
    source_url: str = Field(description="URL do documento original na SEC.")


class FilingData(BaseModel):
    metadata: FilingMetadata = Field(description="Identificação do relatório coletado.")
    items: dict[str, str] = Field(description="Texto de cada seção extraída do relatório.")

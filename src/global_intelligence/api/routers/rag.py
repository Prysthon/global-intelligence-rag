from fastapi import APIRouter

from global_intelligence.api.routers.search import pipeline, settings
from global_intelligence.models import RagRequest, RagResponse
from global_intelligence.retrieval.rag import RagPipeline

router = APIRouter(tags=["rag"])
rag_pipeline = RagPipeline(settings, pipeline)


@router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    return rag_pipeline.generate_answer(payload.query, k=payload.k)

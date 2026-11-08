from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from global_intelligence.config import Settings
from global_intelligence.models.rag import RagResponse, RagResult
from global_intelligence.retrieval.pipeline import RetrievalPipeline


class RagPipeline:
    def __init__(self, settings: Settings, retrieval_pipeline: RetrievalPipeline):
        self._retrieval_pipeline = retrieval_pipeline
        model = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the following financial documents, answer the question.",
                ),
                (
                    "user",
                    "question: {query}\n\n<context>\n{context}\n</context>",
                ),
            ]
        )
        self._chain = prompt | model

    def generate_answer(self, query: str, k: int = 3) -> RagResponse:
        response = self._retrieval_pipeline.search(query, k=k)
        results = [
            RagResult(
                content=result.content,
                metadata=result.metadata,
                colbert_score=result.colbert_score,
            )
            for result in response.results
        ]

        context = "\n\n".join(
            f"[{index}] {result.metadata.company_name} ({result.metadata.ticker})\n"
            f"Relatório: {result.metadata.form_type} | "
            f"Período encerrado em: {result.metadata.report_date}\n"
            f"Fonte: {result.metadata.source_url}\n"
            f"Trecho: {result.content}"
            for index, result in enumerate(results, start=1)
        )
        
        answer = self._chain.invoke({"query": query, "context": context})
        print(answer)
        return RagResponse(query=query, answer=answer.content, results=results)

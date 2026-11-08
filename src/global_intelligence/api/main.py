from fastapi import FastAPI

from global_intelligence.api.routers import health, rag, search

app = FastAPI(title="Global Intelligence API")

app.include_router(health.router)
app.include_router(search.router)
app.include_router(rag.router)

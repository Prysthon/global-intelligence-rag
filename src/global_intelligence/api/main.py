from fastapi import FastAPI

from global_intelligence.api.routers import health, search

app = FastAPI(title="Global Intelligence API")

app.include_router(health.router)
app.include_router(search.router)
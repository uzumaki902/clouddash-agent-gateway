"""
main.py — CloudDash Agent Gateway entrypoint.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI available at: http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI

# TODO: import routers once implemented
# from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | trace=%(trace_id)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="CloudDash Agent Gateway",
    description=(
        "Multi-agent customer support orchestration backend. "
        "Routes user conversations through Triage → Specialist agents "
        "using LangGraph Supervisor Pattern with Supabase RAG."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["ops"])
async def health_check() -> dict[str, str]:
    """Liveness probe — returns 200 when the service is running."""
    return {"status": "ok", "service": "clouddash-agent-gateway"}


# app.include_router(router, prefix="/api/v1")  # uncomment when ready

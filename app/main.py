"""
app/main.py
FastAPI 应用入口。
"""

from fastapi import FastAPI

from app.api.endpoints.chat import router as chat_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="基于 LLM 的智能电商导购 Agent API",
    version="0.1.0",
)

app.include_router(chat_router, prefix="/api", tags=["Chat"])


@app.get("/health")
async def health_check():
    from app.agents.fallback import model_router
    return {
        "status": "ok",
        "app": settings.app_name,
        "models": model_router.get_health_report(),
    }

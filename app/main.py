"""
app/main.py
FastAPI 应用入口。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.endpoints.chat import router as chat_router
from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="Main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理：启动时初始化资源，关闭时释放资源。"""
    from app.core.db_pool import DBPool

    # ── 启动阶段 ──
    try:
        await DBPool.init_pool()
        _logger.info("MySQL 连接池初始化完成")
    except Exception:  # noqa: BLE001
        _logger.warning("MySQL 连接池初始化失败，将使用直连模式")

    yield

    # ── 关闭阶段 ──
    try:
        await DBPool.close_pool()
        _logger.info("MySQL 连接池已关闭")
    except Exception:  # noqa: BLE001
        _logger.warning("MySQL 连接池关闭异常")


app = FastAPI(
    title=settings.app_name,
    description="基于 LLM 的智能电商导购 Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/api", tags=["Chat"])


@app.get("/health")
async def health_check():
    from app.core.llm import get_model_router
    from app.core.db_pool import DBPool

    router = get_model_router()
    pool_status = await DBPool.pool_status()

    return {
        "status": "ok",
        "app": settings.app_name,
        "models": router.get_health_report(),
        "mysql_pool": pool_status,
    }

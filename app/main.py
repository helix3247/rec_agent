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


async def _check_mysql() -> tuple[bool, str]:
    """校验 MySQL 连通性。"""
    try:
        from app.core.db_pool import DBPool

        pool = DBPool.get_pool()
        if pool is None:
            return False, "连接池未初始化"
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _check_es() -> tuple[bool, str]:
    """校验 Elasticsearch 连通性。"""
    try:
        from elasticsearch import Elasticsearch

        es_cfg = settings.es
        es_kwargs: dict = {"hosts": es_cfg.es_host}
        if es_cfg.es_host.startswith("https"):
            es_kwargs["verify_certs"] = False
            es_kwargs["ssl_show_warn"] = False
        if es_cfg.es_username and es_cfg.es_password:
            es_kwargs["basic_auth"] = (es_cfg.es_username, es_cfg.es_password)

        es = Elasticsearch(**es_kwargs)
        health = es.cluster.health(params={"timeout": "5s"})
        return True, f"status={health.get('status', 'unknown')}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _check_milvus() -> tuple[bool, str]:
    """校验 Milvus 连通性。"""
    try:
        from pymilvus import connections, utility

        milvus_cfg = settings.milvus
        connections.connect(
            alias="startup_check",
            host=milvus_cfg.milvus_host,
            port=str(milvus_cfg.milvus_port),
            timeout=5,
        )
        collections = utility.list_collections(using="startup_check")
        connections.disconnect("startup_check")
        return True, f"collections={len(collections)}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _check_redis() -> tuple[bool, str]:
    """校验 Redis 连通性。"""
    try:
        import redis

        redis_cfg = settings.redis
        r = redis.Redis(
            host=redis_cfg.redis_host,
            port=redis_cfg.redis_port,
            db=redis_cfg.redis_db,
            password=redis_cfg.redis_password or None,
            socket_timeout=3,
            decode_responses=True,
        )
        pong = r.ping()
        r.close()
        return bool(pong), "PONG" if pong else "无响应"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _check_llm() -> tuple[bool, str]:
    """校验主模型 LLM 接口可用性（轻量 ping）。"""
    try:
        from app.core.llm import get_llm

        llm = get_llm("primary", temperature=0)
        response = llm.invoke("ping")
        return True, f"response_length={len(response.content)}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


async def _run_startup_checks() -> None:
    """执行所有数据源连通性检查，记录结果日志。"""
    checks = {
        "MySQL": _check_mysql,
        "Elasticsearch": _check_es,
        "Milvus": _check_milvus,
        "Redis": _check_redis,
        "LLM": _check_llm,
    }

    _logger.info("── 启动前连通性检查 ──")
    all_ok = True

    for name, check_fn in checks.items():
        try:
            import asyncio
            if asyncio.iscoroutinefunction(check_fn):
                ok, detail = await check_fn()
            else:
                ok, detail = check_fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, str(e)

        if ok:
            _logger.info("  ✓ {} 连通性检查通过 | {}", name, detail)
        else:
            all_ok = False
            _logger.warning("  ✗ {} 连通性检查失败 | {}", name, detail)

    if all_ok:
        _logger.info("── 所有连通性检查通过 ──")
    else:
        _logger.warning("── 部分服务不可用，应用仍将启动但功能可能受限 ──")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理：启动时初始化资源并校验连通性，关闭时释放资源。"""
    from app.core.db_pool import DBPool

    # ── 启动阶段 ──
    try:
        await DBPool.init_pool()
        _logger.info("MySQL 连接池初始化完成")
    except Exception:  # noqa: BLE001
        _logger.warning("MySQL 连接池初始化失败，将使用直连模式")

    await _run_startup_checks()

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

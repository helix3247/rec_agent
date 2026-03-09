"""
app/core/checkpoint.py
LangGraph Checkpoint 持久化 —— 基于 Redis 的状态恢复能力。

使用 AsyncRedisSaver 将 LangGraph 的每次状态快照自动持久化到 Redis，
实现企业级的对话状态恢复：
    - 进程重启后恢复中断的对话
    - 多实例部署时共享会话状态
    - 自动 TTL 过期清理，避免存储膨胀

与 dialog.py 的职责边界：
    - checkpoint: LangGraph 框架级状态持久化（自动，Graph 内部的 messages/slots/intent 等全量快照）
    - dialog.py:  业务级对话记忆管理（手动，Redis 短期历史 + Milvus 长期记忆迁移）

两者并行运行、互不冲突：
    - checkpoint 由 LangGraph 框架自动管理，用于 Graph 级别的断点恢复
    - dialog.py 的 save_history/load_history 继续负责跨请求的对话历史拼接和长期记忆迁移
"""

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="Checkpoint")

_CHECKPOINT_TTL_MINUTES = 30


def _build_redis_url() -> str:
    """根据配置构建 Redis URL。"""
    password_part = f":{settings.redis.redis_password}@" if settings.redis.redis_password else ""
    return (
        f"redis://{password_part}"
        f"{settings.redis.redis_host}:{settings.redis.redis_port}"
        f"/{settings.redis.redis_db}"
    )


class _CheckpointerHolder:
    """封装 AsyncRedisSaver 单例的延迟初始化，避免模块级 global 变量。"""

    def __init__(self):
        self._instance: AsyncRedisSaver | None = None
        self._setup_done = False

    async def get(self) -> AsyncRedisSaver:
        """
        获取 AsyncRedisSaver 单例。

        首次调用时创建并初始化（asetup 创建索引），后续复用同一实例。
        配置 TTL 自动过期，与 dialog.py 的 _HISTORY_TTL (30 分钟) 保持一致。
        """
        if self._instance is not None and self._setup_done:
            return self._instance

        redis_url = _build_redis_url()
        safe_url = redis_url.replace(settings.redis.redis_password, "***") if settings.redis.redis_password else redis_url
        _logger.info("初始化 LangGraph Redis Checkpoint | url={}", safe_url)

        ttl_config = {
            "default_ttl": _CHECKPOINT_TTL_MINUTES,
            "refresh_on_read": True,
        }

        self._instance = AsyncRedisSaver(redis_url=redis_url, ttl=ttl_config)
        await self._instance.asetup()
        self._setup_done = True

        _logger.info("LangGraph Redis Checkpoint 初始化完成 | ttl={}min", _CHECKPOINT_TTL_MINUTES)
        return self._instance


_holder = _CheckpointerHolder()


async def get_checkpointer() -> AsyncRedisSaver:
    """获取全局 AsyncRedisSaver 单例。"""
    return await _holder.get()

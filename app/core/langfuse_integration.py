"""
app/core/langfuse_integration.py
Langfuse 可观测性集成模块。

提供:
    - Langfuse 客户端初始化（全局单例）
    - LangChain CallbackHandler 获取，用于自动上报 LLM/Tool 调用
    - Trace 管理工具函数（创建 trace、添加 span/generation/event）
"""

from typing import Any

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="Langfuse")

_langfuse_client = None
_initialized = False


def _init_langfuse():
    """懒初始化 Langfuse 客户端。"""
    global _langfuse_client, _initialized

    if _initialized:
        return

    _initialized = True
    lf_cfg = settings.langfuse

    if not lf_cfg.langfuse_enabled:
        _logger.info("Langfuse 已禁用（LANGFUSE_ENABLED=false）")
        return

    if not lf_cfg.langfuse_public_key or not lf_cfg.langfuse_secret_key:
        _logger.warning("Langfuse 密钥未配置，跳过初始化")
        return

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=lf_cfg.langfuse_public_key,
            secret_key=lf_cfg.langfuse_secret_key,
            host=lf_cfg.langfuse_host,
        )
        _logger.info("Langfuse 初始化成功 | host={}", lf_cfg.langfuse_host)
    except Exception as e:
        _logger.error("Langfuse 初始化失败 | error={}", str(e))
        _langfuse_client = None


def get_langfuse_client():
    """获取全局 Langfuse 客户端实例（可能为 None）。"""
    _init_langfuse()
    return _langfuse_client


def get_langfuse_callback(
    trace_id: str = "",
    user_id: str = "",
    session_id: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    获取 Langfuse LangChain CallbackHandler。

    用于在 LLM 调用时自动上报 Token 消耗、延迟等指标。
    若 Langfuse 未启用或初始化失败，返回 None。

    用法:
        callback = get_langfuse_callback(trace_id=trace_id)
        if callback:
            response = llm.invoke(messages, config={"callbacks": [callback]})
    """
    _init_langfuse()

    if _langfuse_client is None:
        return None

    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=settings.langfuse.langfuse_public_key,
            secret_key=settings.langfuse.langfuse_secret_key,
            host=settings.langfuse.langfuse_host,
            trace_name="rec-agent",
            trace_id=trace_id or None,
            user_id=user_id or None,
            session_id=session_id or None,
            tags=tags or [],
            trace_metadata=metadata or {},
        )
        return handler
    except Exception as e:
        _logger.warning("创建 Langfuse CallbackHandler 失败 | error={}", str(e))
        return None


def create_trace(
    trace_id: str,
    name: str = "rec-agent",
    user_id: str = "",
    session_id: str = "",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """
    手动创建一个 Langfuse Trace。

    用于需要细粒度控制的场景（如手动添加 span、event）。
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        trace = client.trace(
            id=trace_id,
            name=name,
            user_id=user_id or None,
            session_id=session_id or None,
            metadata=metadata or {},
            tags=tags or [],
        )
        return trace
    except Exception as e:
        _logger.warning("创建 Langfuse Trace 失败 | error={}", str(e))
        return None


def report_trace_metrics(
    trace_id: str,
    *,
    user_intent: str = "",
    route_path: list[str] | None = None,
    total_latency_ms: float = 0,
    token_usage: dict[str, int] | None = None,
    tool_call_stats: dict[str, Any] | None = None,
    task_status: str = "",
):
    """
    向已有 Trace 追加汇总指标信息（作为 event）。

    由 MonitorAgent 在请求结束时调用。
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(id=trace_id)
        trace.event(
            name="request_summary",
            metadata={
                "user_intent": user_intent,
                "agent_route_path": route_path or [],
                "total_latency_ms": total_latency_ms,
                "token_usage": token_usage or {},
                "tool_call_stats": tool_call_stats or {},
                "task_status": task_status,
            },
        )
        # 设置 token 级别的 usage 供 Langfuse 面板展示
        if token_usage:
            trace.update(
                metadata={
                    "total_latency_ms": total_latency_ms,
                    "task_status": task_status,
                },
            )
    except Exception as e:
        _logger.warning("上报 Trace 指标失败 | trace_id={} | error={}", trace_id, str(e))


def flush():
    """刷新 Langfuse 客户端缓冲区，确保所有数据上报完毕。"""
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass

"""
app/core/llm.py
LLM 客户端工厂。
支持 "primary" (DeepSeek) 和 "fallback" (OpenAI) 两种模型，
通过 SmartModelRouter 进行智能路由与自动降级。

能力:
    - 按任务复杂度（LIGHT/MEDIUM/HEAVY）智能选择模型
    - 记录模型健康指标（失败率、延迟、连续失败）
    - 主模型不健康时自动降级至备用模型
    - 支持超时控制参数
"""

import time
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="LLMClient")

ModelType = Literal["primary", "fallback"]

_DEFAULT_TIMEOUT = 60  # LLM 调用默认超时（秒）

# SmartModelRouter 全局单例（延迟导入避免循环引用）
_model_router = None


def get_model_router():
    """获取全局 SmartModelRouter 单例（复用 fallback.py 中的实例）。"""
    global _model_router
    if _model_router is None:
        from app.agents.fallback import model_router
        _model_router = model_router
    return _model_router


def get_llm(model_type: ModelType = "primary", **kwargs) -> ChatOpenAI:
    """
    获取 LLM 客户端实例。

    Args:
        model_type: "primary" 使用 DeepSeek, "fallback" 使用 OpenAI。
        **kwargs: 传递给 ChatOpenAI 的额外参数（如 temperature, max_tokens 等）。
    """
    llm_cfg = settings.llm

    timeout = kwargs.pop("timeout", _DEFAULT_TIMEOUT)

    if model_type == "primary":
        return ChatOpenAI(
            api_key=llm_cfg.llm_api_key,
            base_url=llm_cfg.llm_base_url,
            model=llm_cfg.llm_model,
            timeout=timeout,
            **kwargs,
        )
    else:
        return ChatOpenAI(
            api_key=llm_cfg.fallback_llm_api_key,
            base_url=llm_cfg.fallback_llm_base_url,
            model=llm_cfg.fallback_llm_model,
            timeout=timeout,
            **kwargs,
        )


def _record_to_router(model_type: str, success: bool, latency_ms: float):
    """将调用结果同步到智能路由器的健康指标。"""
    try:
        router = get_model_router()
        router.record_call(model_type, success, latency_ms)
    except Exception:
        pass


async def invoke_with_smart_routing(
    messages: list[BaseMessage],
    agent_name: str = "",
    intent: str = "",
    message_count: int = 0,
    **kwargs,
) -> str:
    """
    带智能路由和降级的 LLM 异步调用。

    根据 agent_name / intent 自动判断任务复杂度，选择合适的模型，
    首选模型失败后自动降级到另一个模型。

    Args:
        messages: LangChain 消息列表。
        agent_name: 调用方 Agent 名称（用于复杂度判定）。
        intent: 用户意图（用于复杂度判定）。
        message_count: 当前对话消息数（长对话会提升复杂度）。
        **kwargs: 传递给 ChatOpenAI 的额外参数。

    Returns:
        LLM 生成的文本内容。
    """
    router = get_model_router()
    return await router.invoke_with_smart_routing(
        messages,
        intent=intent,
        agent_name=agent_name,
        message_count=message_count,
        **kwargs,
    )


def invoke_with_smart_routing_sync(
    messages: list[BaseMessage],
    agent_name: str = "",
    intent: str = "",
    message_count: int = 0,
    **kwargs,
) -> str:
    """同步版本的智能路由 LLM 调用。"""
    router = get_model_router()
    return router.invoke_with_smart_routing_sync(
        messages,
        intent=intent,
        agent_name=agent_name,
        message_count=message_count,
        **kwargs,
    )


def _invoke_with_fallback_core(messages: list, *, is_async: bool, **kwargs):
    """
    带降级能力的 LLM 调用核心逻辑。

    返回协程（is_async=True）或直接结果（is_async=False）。
    将 primary -> fallback 的双层 try/except 提取为统一流程。
    """
    model_order: list[ModelType] = ["primary", "fallback"]
    model_names = {
        "primary": settings.llm.llm_model,
        "fallback": settings.llm.fallback_llm_model,
    }

    if is_async:
        async def _run():
            last_err: Exception | None = None
            for model_type in model_order:
                t0 = time.time()
                try:
                    llm = get_llm(model_type, **kwargs)
                    response = await llm.ainvoke(messages)
                    latency_ms = (time.time() - t0) * 1000
                    _record_to_router(model_type, True, latency_ms)
                    _logger.info(
                        "LLM 调用成功 | model={} | latency={}ms",
                        model_names[model_type], round(latency_ms),
                    )
                    return response.content
                except Exception as e:
                    latency_ms = (time.time() - t0) * 1000
                    _record_to_router(model_type, False, latency_ms)
                    _logger.warning(
                        "LLM 调用失败 | model={} | error={}",
                        model_names[model_type], str(e),
                    )
                    last_err = e
            raise last_err  # type: ignore[misc]
        return _run()
    else:
        last_err: Exception | None = None
        for model_type in model_order:
            t0 = time.time()
            try:
                llm = get_llm(model_type, **kwargs)
                response = llm.invoke(messages)
                latency_ms = (time.time() - t0) * 1000
                _record_to_router(model_type, True, latency_ms)
                _logger.info(
                    "LLM 调用成功 | model={} | latency={}ms",
                    model_names[model_type], round(latency_ms),
                )
                return response.content
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                _record_to_router(model_type, False, latency_ms)
                _logger.warning(
                    "LLM 调用失败 | model={} | error={}",
                    model_names[model_type], str(e),
                )
                last_err = e
        raise last_err  # type: ignore[misc]


async def invoke_with_fallback(messages: list, **kwargs) -> str:
    """带降级能力的 LLM 异步调用：先尝试主模型，失败后自动切换到 fallback。"""
    return await _invoke_with_fallback_core(messages, is_async=True, **kwargs)


def invoke_with_fallback_sync(messages: list, **kwargs) -> str:
    """带降级能力的 LLM 同步调用：先尝试主模型，失败后自动切换到 fallback。"""
    return _invoke_with_fallback_core(messages, is_async=False, **kwargs)

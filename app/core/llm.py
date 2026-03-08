"""
app/core/llm.py
LLM 客户端工厂。
支持 "primary" (DeepSeek) 和 "fallback" (OpenAI) 两种模型，
调用层自动降级：主模型失败时切换到 fallback。

升级版:
    - 集成 FallbackAgent 智能路由器的调用指标记录
    - 支持超时控制参数
"""

import time
from typing import Literal

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="LLMClient")

ModelType = Literal["primary", "fallback"]

_DEFAULT_TIMEOUT = 60  # LLM 调用默认超时（秒）


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
        from app.agents.fallback import model_router
        model_router.record_call(model_type, success, latency_ms)
    except Exception:
        pass


async def invoke_with_fallback(messages: list, **kwargs) -> str:
    """
    带降级能力的 LLM 调用：先尝试主模型，失败后自动切换到 fallback。

    Args:
        messages: LangChain 消息列表。
        **kwargs: 传递给 ChatOpenAI 的额外参数。

    Returns:
        LLM 生成的文本内容。
    """
    t0 = time.time()
    try:
        llm = get_llm("primary", **kwargs)
        response = await llm.ainvoke(messages)
        latency_ms = (time.time() - t0) * 1000
        _record_to_router("primary", True, latency_ms)
        _logger.info(
            "主模型调用成功 | model={} | latency={}ms",
            settings.llm.llm_model, round(latency_ms),
        )
        return response.content
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        _record_to_router("primary", False, latency_ms)
        _logger.warning(
            "主模型调用失败，准备降级 | model={} | error={}",
            settings.llm.llm_model,
            str(e),
        )
        t1 = time.time()
        try:
            llm_fallback = get_llm("fallback", **kwargs)
            response = await llm_fallback.ainvoke(messages)
            latency_ms = (time.time() - t1) * 1000
            _record_to_router("fallback", True, latency_ms)
            _logger.info(
                "降级成功，使用备用模型 | model={} | latency={}ms",
                settings.llm.fallback_llm_model, round(latency_ms),
            )
            return response.content
        except Exception as fallback_err:
            latency_ms = (time.time() - t1) * 1000
            _record_to_router("fallback", False, latency_ms)
            _logger.error(
                "备用模型也失败 | model={} | error={}",
                settings.llm.fallback_llm_model,
                str(fallback_err),
            )
            raise


def invoke_with_fallback_sync(messages: list, **kwargs) -> str:
    """
    同步版本的带降级 LLM 调用。
    """
    t0 = time.time()
    try:
        llm = get_llm("primary", **kwargs)
        response = llm.invoke(messages)
        latency_ms = (time.time() - t0) * 1000
        _record_to_router("primary", True, latency_ms)
        _logger.info(
            "主模型调用成功 | model={} | latency={}ms",
            settings.llm.llm_model, round(latency_ms),
        )
        return response.content
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        _record_to_router("primary", False, latency_ms)
        _logger.warning(
            "主模型调用失败，准备降级 | model={} | error={}",
            settings.llm.llm_model,
            str(e),
        )
        t1 = time.time()
        try:
            llm_fallback = get_llm("fallback", **kwargs)
            response = llm_fallback.invoke(messages)
            latency_ms = (time.time() - t1) * 1000
            _record_to_router("fallback", True, latency_ms)
            _logger.info(
                "降级成功，使用备用模型 | model={} | latency={}ms",
                settings.llm.fallback_llm_model, round(latency_ms),
            )
            return response.content
        except Exception as fallback_err:
            latency_ms = (time.time() - t1) * 1000
            _record_to_router("fallback", False, latency_ms)
            _logger.error(
                "备用模型也失败 | model={} | error={}",
                settings.llm.fallback_llm_model,
                str(fallback_err),
            )
            raise

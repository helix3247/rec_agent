"""
app/core/llm.py
LLM 客户端工厂。
支持 "primary" (DeepSeek) 和 "fallback" (OpenAI) 两种模型，
调用层自动降级：主模型失败时切换到 fallback。
"""

from typing import Literal

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="LLMClient")

ModelType = Literal["primary", "fallback"]


def get_llm(model_type: ModelType = "primary", **kwargs) -> ChatOpenAI:
    """
    获取 LLM 客户端实例。

    Args:
        model_type: "primary" 使用 DeepSeek, "fallback" 使用 OpenAI。
        **kwargs: 传递给 ChatOpenAI 的额外参数（如 temperature, max_tokens 等）。
    """
    llm_cfg = settings.llm

    if model_type == "primary":
        return ChatOpenAI(
            api_key=llm_cfg.llm_api_key,
            base_url=llm_cfg.llm_base_url,
            model=llm_cfg.llm_model,
            **kwargs,
        )
    else:
        return ChatOpenAI(
            api_key=llm_cfg.fallback_llm_api_key,
            base_url=llm_cfg.fallback_llm_base_url,
            model=llm_cfg.fallback_llm_model,
            **kwargs,
        )


async def invoke_with_fallback(messages: list, **kwargs) -> str:
    """
    带降级能力的 LLM 调用：先尝试主模型，失败后自动切换到 fallback。

    Args:
        messages: LangChain 消息列表。
        **kwargs: 传递给 ChatOpenAI 的额外参数。

    Returns:
        LLM 生成的文本内容。
    """
    try:
        llm = get_llm("primary", **kwargs)
        response = await llm.ainvoke(messages)
        _logger.info(
            "主模型调用成功 | model={}", settings.llm.llm_model
        )
        return response.content
    except Exception as e:
        _logger.warning(
            "主模型调用失败，准备降级 | model={} | error={}",
            settings.llm.llm_model,
            str(e),
        )
        try:
            llm_fallback = get_llm("fallback", **kwargs)
            response = await llm_fallback.ainvoke(messages)
            _logger.info(
                "降级成功，使用备用模型 | model={}",
                settings.llm.fallback_llm_model,
            )
            return response.content
        except Exception as fallback_err:
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
    try:
        llm = get_llm("primary", **kwargs)
        response = llm.invoke(messages)
        _logger.info(
            "主模型调用成功 | model={}", settings.llm.llm_model
        )
        return response.content
    except Exception as e:
        _logger.warning(
            "主模型调用失败，准备降级 | model={} | error={}",
            settings.llm.llm_model,
            str(e),
        )
        try:
            llm_fallback = get_llm("fallback", **kwargs)
            response = llm_fallback.invoke(messages)
            _logger.info(
                "降级成功，使用备用模型 | model={}",
                settings.llm.fallback_llm_model,
            )
            return response.content
        except Exception as fallback_err:
            _logger.error(
                "备用模型也失败 | model={} | error={}",
                settings.llm.fallback_llm_model,
                str(fallback_err),
            )
            raise

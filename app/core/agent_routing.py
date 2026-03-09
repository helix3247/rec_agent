"""
app/core/agent_routing.py
Agent 智能路由公共模块 —— 消除各 Agent 节点中重复的路由 + 降级 + LLM 调用模板。

典型用法:
    response, token_usage = await invoke_llm_with_routing(
        messages=llm_messages,
        agent_name="ShoppingAgent",
        log=log,
    )
"""

from langchain_core.messages import BaseMessage

from app.core.llm import get_llm, get_model_router
from app.core.metrics import extract_token_usage


async def invoke_llm_with_routing(
    messages: list[BaseMessage],
    agent_name: str,
    log,
    **kwargs,
) -> tuple[str, dict[str, int]]:
    """
    带智能路由和自动降级的 LLM 异步调用。

    流程: classify_complexity -> select_model -> ainvoke -> 失败则降级到另一个模型

    Returns:
        (response_content, token_usage) 元组。调用者仍需处理双模型均失败时的 fallback 逻辑。

    Raises:
        Exception: 当两个模型均调用失败时，抛出最后一个异常。
    """
    router = get_model_router()
    complexity = router.classify_complexity(agent_name=agent_name)
    preferred = router.select_model(complexity)
    fallback_type = "fallback" if preferred == "primary" else "primary"
    log.info("智能路由 | complexity={} | model={}", complexity.value, preferred)

    try:
        llm = get_llm(preferred, **kwargs)
        response = await llm.ainvoke(messages)
        return response.content, extract_token_usage(response)
    except Exception as e:
        log.warning("首选模型调用失败，降级使用 {} | error={}", fallback_type, str(e))
        llm = get_llm(fallback_type, **kwargs)
        response = await llm.ainvoke(messages)
        return response.content, extract_token_usage(response)

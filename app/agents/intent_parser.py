"""
app/agents/intent_parser.py
IntentParserAgent —— 接入 LLM 进行意图识别和槽位抽取。
使用 llm.with_structured_output(IntentResult) 实现结构化输出，
并保留手动 JSON 解析作为 fallback 兼容方案。
"""

import json

from langchain_core.messages import SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, invoke_with_smart_routing_sync, get_model_router
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, extract_token_usage, merge_token_usage
from app.models.intent import IntentResult
from app.prompts.intent import INTENT_SYSTEM_PROMPT

_VALID_INTENTS = {"search", "outfit", "qa", "chat", "compare", "plan", "tool"}


def _parse_intent_json(text: str) -> IntentResult:
    """从 LLM 返回的文本中提取 JSON 并解析为 IntentResult，兼容平铺和嵌套 slots 两种格式。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    data = json.loads(cleaned)
    if data.get("intent") not in _VALID_INTENTS:
        data["intent"] = "chat"
    if "slots" in data and isinstance(data["slots"], dict):
        slots = data.pop("slots")
        for key in ("budget", "category", "scenario", "style", "must_have"):
            if key not in data or data.get(key) is None:
                data[key] = slots.get(key)
    return IntentResult(**data)


def _invoke_structured(llm, messages, log) -> tuple[IntentResult, dict[str, int]]:
    """优先使用 with_structured_output，失败则回退手动 JSON 解析。返回 (结果, token_usage)。"""
    try:
        structured_llm = llm.with_structured_output(IntentResult)
        result = structured_llm.invoke(messages)
        if result.intent not in _VALID_INTENTS:
            result.intent = "chat"
        return result, {}
    except Exception as e:
        log.info("with_structured_output 不可用，回退 JSON 解析 | error={}", str(e)[:80])
        response = llm.invoke(messages)
        log.debug("LLM 原始输出 | content={}", response.content[:200])
        return _parse_intent_json(response.content), extract_token_usage(response)


def intent_parser_node(state: AgentState) -> dict:
    """调用 LLM 进行意图识别和槽位抽取，将结果写入 State。"""
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="IntentParser", trace_id=trace_id)
    log.info("意图识别开始")

    messages = state.get("messages", [])
    prompt_messages = [SystemMessage(content=INTENT_SYSTEM_PROMPT)] + messages
    token_usage: dict[str, int] = {}
    node_success = True
    node_error = ""

    router = get_model_router()
    complexity = router.classify_complexity(agent_name="IntentParser")
    model_type = router.select_model(complexity)
    fallback_type = "fallback" if model_type == "primary" else "primary"
    log.info("智能路由 | complexity={} | model={}", complexity.value, model_type)

    try:
        llm = get_llm(model_type)
        result, token_usage = _invoke_structured(llm, prompt_messages, log)
        log.info("意图识别完成 | intent={} | slots={}", result.intent, {
            "budget": result.budget,
            "category": result.category,
            "scenario": result.scenario,
            "style": result.style,
            "must_have": result.must_have,
        })
    except Exception as e:
        log.warning("首选模型意图识别失败，降级重试 | error={}", str(e))
        try:
            fallback_llm = get_llm(fallback_type)
            result, token_usage = _invoke_structured(fallback_llm, prompt_messages, log)
            log.info("降级模型意图识别完成 | intent={} | model={}", result.intent, fallback_type)
        except Exception as fallback_err:
            log.error("意图识别彻底失败，回退到 chat 意图 | error={}", str(fallback_err))
            result = IntentResult(intent="chat")
            node_success = False
            node_error = str(fallback_err)

    existing_slots = state.get("slots", {})
    merged_slots = {
        "budget": result.budget or existing_slots.get("budget", ""),
        "category": result.category or existing_slots.get("category", ""),
        "scenario": result.scenario or existing_slots.get("scenario", ""),
        "style": result.style or existing_slots.get("style", ""),
        "must_have": result.must_have or existing_slots.get("must_have", ""),
    }

    node_result = {
        "user_intent": result.intent,
        "current_agent": "IntentParser",
        "slots": merged_slots,
    }
    metrics = record_node_metrics(
        state, "IntentParser", t0,
        token_usage=token_usage, success=node_success, error=node_error,
    )
    return {**node_result, **metrics}

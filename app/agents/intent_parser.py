"""
app/agents/intent_parser.py
IntentParserAgent —— 接入 LLM 进行意图识别和槽位抽取。
使用 llm.with_structured_output(IntentResult) 实现结构化输出，
并保留手动 JSON 解析作为 fallback 兼容方案。
"""

import json

from langchain_core.messages import SystemMessage

from app.state import AgentState
from app.core.llm import get_llm
from app.core.logger import get_logger
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


def _invoke_structured(llm, messages, log) -> IntentResult:
    """优先使用 with_structured_output，失败则回退手动 JSON 解析。"""
    try:
        structured_llm = llm.with_structured_output(IntentResult)
        result = structured_llm.invoke(messages)
        if result.intent not in _VALID_INTENTS:
            result.intent = "chat"
        return result
    except Exception as e:
        log.info("with_structured_output 不可用，回退 JSON 解析 | error={}", str(e)[:80])
        response = llm.invoke(messages)
        log.debug("LLM 原始输出 | content={}", response.content[:200])
        return _parse_intent_json(response.content)


def intent_parser_node(state: AgentState) -> dict:
    """调用 LLM 进行意图识别和槽位抽取，将结果写入 State。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="IntentParser", trace_id=trace_id)
    log.info("意图识别开始")

    messages = state.get("messages", [])
    prompt_messages = [SystemMessage(content=INTENT_SYSTEM_PROMPT)] + messages

    try:
        llm = get_llm("primary")
        result = _invoke_structured(llm, prompt_messages, log)
        log.info("意图识别完成 | intent={} | slots={}", result.intent, {
            "budget": result.budget,
            "category": result.category,
            "scenario": result.scenario,
            "style": result.style,
            "must_have": result.must_have,
        })
    except Exception as e:
        log.warning("主模型意图识别失败，使用 fallback 重试 | error={}", str(e))
        try:
            fallback_llm = get_llm("fallback")
            result = _invoke_structured(fallback_llm, prompt_messages, log)
            log.info("Fallback 意图识别完成 | intent={}", result.intent)
        except Exception as fallback_err:
            log.error("意图识别彻底失败，回退到 chat 意图 | error={}", str(fallback_err))
            result = IntentResult(intent="chat")

    existing_slots = state.get("slots", {})
    merged_slots = {
        "budget": result.budget or existing_slots.get("budget", ""),
        "category": result.category or existing_slots.get("category", ""),
        "scenario": result.scenario or existing_slots.get("scenario", ""),
        "style": result.style or existing_slots.get("style", ""),
        "must_have": result.must_have or existing_slots.get("must_have", ""),
    }

    return {
        "user_intent": result.intent,
        "current_agent": "IntentParser",
        "slots": merged_slots,
    }

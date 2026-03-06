"""
app/agents/intent_parser.py
IntentParserAgent —— 接入 LLM 进行意图识别和槽位抽取。
通过 Prompt 要求 JSON 输出 + 手动解析，兼容 DeepSeek / OpenAI。
"""

import json

from langchain_core.messages import SystemMessage

from app.state import AgentState
from app.core.llm import get_llm
from app.core.logger import get_logger
from app.models.intent import IntentResult
from app.prompts.intent import INTENT_SYSTEM_PROMPT

_VALID_INTENTS = {"search", "outfit", "qa", "chat", "compare", "plan"}


def _parse_intent_json(text: str) -> IntentResult:
    """从 LLM 返回的文本中提取 JSON 并解析为 IntentResult。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    data = json.loads(cleaned)
    if data.get("intent") not in _VALID_INTENTS:
        data["intent"] = "chat"
    return IntentResult(**data)


def intent_parser_node(state: AgentState) -> dict:
    """调用 LLM 进行意图识别和槽位抽取，将结果写入 State。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="IntentParser", trace_id=trace_id)
    log.info("意图识别开始")

    messages = state.get("messages", [])

    try:
        llm = get_llm("primary")
        response = llm.invoke(
            [SystemMessage(content=INTENT_SYSTEM_PROMPT)] + messages
        )
        result = _parse_intent_json(response.content)
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
            response = fallback_llm.invoke(
                [SystemMessage(content=INTENT_SYSTEM_PROMPT)] + messages
            )
            result = _parse_intent_json(response.content)
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

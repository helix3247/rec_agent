"""
app/agents/intent_parser.py
IntentParserAgent — Mock 版本。
返回固定的意图识别结果。
"""

from app.state import AgentState
from app.core.logger import get_logger


def intent_parser_node(state: AgentState) -> dict:
    """Mock: 固定返回 search_product 意图。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="IntentParser", trace_id=trace_id)
    log.info("意图识别开始 (Mock)")

    result_intent = "search"
    slots = {"budget": "", "category": "", "scenario": ""}

    log.info("意图识别完成 | intent={}", result_intent)
    return {
        "user_intent": result_intent,
        "current_agent": "IntentParser",
        "slots": slots,
    }

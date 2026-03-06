"""
app/agents/dispatcher.py
TaskDispatcherAgent —— 根据意图和槽位状态进行动态路由。
支持缺槽位时先转 DialogFlowAgent 追问。
"""

from app.state import AgentState
from app.core.logger import get_logger


_SLOT_REQUIREMENTS = {
    "search": ["category", "budget"],
    "outfit": ["scenario"],
}


def dispatcher_node(state: AgentState) -> dict:
    """Dispatcher 节点：记录路由决策日志。实际路由由条件边 dispatch_route 完成。"""
    trace_id = state.get("trace_id", "-")
    intent = state.get("user_intent", "unknown")
    slots = state.get("slots", {})
    log = get_logger(agent_name="Dispatcher", trace_id=trace_id)

    missing = _find_missing_slots(intent, slots)
    if missing:
        log.info("调度决策: 槽位缺失，转 DialogFlow 追问 | intent={} | missing={}", intent, missing)
    else:
        log.info("调度决策: 槽位完整，转专家 Agent | intent={}", intent)

    return {
        "current_agent": "Dispatcher",
        "task_status": "in_progress",
    }


def dispatch_route(state: AgentState) -> str:
    """
    条件边路由函数：根据 user_intent 和槽位完整性决定下一个节点。

    路由规则:
        - search/outfit 且关键槽位缺失 -> dialog（追问）
        - search   -> shopping
        - outfit   -> outfit
        - qa       -> rag
        - tool     -> tool_call
        - compare  -> shopping（对比模式）
        - plan     -> planner
        - chat     -> dialog
        - unknown  -> dialog（澄清）
    """
    intent = state.get("user_intent", "unknown")
    slots = state.get("slots", {})

    if intent in ("search", "outfit"):
        missing = _find_missing_slots(intent, slots)
        if missing:
            return "dialog"

    route_map = {
        "search": "shopping",
        "outfit": "outfit",
        "qa": "rag",
        "tool": "tool_call",
        "compare": "shopping",
        "plan": "planner",
        "chat": "dialog",
        "unknown": "dialog",
    }
    return route_map.get(intent, "dialog")


def _find_missing_slots(intent: str, slots: dict) -> list[str]:
    """检查给定意图所需的关键槽位是否缺失。"""
    required = _SLOT_REQUIREMENTS.get(intent, [])
    return [s for s in required if not slots.get(s)]

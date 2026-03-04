"""
app/agents/dispatcher.py
TaskDispatcherAgent — Mock 版本。
根据 user_intent 路由到下一个 Agent 节点。
"""

from app.state import AgentState
from app.core.logger import get_logger


def dispatcher_node(state: AgentState) -> dict:
    """Mock: 记录路由决策，实际路由由条件边完成。"""
    trace_id = state.get("trace_id", "-")
    intent = state.get("user_intent", "unknown")
    log = get_logger(agent_name="Dispatcher", trace_id=trace_id)
    log.info("调度开始 | intent={}", intent)

    return {
        "current_agent": "Dispatcher",
        "task_status": "in_progress",
    }


def dispatch_route(state: AgentState) -> str:
    """条件边路由函数：根据 user_intent 决定下一个节点。"""
    intent = state.get("user_intent", "unknown")

    route_map = {
        "search": "shopping",
        "outfit": "shopping",
        "qa": "shopping",
        "chat": "shopping",
        "compare": "shopping",
        "plan": "shopping",
        "unknown": "shopping",
    }
    return route_map.get(intent, "shopping")

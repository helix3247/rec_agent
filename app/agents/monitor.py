"""
app/agents/monitor.py
MonitorAgent — Mock 版本。
简单打印请求完成日志。
"""

from app.state import AgentState
from app.core.logger import get_logger


def monitor_node(state: AgentState) -> dict:
    """Mock: 打印请求完成日志。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="MonitorAgent", trace_id=trace_id)

    intent = state.get("user_intent", "-")
    agent = state.get("current_agent", "-")
    status = state.get("task_status", "-")

    log.info(
        "Request completed | intent={} | last_agent={} | status={}",
        intent, agent, status,
    )
    return {"current_agent": "MonitorAgent"}

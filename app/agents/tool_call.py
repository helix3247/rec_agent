"""
app/agents/tool_call.py
ToolCallAgent — Mock 版本。
阶段四将实现通用工具执行器（查订单、物流等）。
"""

from langchain_core.messages import AIMessage

from app.state import AgentState
from app.core.logger import get_logger


def tool_call_node(state: AgentState) -> dict:
    """Mock: 返回固定的工具调用结果。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="ToolCallAgent", trace_id=trace_id)
    log.info("工具调用开始 (Mock)")

    mock_response = "这是 Mock 的工具调用结果。您的订单正在配送中，预计明天送达。"

    log.info("工具调用完成")
    return {
        "current_agent": "ToolCallAgent",
        "response": mock_response,
        "messages": [AIMessage(content=mock_response)],
        "task_status": "completed",
    }

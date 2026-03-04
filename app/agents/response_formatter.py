"""
app/agents/response_formatter.py
ResponseFormatter — Mock 版本。
透传上游输出，拼装标准 Response Body 结构。
"""

from app.state import AgentState
from app.core.logger import get_logger


def response_formatter_node(state: AgentState) -> dict:
    """Mock: 透传上游 response，补充 suggested_questions。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="ResponseFormatter", trace_id=trace_id)
    log.info("响应格式化开始 (Mock)")

    response = state.get("response", "")
    suggested = ["这款相机适合拍夜景吗？", "有没有更便宜的选择？", "这款相机的续航如何？"]

    log.info("响应格式化完成")
    return {
        "current_agent": "ResponseFormatter",
        "response": response,
        "suggested_questions": suggested,
        "task_status": "completed",
    }

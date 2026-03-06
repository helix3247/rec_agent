"""
app/agents/planner.py
PlannerNode — Mock 版本。
阶段五将实现复杂任务拆解与规划执行。
"""

from langchain_core.messages import AIMessage

from app.state import AgentState
from app.core.logger import get_logger


def planner_node(state: AgentState) -> dict:
    """Mock: 返回固定的任务规划结果。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="PlannerNode", trace_id=trace_id)
    log.info("任务规划开始 (Mock)")

    mock_response = (
        "这是 Mock 的任务规划结果。已为您拆解为以下步骤：\n"
        "1. 查询目的地天气\n"
        "2. 推荐适合的服装\n"
        "3. 推荐必备装备"
    )

    log.info("任务规划完成")
    return {
        "current_agent": "PlannerNode",
        "response": mock_response,
        "messages": [AIMessage(content=mock_response)],
        "task_status": "completed",
    }

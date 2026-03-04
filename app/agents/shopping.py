"""
app/agents/shopping.py
ShoppingAgent — Mock 版本。
返回固定的推荐结果。
"""

from langchain_core.messages import AIMessage

from app.state import AgentState
from app.core.logger import get_logger


def shopping_node(state: AgentState) -> dict:
    """Mock: 返回固定推荐结果。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="ShoppingAgent", trace_id=trace_id)
    log.info("导购推荐开始 (Mock)")

    mock_response = "这是 Mock 的推荐结果：Sony A7M4"
    mock_candidates = [
        {
            "product_id": "mock-001",
            "title": "Sony A7M4",
            "price": 15999,
            "reason": "Mock 推荐理由",
        }
    ]

    log.info("导购推荐完成 | candidates_count={}", len(mock_candidates))
    return {
        "current_agent": "ShoppingAgent",
        "response": mock_response,
        "candidates": mock_candidates,
        "messages": [AIMessage(content=mock_response)],
    }

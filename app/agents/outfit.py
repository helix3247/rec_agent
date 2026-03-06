"""
app/agents/outfit.py
OutfitAgent — Mock 版本。
阶段四将实现跨品类穿搭组合推荐。
"""

from langchain_core.messages import AIMessage

from app.state import AgentState
from app.core.logger import get_logger


def outfit_node(state: AgentState) -> dict:
    """Mock: 返回固定的穿搭推荐结果。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="OutfitAgent", trace_id=trace_id)
    log.info("穿搭推荐开始 (Mock)")

    mock_response = "这是 Mock 的穿搭推荐：白色T恤 + 卡其色休闲裤 + 白色帆布鞋"
    mock_candidates = [
        {"product_id": "mock-outfit-001", "title": "白色基础T恤", "price": 99, "reason": "百搭基础款"},
        {"product_id": "mock-outfit-002", "title": "卡其色休闲裤", "price": 199, "reason": "通勤休闲两用"},
        {"product_id": "mock-outfit-003", "title": "白色帆布鞋", "price": 299, "reason": "经典百搭"},
    ]

    log.info("穿搭推荐完成 | candidates_count={}", len(mock_candidates))
    return {
        "current_agent": "OutfitAgent",
        "response": mock_response,
        "candidates": mock_candidates,
        "messages": [AIMessage(content=mock_response)],
        "task_status": "completed",
    }

"""
app/agents/rag.py
RAGAgent — Mock 版本。
阶段四将接入 Milvus 知识库实现真实 RAG 问答。
"""

from langchain_core.messages import AIMessage

from app.state import AgentState
from app.core.logger import get_logger


def rag_node(state: AgentState) -> dict:
    """Mock: 返回固定的知识库问答结果。"""
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="RAGAgent", trace_id=trace_id)
    log.info("RAG 问答开始 (Mock)")

    mock_response = "这是 Mock 的知识库问答结果。该商品评价普遍较好，夜拍效果出色。"

    log.info("RAG 问答完成")
    return {
        "current_agent": "RAGAgent",
        "response": mock_response,
        "messages": [AIMessage(content=mock_response)],
        "task_status": "completed",
    }

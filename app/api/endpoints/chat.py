"""
app/api/endpoints/chat.py
POST /chat 接口 —— 接收用户查询，异步调用 Graph，返回结构化响应。
"""

import uuid

from fastapi import APIRouter

from langchain_core.messages import HumanMessage

from app.models.schemas import ChatRequest, ChatResponse, CandidateItem
from app.graph import app_graph
from app.core.logger import get_logger

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户对话请求。"""
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    log = get_logger(agent_name="API", trace_id=trace_id)

    log.info("收到请求 | query={} | thread_id={}", request.query, thread_id)

    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "trace_id": trace_id,
        "thread_id": thread_id,
        "user_id": request.user_id or "",
        "selected_product_id": request.selected_product_id or "",
        "task_status": "pending",
    }

    result = await app_graph.ainvoke(initial_state)

    candidates = [
        CandidateItem(**c) for c in result.get("candidates", [])
    ]

    response = ChatResponse(
        response=result.get("response", ""),
        trace_id=trace_id,
        thread_id=thread_id,
        suggested_questions=result.get("suggested_questions", []),
        candidates=candidates,
    )

    log.info("请求处理完成 | status={}", result.get("task_status", "-"))
    return response

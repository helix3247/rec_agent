"""
app/api/endpoints/chat.py
POST /chat 接口 —— 接收用户查询，异步调用 Graph，返回结构化响应。
支持多轮对话：从 Redis 加载历史消息注入 State。
"""

import uuid

from fastapi import APIRouter

from langchain_core.messages import HumanMessage

from app.models.schemas import ChatRequest, ChatResponse, CandidateItem
from app.graph import app_graph
from app.agents.dialog import load_history, load_slots
from app.core.logger import get_logger

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户对话请求。"""
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    log = get_logger(agent_name="API", trace_id=trace_id)

    log.info("收到请求 | query={} | thread_id={}", request.query, thread_id)

    # 从 Redis 加载已有的对话历史和槽位
    history = load_history(thread_id)
    stored_slots = load_slots(thread_id)

    if history:
        log.info("加载历史对话 | history_count={}", len(history))

    messages = history + [HumanMessage(content=request.query)]

    initial_state = {
        "messages": messages,
        "trace_id": trace_id,
        "thread_id": thread_id,
        "user_id": request.user_id or "",
        "selected_product_id": request.selected_product_id or "",
        "task_status": "pending",
        "slots": stored_slots,
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

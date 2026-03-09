"""
app/api/endpoints/chat.py
POST /chat 接口 —— 接收用户查询，异步调用 Graph，返回结构化响应。
支持多轮对话：从 Redis 加载历史消息注入 State。
支持长期记忆：新会话时从 Milvus 加载用户历史偏好。
集成幂等保护：同一请求避免重复处理。
"""

import hashlib
import time
import uuid

from fastapi import APIRouter

from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import ChatRequest, ChatResponse, CandidateItem
from app.graph import app_graph
from app.agents.dialog import (
    load_history,
    load_slots,
    load_long_term_context,
    end_session_and_migrate,
)
from app.core.logger import get_logger
from app.core.reliability import idempotency_guard
from app.core.langfuse_integration import (
    create_trace,
    get_langfuse_callback,
    flush as langfuse_flush,
)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户对话请求。"""
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    is_new_session = not request.thread_id
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or ""
    log = get_logger(agent_name="API", trace_id=trace_id)

    log.info("收到请求 | query={} | thread_id={} | new_session={}", request.query, thread_id, is_new_session)

    # 创建 Langfuse Trace
    create_trace(
        trace_id=trace_id,
        name="rec-agent-chat",
        user_id=user_id,
        session_id=thread_id,
        metadata={"query": request.query, "is_new_session": is_new_session},
        tags=["chat"],
    )
    langfuse_callback = get_langfuse_callback(
        trace_id=trace_id,
        user_id=user_id,
        session_id=thread_id,
    )

    # 从 Redis 加载已有的对话历史和槽位
    history = load_history(thread_id)
    stored_slots = load_slots(thread_id)

    if history:
        log.info("加载历史对话 | history_count={}", len(history))

    messages = list(history)

    # 新会话且有 user_id 时，从 Milvus 加载长期记忆注入上下文
    if is_new_session and user_id:
        memory_context = load_long_term_context(user_id, query=request.query)
        if memory_context:
            log.info("注入长期记忆上下文 | user_id={} | context_len={}", user_id, len(memory_context))
            messages.insert(0, SystemMessage(content=memory_context))

    messages.append(HumanMessage(content=request.query))

    initial_state = {
        "messages": messages,
        "trace_id": trace_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "selected_product_id": request.selected_product_id or "",
        "task_status": "pending",
        "slots": stored_slots,
        "_request_start_time": time.time(),
        "_node_metrics": [],
        "_agent_route_path": [],
    }

    # 幂等保护：相同会话内相同查询不重复处理
    # 使用 thread_id 作为幂等键的 scope（而非每次变化的 trace_id），确保缓存能命中
    params_hash = hashlib.md5(f"{thread_id}:{request.query}".encode()).hexdigest()
    is_dup, cached = idempotency_guard.check_and_set(thread_id, "graph_invoke", params_hash)
    if is_dup and cached:
        log.info("幂等命中，返回缓存结果 | trace_id={}", trace_id)
        return cached

    invoke_config = {}
    if langfuse_callback:
        invoke_config["callbacks"] = [langfuse_callback]

    result = await app_graph.ainvoke(initial_state, config=invoke_config)

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

    # 缓存结果用于幂等保护（scope 与查询时一致，使用 thread_id）
    idempotency_guard.check_and_set(thread_id, "graph_invoke", params_hash, response)

    log.info("请求处理完成 | status={}", result.get("task_status", "-"))

    langfuse_flush()

    return response


@router.post("/chat/end-session")
async def end_session(thread_id: str, user_id: str):
    """
    显式结束会话，触发对话记忆迁移到 Milvus 长期记忆。
    """
    log = get_logger(agent_name="API")
    log.info("收到结束会话请求 | thread_id={} | user_id={}", thread_id, user_id)

    success = end_session_and_migrate(thread_id, user_id)
    return {
        "success": success,
        "message": "会话记忆已迁移到长期记忆" if success else "迁移失败或无可迁移内容",
    }

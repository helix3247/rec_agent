"""
app/api/endpoints/chat.py
POST /chat 接口 —— 接收用户查询，异步调用 Graph，返回结构化响应。
POST /chat/stream 接口 —— SSE 流式输出，逐 token 推送润色结果。
支持多轮对话：从 Redis 加载历史消息注入 State。
支持长期记忆：新会话时从 Milvus 加载用户历史偏好。
集成幂等保护：同一请求避免重复处理。
"""

import hashlib
import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import ChatRequest, ChatResponse, CandidateItem
from app.graph import app_graph, build_pre_formatter_graph
from app.agents.response_formatter import (
    stream_polish_response,
    generate_suggested_questions_from_state,
)
from app.agents.monitor import monitor_node
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


def _build_initial_state(
    request: ChatRequest,
    trace_id: str,
    thread_id: str,
    user_id: str,
    is_new_session: bool,
    log,
) -> dict:
    """构建 Graph 初始 state（chat 和 stream 共用）。"""
    history = load_history(thread_id)
    stored_slots = load_slots(thread_id)

    if history:
        log.info("加载历史对话 | history_count={}", len(history))

    messages = list(history)

    if is_new_session and user_id:
        memory_context = load_long_term_context(user_id, query=request.query)
        if memory_context:
            log.info("注入长期记忆上下文 | user_id={} | context_len={}", user_id, len(memory_context))
            messages.insert(0, SystemMessage(content=memory_context))

    messages.append(HumanMessage(content=request.query))

    return {
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


def _sse_event(event: str, data: dict | str) -> str:
    """构造符合 SSE 协议的事件帧。"""
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


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

    initial_state = _build_initial_state(request, trace_id, thread_id, user_id, is_new_session, log)

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


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    """
    流式对话端点 —— SSE 格式逐 token 推送。

    SSE 事件类型:
        - token    : 润色后的文本片段 {"content": "..."}
        - candidates : 候选商品列表 {"items": [...]}
        - suggestions: 推荐问题列表 {"questions": [...]}
        - done     : 流结束信号 {"trace_id": "...", "thread_id": "..."}
        - error    : 错误信息 {"message": "..."}
    """
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    is_new_session = not request.thread_id
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or ""
    log = get_logger(agent_name="API-Stream", trace_id=trace_id)

    log.info("收到流式请求 | query={} | thread_id={} | new_session={}", request.query, thread_id, is_new_session)

    create_trace(
        trace_id=trace_id,
        name="rec-agent-chat-stream",
        user_id=user_id,
        session_id=thread_id,
        metadata={"query": request.query, "is_new_session": is_new_session, "mode": "stream"},
        tags=["chat", "stream"],
    )
    langfuse_callback = get_langfuse_callback(
        trace_id=trace_id,
        user_id=user_id,
        session_id=thread_id,
    )

    initial_state = _build_initial_state(request, trace_id, thread_id, user_id, is_new_session, log)

    async def _event_generator():
        """SSE 事件生成器：先跑前置 Graph 节点，再流式润色，最后生成推荐问题。"""
        pre_result = None
        try:
            invoke_config = {}
            if langfuse_callback:
                invoke_config["callbacks"] = [langfuse_callback]

            # Phase 1: 执行前置节点（intent_parser -> dispatcher -> 业务 agent -> reflector）
            pre_graph = build_pre_formatter_graph()
            pre_result = await pre_graph.ainvoke(initial_state, config=invoke_config)

            # 检查客户端是否已断开
            if await raw_request.is_disconnected():
                log.info("客户端已断开，终止流式输出")
                return

            # Phase 2: 推送候选商品
            candidates = pre_result.get("candidates", [])
            if candidates:
                yield _sse_event("candidates", {"items": candidates})

            # Phase 3: 流式润色回答
            async for token in stream_polish_response(pre_result):
                if await raw_request.is_disconnected():
                    log.info("客户端已断开，终止流式输出")
                    return
                yield _sse_event("token", {"content": token})

            # Phase 4: 生成推荐问题
            suggested_questions = await generate_suggested_questions_from_state(pre_result)
            if suggested_questions:
                yield _sse_event("suggestions", {"questions": suggested_questions})

            # Phase 5: 完成信号
            yield _sse_event("done", {
                "trace_id": trace_id,
                "thread_id": thread_id,
            })

        except Exception as e:  # noqa: BLE001
            log.error("流式处理异常 | error={}", str(e))
            yield _sse_event("error", {"message": f"处理异常: {str(e)}"})
        finally:
            if pre_result is not None:
                try:
                    monitor_node(pre_result)
                except Exception:  # noqa: BLE001
                    pass
            langfuse_flush()
            log.info("流式请求处理完成")

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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

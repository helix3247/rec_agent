"""
app/agents/rag.py
RAGAgent —— 商品知识问答。
接入 Milvus 知识库检索，支持按 selected_product_id 过滤。
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, get_model_router
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, extract_token_usage
from app.tools.knowledge import query_knowledge
from app.tools.db import get_product_by_id, list_favorites
from app.prompts.rag import RAG_SYSTEM_PROMPT


def _format_chunks(chunks: list[dict]) -> str:
    """将知识库检索结果格式化为 Prompt 文本。"""
    if not chunks:
        return "（未检索到相关内容）"

    lines = []
    for c in chunks:
        doc_type_label = {"review": "用户评论", "faq": "FAQ", "manual": "说明书"}.get(
            c.get("doc_type", ""), c.get("doc_type", "")
        )
        lines.append(f"[{doc_type_label}] {c.get('text', '')}")
    return "\n---\n".join(lines)


def _format_product_info(product: dict | None) -> str:
    """将商品信息格式化为简要说明。"""
    if not product:
        return "（未指定具体商品）"

    return (
        f"{product.get('name', '')} | {product.get('brand', '')} | "
        f"¥{product.get('price', 0)} | {product.get('category', '')}"
    )


def rag_node(state: AgentState) -> dict:
    """RAGAgent 节点：检索知识库 -> 上下文注入 -> LLM 生成回答。"""
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    user_id = state.get("user_id", "")
    selected_product_id = state.get("selected_product_id", "")
    messages = state.get("messages", [])
    log = get_logger(agent_name="RAGAgent", trace_id=trace_id)

    log.info("RAG 问答开始 | selected_product_id={}", selected_product_id)

    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    tool_calls_log: list[dict] = []
    token_usage: dict[str, int] = {}
    product_info = None
    target_product_id = selected_product_id

    if target_product_id:
        product_info = get_product_by_id(target_product_id)
        tool_calls_log.append({"tool_name": "get_product_by_id", "success": product_info is not None})
        log.info("使用指定商品 | product_id={}", target_product_id)
    else:
        if user_id:
            favorites = list_favorites(user_id, limit=5)
            tool_calls_log.append({"tool_name": "list_favorites", "success": True})
            if favorites:
                log.info("无指定商品，引导用户从收藏夹选择 | favorites_count={}", len(favorites))
                fav_text = "\n".join(
                    f"- {f['name']}（{f['brand']}，¥{f['price']}）[ID: {f['product_id']}]"
                    for f in favorites
                )
                reply = (
                    f"您还没有指定要了解的商品。以下是您收藏夹中的商品，"
                    f"您可以告诉我想了解哪一个：\n\n{fav_text}\n\n"
                    f"您也可以直接描述您想了解的商品。"
                )
                node_result = {
                    "current_agent": "RAGAgent",
                    "response": reply,
                    "messages": [AIMessage(content=reply)],
                    "task_status": "clarifying",
                }
                metrics = record_node_metrics(
                    state, "RAGAgent", t0, tool_calls=tool_calls_log,
                )
                return {**node_result, **metrics}

    try:
        chunks = query_knowledge(
            query=query, product_id=target_product_id or None, top_k=5,
        )
        log.info("知识库检索返回 {} 个 chunk", len(chunks))
        tool_calls_log.append({"tool_name": "query_knowledge", "success": True})
    except Exception as e:
        log.error("知识库检索失败 | error={}", str(e))
        chunks = []
        tool_calls_log.append({"tool_name": "query_knowledge", "success": False, "error": str(e)})

    system_prompt = RAG_SYSTEM_PROMPT.format(
        query=query,
        product_info=_format_product_info(product_info),
        knowledge_chunks=_format_chunks(chunks),
    )

    node_success = True
    node_error = ""
    router = get_model_router()
    complexity = router.classify_complexity(agent_name="RAGAgent")
    preferred = router.select_model(complexity)
    fallback_type = "fallback" if preferred == "primary" else "primary"
    log.info("智能路由 | complexity={} | model={}", complexity.value, preferred)

    try:
        llm = get_llm(preferred)
        llm_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(llm_messages)
        reply = response.content
        token_usage = extract_token_usage(response)
    except Exception as e:
        log.warning("首选模型调用失败，降级使用 {} | error={}", fallback_type, str(e))
        try:
            llm = get_llm(fallback_type)
            llm_messages = [SystemMessage(content=system_prompt)] + messages
            response = llm.invoke(llm_messages)
            reply = response.content
            token_usage = extract_token_usage(response)
        except Exception as fe:
            if chunks:
                reply = "根据相关评价：" + chunks[0].get("text", "暂无详细信息。")
            else:
                reply = "抱歉，暂时没有找到该商品的相关信息。您可以尝试提供更具体的商品名称。"
            node_success = False
            node_error = str(fe)

    log.info("RAG 问答完成")
    node_result = {
        "current_agent": "RAGAgent",
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "task_status": "completed",
    }
    metrics = record_node_metrics(
        state, "RAGAgent", t0,
        token_usage=token_usage, tool_calls=tool_calls_log,
        success=node_success, error=node_error,
    )
    return {**node_result, **metrics}

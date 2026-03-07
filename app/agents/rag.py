"""
app/agents/rag.py
RAGAgent —— 商品知识问答。
接入 Milvus 知识库检索，支持按 selected_product_id 过滤。
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm
from app.core.logger import get_logger
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
    trace_id = state.get("trace_id", "-")
    user_id = state.get("user_id", "")
    selected_product_id = state.get("selected_product_id", "")
    messages = state.get("messages", [])
    log = get_logger(agent_name="RAGAgent", trace_id=trace_id)

    log.info("RAG 问答开始 | selected_product_id={}", selected_product_id)

    # 获取用户查询
    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # ── 1. 确定目标商品 ──
    product_info = None
    target_product_id = selected_product_id

    if target_product_id:
        product_info = get_product_by_id(target_product_id)
        log.info("使用指定商品 | product_id={}", target_product_id)
    else:
        # 无 selected_product_id 时，尝试从收藏夹引导
        if user_id:
            favorites = list_favorites(user_id, limit=5)
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
                return {
                    "current_agent": "RAGAgent",
                    "response": reply,
                    "messages": [AIMessage(content=reply)],
                    "task_status": "clarifying",
                }

    # ── 2. 检索知识库 ──
    try:
        chunks = query_knowledge(
            query=query,
            product_id=target_product_id or None,
            top_k=5,
        )
        log.info("知识库检索返回 {} 个 chunk", len(chunks))
    except Exception as e:
        log.error("知识库检索失败 | error={}", str(e))
        chunks = []

    # ── 3. LLM 生成回答 ──
    system_prompt = RAG_SYSTEM_PROMPT.format(
        query=query,
        product_info=_format_product_info(product_info),
        knowledge_chunks=_format_chunks(chunks),
    )

    try:
        llm = get_llm("primary")
        llm_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(llm_messages)
        reply = response.content
    except Exception as e:
        log.warning("主模型调用失败，使用 fallback | error={}", str(e))
        try:
            llm = get_llm("fallback")
            llm_messages = [SystemMessage(content=system_prompt)] + messages
            response = llm.invoke(llm_messages)
            reply = response.content
        except Exception:
            if chunks:
                reply = "根据相关评价：" + chunks[0].get("text", "暂无详细信息。")
            else:
                reply = "抱歉，暂时没有找到该商品的相关信息。您可以尝试提供更具体的商品名称。"

    log.info("RAG 问答完成")
    return {
        "current_agent": "RAGAgent",
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "task_status": "completed",
    }

"""
app/agents/shopping.py
ShoppingAgent —— 导购推荐专家。
接入 ES 商品检索、用户画像、个性化排序，生成真实推荐结果。
"""

import json
import re

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm
from app.core.logger import get_logger
from app.tools.search import search_products
from app.tools.db import get_user_profile
from app.tools.personalization import rerank_by_user_profile
from app.prompts.shopping import SHOPPING_SYSTEM_PROMPT, COMPARE_SYSTEM_PROMPT


def _parse_price_range(budget_str: str) -> tuple[float | None, float | None]:
    """从预算字符串中解析价格区间。"""
    if not budget_str:
        return None, None

    budget_str = budget_str.replace("，", ",").replace("～", "-").replace("~", "-")
    budget_str = budget_str.replace("元", "").replace("块", "").replace("¥", "").replace("￥", "")

    # "5000以内" / "5000以下"
    m = re.search(r"(\d+)\s*以[内下]", budget_str)
    if m:
        return None, float(m.group(1))

    # "5000以上"
    m = re.search(r"(\d+)\s*以上", budget_str)
    if m:
        return float(m.group(1)), None

    # "3000-5000" / "3000到5000"
    m = re.search(r"(\d+)\s*[-到至]\s*(\d+)", budget_str)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 纯数字
    m = re.search(r"(\d+)", budget_str)
    if m:
        val = float(m.group(1))
        return val * 0.7, val * 1.3

    return None, None


def _format_products_for_prompt(products: list[dict]) -> str:
    """将商品列表格式化为 Prompt 可读的文本。"""
    if not products:
        return "（无检索结果）"

    lines = []
    for i, p in enumerate(products, 1):
        lines.append(
            f"{i}. [{p.get('brand', '')}] {p.get('name', '')} "
            f"| 价格: ¥{p.get('price', 0)} "
            f"| 品类: {p.get('category', '')} "
            f"| 标签: {', '.join(p.get('tags', []))}"
        )
    return "\n".join(lines)


def _format_user_profile_summary(profile: dict | None) -> str:
    """将用户画像格式化为简要摘要。"""
    if not profile:
        return "（无用户画像）"

    parts = [
        f"性别: {profile.get('gender', '未知')}",
        f"预算水平: {profile.get('budget_level', '未知')}",
    ]
    if profile.get("style_preference"):
        parts.append(f"风格偏好: {', '.join(profile['style_preference'][:5])}")
    if profile.get("liked_brands"):
        parts.append(f"偏好品牌: {', '.join(profile['liked_brands'][:5])}")
    if profile.get("liked_categories"):
        parts.append(f"关注品类: {', '.join(profile['liked_categories'][:5])}")
    if profile.get("price_range", {}).get("avg"):
        parts.append(f"平均消费: ¥{profile['price_range']['avg']}")
    return " | ".join(parts)


def _build_candidates(products: list[dict], max_count: int = 5) -> list[dict]:
    """从检索结果构建候选商品输出结构。"""
    candidates = []
    for p in products[:max_count]:
        candidates.append({
            "product_id": p.get("product_id", ""),
            "title": p.get("name", ""),
            "price": p.get("price", 0),
            "reason": f"{p.get('brand', '')} | {p.get('category', '')} | {', '.join(p.get('tags', [])[:3])}",
        })
    return candidates


def shopping_node(state: AgentState) -> dict:
    """ShoppingAgent 节点：调用检索工具 -> 个性化排序 -> LLM 生成推荐。"""
    trace_id = state.get("trace_id", "-")
    user_id = state.get("user_id", "")
    intent = state.get("user_intent", "search")
    slots = state.get("slots", {})
    messages = state.get("messages", [])
    reflection_feedback = state.get("reflection_feedback", "")
    retry_count = state.get("reflection_count", 0)
    log = get_logger(agent_name="ShoppingAgent", trace_id=trace_id)

    log.info("导购推荐开始 | intent={} | slots={} | retry={}", intent, slots, retry_count)

    if reflection_feedback:
        log.info("收到反思修正建议 | feedback={}", reflection_feedback[:100])

    # 获取用户最后一条查询
    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # 若 reflector 指示改写查询，则使用改写后的查询
    if reflection_feedback and "改写查询为:" in reflection_feedback:
        rewritten = reflection_feedback.split("改写查询为:")[-1].strip()
        if rewritten:
            log.info("使用反思改写查询 | original={} | rewritten={}", query, rewritten)
            query = rewritten

    # ── 1. 解析检索参数 ──
    category = slots.get("category", "")
    budget = slots.get("budget", "")
    min_price, max_price = _parse_price_range(budget)

    # 重试时若仍无结果，逐步放宽过滤
    relax_category = retry_count >= 2

    log.info(
        "检索参数 | query={} | category={} | price=[{}, {}] | relax={}",
        query, category, min_price, max_price, relax_category,
    )

    # ── 2. 调用 ES 检索 ──
    try:
        products = search_products(
            query=query,
            category=(category or None) if not relax_category else None,
            min_price=min_price,
            max_price=max_price,
            top_k=10,
        )
        # 如果结果仍然不足且有价格限制，去掉价格限制再试一次
        if len(products) < 3 and (min_price or max_price):
            log.info("结果不足，放宽价格限制重试")
            products = search_products(
                query=query,
                category=(category or None) if not relax_category else None,
                top_k=10,
            )
        log.info("ES 检索返回 {} 个商品", len(products))
    except Exception as e:
        log.error("商品检索失败 | error={}", str(e))
        products = []

    # ── 3. 获取用户画像并个性化排序 ──
    user_profile = None
    if user_id:
        try:
            user_profile = get_user_profile(user_id)
        except Exception as e:
            log.warning("用户画像获取失败 | error={}", str(e))

    products = rerank_by_user_profile(products, user_profile)

    # ── 4. LLM 生成推荐回答 ──
    products_text = _format_products_for_prompt(products)
    profile_summary = _format_user_profile_summary(user_profile)

    if intent == "compare":
        system_prompt = COMPARE_SYSTEM_PROMPT.format(
            query=query,
            products=products_text,
        )
    else:
        system_prompt = SHOPPING_SYSTEM_PROMPT.format(
            query=query,
            slots=json.dumps(slots, ensure_ascii=False),
            products=products_text,
            user_profile_summary=profile_summary,
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
            reply = _build_fallback_response(products)

    candidates = _build_candidates(products)

    log.info("导购推荐完成 | candidates_count={}", len(candidates))
    return {
        "current_agent": "ShoppingAgent",
        "response": reply,
        "candidates": candidates,
        "messages": [AIMessage(content=reply)],
        "task_status": "completed",
    }


def _build_fallback_response(products: list[dict]) -> str:
    """LLM 不可用时的降级回答。"""
    if not products:
        return "抱歉，暂时没有找到符合条件的商品。您可以尝试调整预算或品类范围。"

    lines = ["为您推荐以下商品：\n"]
    for i, p in enumerate(products[:5], 1):
        lines.append(f"{i}. {p.get('name', '')} - ¥{p.get('price', 0)} ({p.get('brand', '')})")
    return "\n".join(lines)

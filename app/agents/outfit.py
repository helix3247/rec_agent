"""
app/agents/outfit.py
OutfitAgent —— 穿搭/组合推荐。
将跨品类需求拆解为多次检索并整合，生成全身搭配方案。
支持并发检索：多品类同时检索，限制最大并发数。
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, get_model_router
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, extract_token_usage
from app.tools.search import search_products
from app.tools.db import get_user_profile
from app.tools.personalization import rerank_by_user_profile
from app.prompts.outfit import OUTFIT_SYSTEM_PROMPT


# 穿搭品类拆解
_OUTFIT_CATEGORIES = ["上衣", "裤子", "鞋子", "配饰"]

# 场景 -> 关联标签
_SCENARIO_TAGS = {
    "通勤": ["商务", "正式", "简约", "百搭"],
    "约会": ["时尚", "精致", "优雅"],
    "休闲": ["休闲", "舒适", "运动", "百搭"],
    "运动": ["运动", "透气", "速干"],
    "旅行": ["户外", "舒适", "防风", "轻便"],
}


def _parse_price_per_category(budget_str: str) -> float | None:
    """将总预算平均分配到各品类。"""
    if not budget_str:
        return None

    budget_str = budget_str.replace("元", "").replace("块", "").replace("¥", "").replace("￥", "")

    m = re.search(r"(\d+)", budget_str)
    if m:
        total = float(m.group(1))
        return total / len(_OUTFIT_CATEGORIES)
    return None


def _format_category_products(category_results: dict[str, list[dict]]) -> str:
    """格式化各品类检索结果。"""
    lines = []
    for cat, products in category_results.items():
        lines.append(f"\n## {cat}")
        if not products:
            lines.append("  （未检索到相关商品）")
            continue
        for i, p in enumerate(products[:5], 1):
            lines.append(
                f"  {i}. [{p.get('brand', '')}] {p.get('name', '')} "
                f"| ¥{p.get('price', 0)} "
                f"| 标签: {', '.join(p.get('tags', [])[:3])}"
            )
    return "\n".join(lines)


def _build_outfit_candidates(category_results: dict[str, list[dict]]) -> list[dict]:
    """从各品类中选取最佳商品构建候选列表。"""
    candidates = []
    for cat, products in category_results.items():
        if products:
            top = products[0]
            candidates.append({
                "product_id": top.get("product_id", ""),
                "title": f"[{cat}] {top.get('name', '')}",
                "price": top.get("price", 0),
                "reason": f"{cat}推荐 | {top.get('brand', '')} | {', '.join(top.get('tags', [])[:3])}",
            })
    return candidates


def outfit_node(state: AgentState) -> dict:
    """OutfitAgent 节点：多品类检索 -> 个性化排序 -> LLM 生成穿搭方案。"""
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    user_id = state.get("user_id", "")
    slots = state.get("slots", {})
    messages = state.get("messages", [])
    reflection_feedback = state.get("reflection_feedback", "")
    retry_count = state.get("reflection_count", 0)
    log = get_logger(agent_name="OutfitAgent", trace_id=trace_id)

    log.info("穿搭推荐开始 | slots={} | retry={}", slots, retry_count)

    if reflection_feedback:
        log.info("收到反思修正建议 | feedback={}", reflection_feedback[:100])

    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    scenario = slots.get("scenario", "")
    style = slots.get("style", "")
    budget = slots.get("budget", "")
    price_per_cat = _parse_price_per_category(budget)
    tool_calls_log: list[dict] = []
    token_usage: dict[str, int] = {}

    user_profile = None
    if user_id:
        try:
            user_profile = get_user_profile(user_id)
            tool_calls_log.append({"tool_name": "get_user_profile", "success": True})
        except Exception as e:
            log.warning("用户画像获取失败 | error={}", str(e))
            tool_calls_log.append({"tool_name": "get_user_profile", "success": False, "error": str(e)})

    scenario_tags = _SCENARIO_TAGS.get(scenario, [])
    relax_price = retry_count >= 1
    effective_price = None if relax_price else price_per_cat

    _MAX_CONCURRENT_SEARCHES = 4

    def _search_single_category(cat: str) -> tuple[str, list[dict]]:
        search_query = f"{scenario} {style} {cat}".strip() or cat
        try:
            products = search_products(
                query=search_query, category=cat,
                max_price=effective_price, tags=scenario_tags or None, top_k=5,
            )
            if len(products) < 3:
                products = search_products(
                    query=search_query, category=cat, max_price=effective_price, top_k=5,
                )
            if len(products) < 2:
                products = search_products(query=cat, top_k=5)
            products = rerank_by_user_profile(products, user_profile)
            return cat, products
        except Exception as e:
            log.error("品类 {} 检索失败 | error={}", cat, str(e))
            return cat, []

    category_results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_SEARCHES) as executor:
        futures = {
            executor.submit(_search_single_category, cat): cat
            for cat in _OUTFIT_CATEGORIES
        }
        for future in as_completed(futures):
            cat_name = futures[future]
            try:
                cat, products = future.result(timeout=30)
                category_results[cat] = products
                log.info("品类 {} 检索返回 {} 个商品", cat, len(products))
                tool_calls_log.append({"tool_name": f"search_{cat}", "success": True})
            except Exception as e:
                log.error("品类 {} 并发检索异常 | error={}", cat_name, str(e))
                category_results[cat_name] = []
                tool_calls_log.append({"tool_name": f"search_{cat_name}", "success": False, "error": str(e)})

    category_products_text = _format_category_products(category_results)
    system_prompt = OUTFIT_SYSTEM_PROMPT.format(
        query=query, scenario=scenario or "未指定",
        style=style or "未指定", budget=budget or "不限",
        category_products=category_products_text,
    )

    node_success = True
    node_error = ""
    router = get_model_router()
    complexity = router.classify_complexity(agent_name="OutfitAgent")
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
            reply = _build_fallback_response(category_results)
            node_success = False
            node_error = str(fe)

    candidates = _build_outfit_candidates(category_results)

    log.info("穿搭推荐完成 | candidates_count={}", len(candidates))
    node_result = {
        "current_agent": "OutfitAgent",
        "response": reply,
        "candidates": candidates,
        "messages": [AIMessage(content=reply)],
        "task_status": "completed",
    }
    metrics = record_node_metrics(
        state, "OutfitAgent", t0,
        token_usage=token_usage, tool_calls=tool_calls_log,
        success=node_success, error=node_error,
    )
    return {**node_result, **metrics}


def _build_fallback_response(category_results: dict[str, list[dict]]) -> str:
    """LLM 不可用时的降级回答。"""
    lines = ["为您推荐以下穿搭方案：\n"]
    for cat, products in category_results.items():
        if products:
            top = products[0]
            lines.append(f"- {cat}：{top.get('name', '')}（{top.get('brand', '')}，¥{top.get('price', 0)}）")
        else:
            lines.append(f"- {cat}：暂无推荐")
    return "\n".join(lines)

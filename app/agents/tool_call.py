"""
app/agents/tool_call.py
ToolCallAgent —— 通用工具执行器。
处理非对话类的纯工具调用任务（查订单状态、物流查询等）。

工具路由机制:
    根据用户查询文本匹配工具类型，当前支持 order（订单查询），
    预留 logistics（物流）、return（退货）、complaint（投诉）扩展点。
"""

from enum import Enum
from typing import NamedTuple

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, get_model_router
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, extract_token_usage
from app.tools.db import query_order_status
from app.prompts.tool_call import TOOL_CALL_SYSTEM_PROMPT


class ToolType(str, Enum):
    """支持的工具类型。"""
    ORDER = "order"
    LOGISTICS = "logistics"
    RETURN = "return"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class ToolRouteResult(NamedTuple):
    tool_type: ToolType
    confidence: float


_TOOL_KEYWORDS: dict[ToolType, list[str]] = {
    ToolType.ORDER: ["订单", "下单", "购买", "付款", "支付", "买了", "订购", "交易"],
    ToolType.LOGISTICS: ["物流", "快递", "配送", "发货", "运输", "到了吗", "到哪了", "签收"],
    ToolType.RETURN: ["退货", "退款", "退回", "换货", "售后", "退换"],
    ToolType.COMPLAINT: ["投诉", "举报", "不满", "差评", "问题反馈"],
}


def _route_to_tool(query: str) -> ToolRouteResult:
    """
    根据用户查询文本判断应该调用哪个工具。

    通过关键词匹配计算各工具类型的得分，返回最匹配的工具及置信度。
    当前仅 ORDER 工具有实际实现，其余类型预留扩展。

    Args:
        query: 用户查询文本。

    Returns:
        ToolRouteResult(tool_type, confidence)。
    """
    if not query:
        return ToolRouteResult(ToolType.UNKNOWN, 0.0)

    scores: dict[ToolType, int] = {}
    for tool_type, keywords in _TOOL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query)
        if score > 0:
            scores[tool_type] = score

    if not scores:
        return ToolRouteResult(ToolType.ORDER, 0.3)

    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    total_keywords = len(_TOOL_KEYWORDS[best_type])
    confidence = min(scores[best_type] / max(total_keywords, 1), 1.0)
    return ToolRouteResult(best_type, round(confidence, 2))


_STATUS_LABELS = {
    "pending": "待付款",
    "paid": "已付款，等待发货",
    "shipped": "已发货，配送中",
    "done": "已完成",
    "cancelled": "已取消",
}


def _format_order_results(orders: list[dict]) -> str:
    """将订单查询结果格式化为文本。"""
    if not orders:
        return "（未查询到订单信息）"

    lines = []
    for o in orders:
        status_label = _STATUS_LABELS.get(o.get("status", ""), o.get("status", ""))
        lines.append(
            f"订单 {o.get('order_id', '')}：{o.get('product_name', '')} × {o.get('quantity', 1)} "
            f"| ¥{o.get('total_price', 0)} | 状态：{status_label} | 时间：{o.get('created_at', '')}"
        )
    return "\n".join(lines)


def _execute_tool(tool_type: ToolType, user_id: str, query: str, log) -> tuple[str, list[dict]]:
    """
    根据工具类型执行对应工具，返回结果文本和调用日志。

    当前仅 ORDER 有实际实现，其余类型返回预留提示。
    """
    tool_calls_log: list[dict] = []

    if tool_type == ToolType.ORDER:
        if not user_id:
            return "（未提供用户 ID，无法查询订单信息）", tool_calls_log
        try:
            orders = query_order_status(user_id)
            result = _format_order_results(orders)
            log.info("订单查询完成 | orders_count={}", len(orders))
            tool_calls_log.append({"tool_name": "query_order_status", "success": True})
            return result, tool_calls_log
        except Exception as e:
            log.error("订单查询失败 | error={}", str(e))
            tool_calls_log.append({"tool_name": "query_order_status", "success": False, "error": str(e)})
            return "（订单查询失败，请稍后重试）", tool_calls_log

    if tool_type == ToolType.LOGISTICS:
        log.info("物流查询工具尚未实现，降级到订单查询")
        return _execute_tool(ToolType.ORDER, user_id, query, log)

    if tool_type == ToolType.RETURN:
        log.info("退货工具尚未实现")
        return "（退货/售后功能正在开发中，请联系客服处理）", tool_calls_log

    if tool_type == ToolType.COMPLAINT:
        log.info("投诉工具尚未实现")
        return "（投诉功能正在开发中，请联系客服处理）", tool_calls_log

    if user_id:
        return _execute_tool(ToolType.ORDER, user_id, query, log)
    return "（暂不支持该类型的查询）", tool_calls_log


def tool_call_node(state: AgentState) -> dict:
    """ToolCallAgent 节点：根据用户请求路由到对应工具并返回结果。"""
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    user_id = state.get("user_id", "")
    messages = state.get("messages", [])
    log = get_logger(agent_name="ToolCallAgent", trace_id=trace_id)

    log.info("工具调用开始 | user_id={}", user_id)

    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    route = _route_to_tool(query)
    log.info("工具路由 | tool_type={} | confidence={}", route.tool_type.value, route.confidence)

    token_usage: dict[str, int] = {}
    tool_result, tool_calls_log = _execute_tool(route.tool_type, user_id, query, log)

    system_prompt = TOOL_CALL_SYSTEM_PROMPT.format(query=query, tool_result=tool_result)

    node_success = True
    node_error = ""
    router = get_model_router()
    complexity = router.classify_complexity(agent_name="ToolCallAgent")
    preferred = router.select_model(complexity)
    fallback_type = "fallback" if preferred == "primary" else "primary"
    log.info("智能路由 | complexity={} | model={}", complexity.value, preferred)

    try:
        llm = get_llm(preferred)
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
        reply = response.content
        token_usage = extract_token_usage(response)
    except Exception as e:
        log.warning("首选模型调用失败，降级使用 {} | error={}", fallback_type, str(e))
        try:
            llm = get_llm(fallback_type)
            response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
            reply = response.content
            token_usage = extract_token_usage(response)
        except Exception as fe:
            if tool_result and "未查询" not in tool_result:
                reply = f"以下是您的订单信息：\n{tool_result}"
            else:
                reply = "抱歉，暂时无法查询到您的订单信息。请确认您是否已登录，或提供订单号进行查询。"
            node_success = False
            node_error = str(fe)

    log.info("工具调用完成")
    node_result = {
        "current_agent": "ToolCallAgent",
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "task_status": "completed",
    }
    metrics = record_node_metrics(
        state, "ToolCallAgent", t0,
        token_usage=token_usage, tool_calls=tool_calls_log,
        success=node_success, error=node_error,
    )
    return {**node_result, **metrics}

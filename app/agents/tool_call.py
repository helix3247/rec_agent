"""
app/agents/tool_call.py
ToolCallAgent —— 通用工具执行器。
处理非对话类的纯工具调用任务（查订单状态、物流查询等）。
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, get_model_router
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, extract_token_usage
from app.tools.db import query_order_status
from app.prompts.tool_call import TOOL_CALL_SYSTEM_PROMPT


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


def tool_call_node(state: AgentState) -> dict:
    """ToolCallAgent 节点：根据用户请求调用工具并返回结果。"""
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

    tool_calls_log: list[dict] = []
    token_usage: dict[str, int] = {}
    tool_result = ""
    if user_id:
        try:
            orders = query_order_status(user_id)
            tool_result = _format_order_results(orders)
            log.info("订单查询完成 | orders_count={}", len(orders))
            tool_calls_log.append({"tool_name": "query_order_status", "success": True})
        except Exception as e:
            log.error("订单查询失败 | error={}", str(e))
            tool_result = "（订单查询失败，请稍后重试）"
            tool_calls_log.append({"tool_name": "query_order_status", "success": False, "error": str(e)})
    else:
        tool_result = "（未提供用户 ID，无法查询订单信息）"

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

"""
app/agents/monitor.py
MonitorAgent —— 全链路可观测性节点。

位于 Graph 末尾（ResponseFormatter -> Monitor -> END），负责:
    - 汇总本次请求的全链路指标（总耗时、各节点耗时、Token 消耗、工具调用成功率）
    - 记录 Agent 路由路径
    - 输出结构化 JSON 日志，支持按 trace_id 串联完整请求链路
"""

import json
import time
from typing import Any

from app.state import AgentState
from app.core.llm import get_model_router
from app.core.logger import get_logger
from app.core.langfuse_integration import report_trace_metrics


def _calc_tool_call_stats(node_metrics: list[dict]) -> dict[str, Any]:
    """统计所有节点中的工具调用成功率。"""
    total = 0
    success = 0
    failures: list[dict[str, str]] = []

    for nm in node_metrics:
        for tc in nm.get("tool_calls", []):
            total += 1
            if tc.get("success", False):
                success += 1
            else:
                failures.append({
                    "node": nm.get("node_name", "-"),
                    "tool": tc.get("tool_name", "-"),
                    "error": tc.get("error", ""),
                })

    return {
        "total": total,
        "success": success,
        "failed": total - success,
        "success_rate": round(success / total, 4) if total > 0 else 1.0,
        "failures": failures,
    }


def _calc_token_summary(node_metrics: list[dict]) -> dict[str, int]:
    """汇总所有节点的 Token 使用量。"""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for nm in node_metrics:
        usage = nm.get("token_usage", {})
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _build_node_latency_breakdown(node_metrics: list[dict]) -> list[dict[str, Any]]:
    """构建各节点的耗时细分列表。"""
    breakdown = []
    for nm in node_metrics:
        breakdown.append({
            "node": nm.get("node_name", "-"),
            "latency_ms": round(nm.get("latency_ms", 0), 1),
            "success": nm.get("success", True),
            "error": nm.get("error", ""),
        })
    return breakdown


def monitor_node(state: AgentState) -> dict:
    """
    MonitorAgent 节点：汇总全链路指标并输出结构化日志。

    日志为 JSON 格式，包含:
        - trace_id / thread_id / user_id
        - user_intent / agent_route_path
        - total_latency_ms / node_latency_breakdown
        - token_usage
        - tool_call_stats
        - task_status
    """
    trace_id = state.get("trace_id", "-")
    log = get_logger(agent_name="MonitorAgent", trace_id=trace_id)

    request_start = state.get("_request_start_time", 0)
    total_latency_ms = round((time.time() - request_start) * 1000, 1) if request_start else 0

    node_metrics: list[dict] = state.get("_node_metrics", [])
    route_path: list[str] = state.get("_agent_route_path", [])

    token_summary = _calc_token_summary(node_metrics)
    tool_stats = _calc_tool_call_stats(node_metrics)
    node_breakdown = _build_node_latency_breakdown(node_metrics)

    # 汇总模型路由健康状态
    router = get_model_router()
    model_routing = router.get_health_report()

    trace_report = {
        "trace_id": trace_id,
        "thread_id": state.get("thread_id", "-"),
        "user_id": state.get("user_id", "-"),
        "user_intent": state.get("user_intent", "-"),
        "task_status": state.get("task_status", "-"),
        "agent_route_path": route_path,
        "total_latency_ms": total_latency_ms,
        "token_usage": token_summary,
        "tool_call_stats": {
            "total": tool_stats["total"],
            "success": tool_stats["success"],
            "failed": tool_stats["failed"],
            "success_rate": tool_stats["success_rate"],
        },
        "model_routing": model_routing,
        "node_latency_breakdown": node_breakdown,
    }

    log.info("TRACE_REPORT | {}", json.dumps(trace_report, ensure_ascii=False))

    if tool_stats["failures"]:
        for f in tool_stats["failures"]:
            log.warning(
                "工具调用失败 | node={} | tool={} | error={}",
                f["node"], f["tool"], f["error"],
            )

    # 检查模型健康状态，对频繁降级到 fallback 的情况打 warning
    for model_name, metrics in model_routing.items():
        if not metrics.get("healthy", True):
            log.warning(
                "模型不健康 | model={} | error_rate={} | consecutive_failures={}",
                model_name, metrics.get("error_rate", 0), metrics.get("consecutive_failures", 0),
            )
        if metrics.get("error_rate", 0) > 0.3:
            log.warning(
                "模型降级频繁 | model={} | error_rate={:.2%} | total_calls={}",
                model_name, metrics["error_rate"], metrics.get("total_calls", 0),
            )

    slow_nodes = [n for n in node_breakdown if n["latency_ms"] > 5000]
    for sn in slow_nodes:
        log.warning(
            "慢节点警告 | node={} | latency={}ms",
            sn["node"], sn["latency_ms"],
        )

    if not node_metrics:
        log.info(
            "Request completed | intent={} | last_agent={} | status={} | latency={}ms",
            state.get("user_intent", "-"),
            state.get("current_agent", "-"),
            state.get("task_status", "-"),
            total_latency_ms,
        )

    # 上报汇总指标到 Langfuse
    report_trace_metrics(
        trace_id=trace_id,
        user_intent=state.get("user_intent", ""),
        route_path=route_path,
        total_latency_ms=total_latency_ms,
        token_usage=token_summary,
        tool_call_stats={
            "total": tool_stats["total"],
            "success": tool_stats["success"],
            "failed": tool_stats["failed"],
            "success_rate": tool_stats["success_rate"],
        },
        task_status=state.get("task_status", ""),
    )

    return {"current_agent": "MonitorAgent"}

"""
app/core/metrics.py
节点指标采集工具 —— 为各 Agent 节点提供统一的指标记录能力。

使用方式:
    在节点函数中调用 record_node_metrics(state, ...) 将当前节点的
    耗时、Token 消耗、工具调用等信息追加到 state 中，由 MonitorAgent 汇总。
"""

import time
from typing import Any

from app.state import AgentState, NodeMetrics


def start_node_timer() -> float:
    """返回当前时间戳（秒），用于节点开始计时。"""
    return time.time()


def extract_token_usage(response) -> dict[str, int]:
    """从 LangChain LLM 响应对象中提取 token_usage 信息。"""
    usage: dict[str, int] = {}
    if response is None:
        return usage
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if token_usage:
        usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] = token_usage.get("completion_tokens", 0)
        usage["total_tokens"] = token_usage.get("total_tokens", 0)
    return usage


def merge_token_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """合并两个 token_usage 字典。"""
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
    }


def record_node_metrics(
    state: AgentState,
    node_name: str,
    start_time: float,
    *,
    token_usage: dict[str, int] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    success: bool = True,
    error: str = "",
) -> dict:
    """
    构建当前节点的指标并返回需要合并到 state 的增量字典。

    调用方将返回值合并到节点返回的 dict 中即可:
        metrics_update = record_node_metrics(state, "ShoppingAgent", t0, ...)
        return {**result, **metrics_update}
    """
    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 1)

    node_metric: NodeMetrics = {
        "node_name": node_name,
        "start_time": start_time,
        "end_time": end_time,
        "latency_ms": latency_ms,
        "token_usage": token_usage or {},
        "tool_calls": tool_calls or [],
        "success": success,
        "error": error,
    }

    existing_metrics = list(state.get("_node_metrics", []))
    existing_metrics.append(node_metric)

    existing_path = list(state.get("_agent_route_path", []))
    existing_path.append(node_name)

    return {
        "_node_metrics": existing_metrics,
        "_agent_route_path": existing_path,
    }

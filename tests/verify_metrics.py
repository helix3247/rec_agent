"""
tests/verify_metrics.py
验证 Monitor 节点的指标汇总功能 —— 按当前 AgentState schema 编写。

可以通过 pytest 运行，也可以直接 `python tests/verify_metrics.py` 执行。
"""

import time
from unittest.mock import patch

import pytest

from app.state import AgentState
from app.agents.monitor import (
    _calc_token_summary,
    _calc_tool_call_stats,
    _build_node_latency_breakdown,
    monitor_node,
)


# ────────────────── 辅助工厂 ──────────────────


def _make_state(**overrides) -> AgentState:
    """创建一个基本的 AgentState，可通过 overrides 覆盖字段。"""
    from langchain_core.messages import HumanMessage, AIMessage

    base: AgentState = {
        "messages": [
            HumanMessage(content="帮我找一双运动鞋"),
            AIMessage(content="为您推荐以下运动鞋"),
        ],
        "user_intent": "search",
        "current_agent": "MonitorAgent",
        "task_status": "completed",
        "trace_id": "test-trace-metrics",
        "thread_id": "test-thread-001",
        "user_id": "user-001",
        "selected_product_id": "",
        "slots": {"category": "运动鞋"},
        "response": "推荐运动鞋列表",
        "candidates": [],
        "suggested_questions": [],
        "reflection_count": 0,
        "reflection_feedback": "",
        "plan_steps": [],
        "plan_current_step": 0,
        "plan_results": [],
        "_request_start_time": time.time() - 3.0,
        "_node_metrics": [],
        "_agent_route_path": ["intent_parser", "dispatcher", "shopping", "reflector", "response_formatter"],
    }
    base.update(overrides)
    return base


SAMPLE_NODE_METRICS = [
    {
        "node_name": "intent_parser",
        "start_time": 1000.0,
        "end_time": 1000.5,
        "latency_ms": 500,
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "tool_calls": [],
        "success": True,
        "error": "",
    },
    {
        "node_name": "shopping",
        "start_time": 1000.5,
        "end_time": 1001.5,
        "latency_ms": 1000,
        "token_usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        "tool_calls": [
            {"tool_name": "search_products", "success": True, "error": ""},
            {"tool_name": "get_user_profile", "success": True, "error": ""},
        ],
        "success": True,
        "error": "",
    },
    {
        "node_name": "response_formatter",
        "start_time": 1001.5,
        "end_time": 1002.0,
        "latency_ms": 500,
        "token_usage": {"prompt_tokens": 150, "completion_tokens": 80, "total_tokens": 230},
        "tool_calls": [],
        "success": True,
        "error": "",
    },
]


# ────────────────── Token 汇总测试 ──────────────────


@pytest.mark.unit
class TestTokenSummary:

    def test_calc_token_summary_normal(self):
        result = _calc_token_summary(SAMPLE_NODE_METRICS)
        assert result["prompt_tokens"] == 450
        assert result["completion_tokens"] == 230
        assert result["total_tokens"] == 680

    def test_calc_token_summary_empty(self):
        result = _calc_token_summary([])
        assert result["prompt_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_calc_token_summary_missing_usage(self):
        metrics = [{"node_name": "test", "success": True}]
        result = _calc_token_summary(metrics)
        assert result["total_tokens"] == 0


# ────────────────── 工具调用统计测试 ──────────────────


@pytest.mark.unit
class TestToolCallStats:

    def test_all_success(self):
        result = _calc_tool_call_stats(SAMPLE_NODE_METRICS)
        assert result["total"] == 2
        assert result["success"] == 2
        assert result["failed"] == 0
        assert result["success_rate"] == 1.0

    def test_with_failures(self):
        metrics = [
            {
                "node_name": "shopping",
                "tool_calls": [
                    {"tool_name": "search_products", "success": True, "error": ""},
                    {"tool_name": "get_user_profile", "success": False, "error": "connection timeout"},
                ],
            },
        ]
        result = _calc_tool_call_stats(metrics)
        assert result["total"] == 2
        assert result["success"] == 1
        assert result["failed"] == 1
        assert result["success_rate"] == 0.5
        assert len(result["failures"]) == 1
        assert result["failures"][0]["tool"] == "get_user_profile"

    def test_no_tool_calls(self):
        metrics = [{"node_name": "test", "tool_calls": []}]
        result = _calc_tool_call_stats(metrics)
        assert result["total"] == 0
        assert result["success_rate"] == 1.0


# ────────────────── 节点耗时细分测试 ──────────────────


@pytest.mark.unit
class TestNodeLatencyBreakdown:

    def test_breakdown_structure(self):
        result = _build_node_latency_breakdown(SAMPLE_NODE_METRICS)
        assert len(result) == 3
        assert result[0]["node"] == "intent_parser"
        assert result[0]["latency_ms"] == 500
        assert result[1]["node"] == "shopping"
        assert result[1]["latency_ms"] == 1000

    def test_breakdown_empty(self):
        result = _build_node_latency_breakdown([])
        assert result == []


# ────────────────── Monitor 节点完整测试 ──────────────────


@pytest.mark.unit
class TestMonitorNode:

    @patch("app.agents.monitor.report_trace_metrics")
    @patch("app.agents.monitor.get_model_router")
    def test_monitor_node_normal_flow(self, mock_router, mock_report):
        mock_router.return_value.get_health_report.return_value = {
            "primary": {"healthy": True, "error_rate": 0.0, "total_calls": 10},
            "fallback": {"healthy": True, "error_rate": 0.0, "total_calls": 2},
        }

        state = _make_state(_node_metrics=SAMPLE_NODE_METRICS)
        result = monitor_node(state)

        assert result == {"current_agent": "MonitorAgent"}
        mock_report.assert_called_once()

        call_kwargs = mock_report.call_args
        assert call_kwargs.kwargs["trace_id"] == "test-trace-metrics"
        assert call_kwargs.kwargs["user_intent"] == "search"
        assert call_kwargs.kwargs["token_usage"]["total_tokens"] == 680

    @patch("app.agents.monitor.report_trace_metrics")
    @patch("app.agents.monitor.get_model_router")
    def test_monitor_node_empty_metrics(self, mock_router, mock_report):
        mock_router.return_value.get_health_report.return_value = {}

        state = _make_state(_node_metrics=[])
        result = monitor_node(state)

        assert result == {"current_agent": "MonitorAgent"}

    @patch("app.agents.monitor.report_trace_metrics")
    @patch("app.agents.monitor.get_model_router")
    def test_monitor_detects_unhealthy_model(self, mock_router, mock_report):
        mock_router.return_value.get_health_report.return_value = {
            "primary": {
                "healthy": False,
                "error_rate": 0.5,
                "total_calls": 20,
                "consecutive_failures": 4,
            },
        }

        state = _make_state(_node_metrics=SAMPLE_NODE_METRICS)
        monitor_node(state)
        # 不应抛异常，应正常完成并记录警告日志


# ────────────────── 独立运行入口 ──────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

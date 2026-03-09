"""
tests/test_routing.py
验证不同意图输入通过 Dispatcher 的路由路径是否正确。

分两层测试:
1. 纯单元测试：直接验证 dispatch_route 函数的路由逻辑（无 LLM 依赖）。
2. 端到端集成测试：使用真实 LLM 识别意图后验证完整路由路径（需要外部服务）。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.agents.dispatcher import dispatch_route


# ─────────────── 单元测试：dispatch_route 路由逻辑 ───────────────

@pytest.mark.parametrize("intent,slots,expected_route", [
    ("search", {"category": "相机", "budget": "5000"}, "shopping"),
    ("search", {"category": "相机"}, "dialog"),
    ("search", {}, "dialog"),
    ("outfit", {"scenario": "通勤"}, "outfit"),
    ("outfit", {}, "dialog"),
    ("qa", {}, "rag"),
    ("compare", {}, "shopping"),
    ("plan", {}, "planner"),
    ("chat", {}, "dialog"),
    ("tool", {}, "tool_call"),
    ("unknown", {}, "dialog"),
])
def test_dispatch_route(intent, slots, expected_route):
    state = {"user_intent": intent, "slots": slots}
    assert dispatch_route(state) == expected_route


def test_search_with_all_slots_routes_to_shopping():
    state = {
        "user_intent": "search",
        "slots": {"category": "手机", "budget": "3000以内"},
    }
    assert dispatch_route(state) == "shopping"


def test_search_missing_budget_routes_to_dialog():
    state = {
        "user_intent": "search",
        "slots": {"category": "手机"},
    }
    assert dispatch_route(state) == "dialog"


# ─────────────── 端到端集成测试（需要 LLM 等外部服务） ───────────────


ROUTE_CASES = [
    {
        "query": "推荐一款5000元的微单相机",
        "expected_intent": "search",
        "expected_route": "shopping",
    },
    {
        "query": "男生休闲穿搭推荐，预算500",
        "expected_intent": "outfit",
        "expected_route": "outfit",
    },
    {
        "query": "想买个相机",
        "expected_intent": "search",
        "expected_route": "dialog",
    },
    {
        "query": "你好呀",
        "expected_intent": "chat",
        "expected_route": "dialog",
    },
    {
        "query": "Sony A7M4 的续航表现怎么样",
        "expected_intent": "qa",
        "expected_route": "rag",
    },
    {
        "query": "iPhone 16 和 Samsung S25 哪个好",
        "expected_intent": "compare",
        "expected_route": "shopping",
    },
    {
        "query": "去西藏旅游需要准备哪些装备",
        "expected_intent": "plan",
        "expected_route": "planner",
    },
]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("case", ROUTE_CASES, ids=[c["query"][:20] for c in ROUTE_CASES])
async def test_e2e_routing(case):
    """端到端路由验证：通过真实 LLM 意图识别后检查路由目标。"""
    from app.graph import app_graph

    result = await app_graph.ainvoke({
        "messages": [HumanMessage(content=case["query"])],
        "trace_id": "test-route",
        "thread_id": "test-thread-route",
        "user_id": "user-test",
    })

    actual_intent = result.get("user_intent", "")
    actual_route = dispatch_route({
        "user_intent": actual_intent,
        "slots": result.get("slots", {}),
    })

    assert actual_intent == case["expected_intent"], (
        f"Intent mismatch: expected={case['expected_intent']}, got={actual_intent}"
    )
    assert actual_route == case["expected_route"], (
        f"Route mismatch: expected={case['expected_route']}, got={actual_route}"
    )

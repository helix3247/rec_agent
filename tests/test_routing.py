"""
tests/test_routing.py
验证不同意图输入通过 Dispatcher 的路由路径是否正确。
使用真实 LLM 识别意图，然后检查最终走了哪条路径。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from app.graph import app_graph


ROUTE_CASES = [
    {
        "query": "推荐一款5000元的微单相机",
        "expected_intent": "search",
        "expected_route": "shopping",
    },
    {
        "query": "男生休闲穿搭推荐，预算500",
        "expected_intent": "outfit",
        "expected_route": "shopping",
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
        "expected_route": "shopping",
    },
    {
        "query": "iPhone 16 和 Samsung S25 哪个好",
        "expected_intent": "compare",
        "expected_route": "shopping",
    },
]


def determine_route(result: dict) -> str:
    """根据结果的 current_agent 链路判断实际走了 shopping 还是 dialog 路径。"""
    response = result.get("response", "")
    task_status = result.get("task_status", "")
    candidates = result.get("candidates", [])

    if "Mock" in response and candidates:
        return "shopping"
    if task_status == "clarifying":
        return "dialog"
    if not candidates:
        return "dialog"
    return "shopping"


def main():
    print("=" * 60)
    print("  Dispatcher 路由正确性测试")
    print("=" * 60)

    passed = 0

    for i, case in enumerate(ROUTE_CASES, 1):
        query = case["query"]
        expected_intent = case["expected_intent"]
        expected_route = case["expected_route"]

        result = app_graph.invoke({
            "messages": [HumanMessage(content=query)],
            "trace_id": f"test-route-{i:03d}",
            "thread_id": f"test-thread-route-{i}",
            "user_id": "user-test",
        })

        actual_intent = result.get("user_intent", "")
        actual_route = determine_route(result)

        intent_ok = actual_intent == expected_intent
        route_ok = actual_route == expected_route

        if intent_ok and route_ok:
            passed += 1
            status = "[OK]"
        else:
            status = "[FAIL]"

        print(f"  {status} Case {i} | query=\"{query}\"")
        print(f"         intent: expected={expected_intent}, actual={actual_intent} {'OK' if intent_ok else 'MISMATCH'}")
        print(f"         route:  expected={expected_route}, actual={actual_route} {'OK' if route_ok else 'MISMATCH'}")

    print(f"\n  结果: {passed}/{len(ROUTE_CASES)} 通过")
    if passed == len(ROUTE_CASES):
        print("  [OK] 全部路由测试通过!")
    else:
        print("  [WARN] 部分路由不符合预期")
    print("=" * 60)


if __name__ == "__main__":
    main()

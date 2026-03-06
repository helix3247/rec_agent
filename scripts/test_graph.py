"""
scripts/test_graph.py
验证 LangGraph 图结构：打印 ASCII 图 + 执行一次完整流转。
已适配阶段三真实 LLM 流程。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from app.graph import app_graph


def test_draw():
    """打印图结构的 ASCII 表示。"""
    print("=" * 60)
    print("  Graph ASCII 结构")
    print("=" * 60)
    print(app_graph.get_graph().draw_ascii())


def test_invoke_search():
    """执行一次 search 意图（槽位完整）的流转。"""
    print("=" * 60)
    print("  测试: search 意图（槽位完整）")
    print("=" * 60)

    result = app_graph.invoke({
        "messages": [HumanMessage(content="推荐一款5000元的微单相机")],
        "trace_id": "test-graph-search",
        "thread_id": "thread-graph-test",
        "user_id": "user-test",
    })

    print(f"\n  user_intent        : {result.get('user_intent')}")
    print(f"  current_agent      : {result.get('current_agent')}")
    print(f"  task_status        : {result.get('task_status')}")
    print(f"  slots              : {result.get('slots')}")
    print(f"  response           : {result.get('response', '')[:80]}...")
    print(f"  candidates         : {result.get('candidates')}")
    print(f"  suggested_questions: {result.get('suggested_questions')}")
    print(f"  messages count     : {len(result.get('messages', []))}")

    assert result["user_intent"] == "search", f"Expected search, got {result['user_intent']}"
    assert result["task_status"] == "completed"
    assert result.get("response"), "Response should not be empty"
    print("\n  [OK] search 流转测试通过!")


def test_invoke_clarify():
    """执行一次需要澄清的 search 意图流转。"""
    print("\n" + "=" * 60)
    print("  测试: search 意图（需要澄清）")
    print("=" * 60)

    result = app_graph.invoke({
        "messages": [HumanMessage(content="想买个相机")],
        "trace_id": "test-graph-clarify",
        "thread_id": "thread-graph-clarify",
        "user_id": "user-test",
    })

    print(f"\n  user_intent        : {result.get('user_intent')}")
    print(f"  task_status        : {result.get('task_status')}")
    print(f"  slots              : {result.get('slots')}")
    print(f"  response           : {result.get('response', '')[:120]}...")

    assert result["user_intent"] == "search"
    assert result.get("response"), "Response should not be empty"
    print("\n  [OK] 澄清流转测试通过!")


if __name__ == "__main__":
    test_draw()
    test_invoke_search()
    test_invoke_clarify()

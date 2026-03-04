"""
scripts/test_graph.py
验证 LangGraph 图结构：打印 ASCII 图 + 执行一次完整流转。
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


def test_invoke():
    """执行一次完整的 Mock 流转。"""
    print("=" * 60)
    print("  Graph Mock 执行测试")
    print("=" * 60)

    initial_state = {
        "messages": [HumanMessage(content="推荐一款相机")],
        "trace_id": "test-graph-001",
        "thread_id": "thread-test",
        "user_id": "user-test",
    }

    result = app_graph.invoke(initial_state)

    print(f"\n  user_intent      : {result.get('user_intent')}")
    print(f"  current_agent    : {result.get('current_agent')}")
    print(f"  task_status      : {result.get('task_status')}")
    print(f"  response         : {result.get('response')}")
    print(f"  candidates       : {result.get('candidates')}")
    print(f"  suggested_questions: {result.get('suggested_questions')}")
    print(f"  messages count   : {len(result.get('messages', []))}")

    assert result["user_intent"] == "search"
    assert result["task_status"] == "completed"
    assert result["response"] == "这是 Mock 的推荐结果：Sony A7M4"
    assert len(result["candidates"]) == 1
    assert len(result["suggested_questions"]) == 3

    print("\n  [OK] Graph Mock 执行测试通过!")


if __name__ == "__main__":
    test_draw()
    test_invoke()

"""
tests/test_state.py
验证 AgentState 可以被正确初始化和修改。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage
from app.state import AgentState


def test_state_init():
    """AgentState 可以用最少字段初始化。"""
    state: AgentState = {
        "messages": [],
        "user_intent": "",
        "current_agent": "",
        "task_status": "pending",
        "trace_id": "test-trace-001",
    }
    assert state["trace_id"] == "test-trace-001"
    assert state["task_status"] == "pending"
    assert state["messages"] == []
    print("  [OK] test_state_init")


def test_state_with_messages():
    """AgentState 可以正确存放 LangChain 消息。"""
    state: AgentState = {
        "messages": [
            HumanMessage(content="想买个相机"),
            AIMessage(content="请问您的预算是多少？"),
        ],
        "user_intent": "search",
        "current_agent": "IntentParser",
        "task_status": "in_progress",
        "trace_id": "test-trace-002",
    }
    assert len(state["messages"]) == 2
    assert state["messages"][0].content == "想买个相机"
    assert state["user_intent"] == "search"
    print("  [OK] test_state_with_messages")


def test_state_slots():
    """AgentState 的 slots 字段可以正确存取。"""
    state: AgentState = {
        "messages": [],
        "user_intent": "search",
        "current_agent": "IntentParser",
        "task_status": "pending",
        "trace_id": "test-trace-003",
        "slots": {
            "budget": "5000",
            "category": "相机",
            "scenario": "旅行",
        },
    }
    assert state["slots"]["budget"] == "5000"
    assert state["slots"]["category"] == "相机"
    print("  [OK] test_state_slots")


def test_state_full_fields():
    """AgentState 包含所有预留字段。"""
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "user_intent": "search",
        "current_agent": "ShoppingAgent",
        "task_status": "completed",
        "trace_id": "test-trace-004",
        "thread_id": "thread-abc",
        "user_id": "user-001",
        "selected_product_id": "prod-123",
        "slots": {"budget": "3000", "style": "复古"},
        "response": "推荐 Sony A7M4",
        "candidates": [{"product_id": "p1", "title": "Sony A7M4", "price": 15999}],
        "suggested_questions": ["这款相机夜拍效果如何？"],
    }
    assert state["thread_id"] == "thread-abc"
    assert state["user_id"] == "user-001"
    assert state["selected_product_id"] == "prod-123"
    assert len(state["candidates"]) == 1
    assert state["suggested_questions"][0] == "这款相机夜拍效果如何？"
    print("  [OK] test_state_full_fields")


def test_state_modify():
    """AgentState 的字段可以被修改。"""
    state: AgentState = {
        "messages": [],
        "user_intent": "",
        "current_agent": "",
        "task_status": "pending",
        "trace_id": "test-trace-005",
    }
    state["user_intent"] = "outfit"
    state["current_agent"] = "OutfitAgent"
    state["task_status"] = "in_progress"
    assert state["user_intent"] == "outfit"
    assert state["current_agent"] == "OutfitAgent"
    assert state["task_status"] == "in_progress"
    print("  [OK] test_state_modify")


if __name__ == "__main__":
    print("=" * 50)
    print("  AgentState 单元测试")
    print("=" * 50)
    test_state_init()
    test_state_with_messages()
    test_state_slots()
    test_state_full_fields()
    test_state_modify()
    print("=" * 50)
    print("  全部通过!")
    print("=" * 50)

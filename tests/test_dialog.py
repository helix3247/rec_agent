"""
tests/test_dialog.py
DialogFlowAgent 多轮对话测试。
验证：
1. 槽位缺失时生成追问（澄清式对话）
2. 闲聊意图正常响应
3. 多轮对话中槽位逐步填充
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage
from app.graph import app_graph
from app.core.logger import get_logger

log = get_logger(agent_name="TestDialog", trace_id="test-dialog")


def test_clarification_flow():
    """
    测试澄清式导购流程：
    用户输入"想买个相机"时，因 category 被提取但 budget 缺失，
    系统应转入 DialogFlow 追问预算。
    """
    print("=" * 60)
    print("  测试: 澄清式导购（槽位缺失追问）")
    print("=" * 60)

    result = app_graph.invoke({
        "messages": [HumanMessage(content="想买个相机")],
        "trace_id": "test-clarify-001",
        "thread_id": "test-thread-clarify",
        "user_id": "user-test",
    })

    intent = result.get("user_intent")
    response = result.get("response", "")
    status = result.get("task_status")
    slots = result.get("slots", {})

    print(f"  intent       : {intent}")
    print(f"  task_status  : {status}")
    print(f"  slots        : {slots}")
    print(f"  response     : {response[:120]}...")

    assert intent == "search", f"Expected intent=search, got {intent}"
    assert response, "Response should not be empty"
    print("\n  [OK] 澄清式导购测试通过")
    return True


def test_chat_flow():
    """
    测试闲聊流程：用户输入"你好"时，意图为 chat，
    系统应通过 DialogFlow 生成闲聊回复。
    """
    print("\n" + "=" * 60)
    print("  测试: 闲聊意图")
    print("=" * 60)

    result = app_graph.invoke({
        "messages": [HumanMessage(content="你好，今天天气不错")],
        "trace_id": "test-chat-001",
        "thread_id": "test-thread-chat",
        "user_id": "user-test",
    })

    intent = result.get("user_intent")
    response = result.get("response", "")
    status = result.get("task_status")

    print(f"  intent       : {intent}")
    print(f"  task_status  : {status}")
    print(f"  response     : {response[:120]}...")

    assert intent == "chat", f"Expected intent=chat, got {intent}"
    assert response, "Response should not be empty"
    print("\n  [OK] 闲聊意图测试通过")
    return True


def test_multi_turn_slot_filling():
    """
    测试多轮对话槽位填充：
    第1轮：用户说"想买个相机" -> 系统追问预算
    第2轮：用户补充"5000元以内，用来旅行拍照" -> 系统应识别到完整槽位，路由到 ShoppingAgent
    """
    print("\n" + "=" * 60)
    print("  测试: 多轮对话槽位填充")
    print("=" * 60)

    print("\n  -- 第1轮：用户说'想买个相机' --")
    result1 = app_graph.invoke({
        "messages": [HumanMessage(content="想买个相机")],
        "trace_id": "test-multi-001",
        "thread_id": "test-thread-multi",
        "user_id": "user-test",
    })

    slots1 = result1.get("slots", {})
    response1 = result1.get("response", "")
    print(f"  intent       : {result1.get('user_intent')}")
    print(f"  slots        : {slots1}")
    print(f"  response     : {response1[:120]}...")

    print("\n  -- 第2轮：用户补充'5000元以内，用来旅行拍照' --")
    messages_round2 = list(result1.get("messages", []))
    messages_round2.append(HumanMessage(content="5000元以内，用来旅行拍照"))

    result2 = app_graph.invoke({
        "messages": messages_round2,
        "trace_id": "test-multi-002",
        "thread_id": "test-thread-multi",
        "user_id": "user-test",
        "slots": slots1,
    })

    slots2 = result2.get("slots", {})
    response2 = result2.get("response", "")
    current_agent = result2.get("current_agent", "")
    print(f"  intent       : {result2.get('user_intent')}")
    print(f"  slots        : {slots2}")
    print(f"  current_agent: {current_agent}")
    print(f"  response     : {response2[:120]}...")

    assert slots2.get("budget"), "Budget should be filled after round 2"
    assert slots2.get("category"), "Category should be filled"
    print("\n  [OK] 多轮对话槽位填充测试通过")
    return True


def main():
    results = {}
    results["clarification_flow"] = test_clarification_flow()
    results["chat_flow"] = test_chat_flow()
    results["multi_turn_slot_filling"] = test_multi_turn_slot_filling()

    print("\n" + "=" * 60)
    print("  DialogFlow 测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    if all(results.values()):
        print("\n  全部通过!")
    else:
        print("\n  部分测试失败")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
tests/test_dialog.py
DialogFlowAgent 多轮对话测试。
验证：
1. 槽位缺失时生成追问（澄清式对话）
2. 闲聊意图正常响应
3. 多轮对话中槽位逐步填充

注意：这些是端到端集成测试，需要 LLM / Redis 等外部服务。
在 CI 中应通过 marker 跳过，仅在有环境时手动运行。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.graph import app_graph
from app.core.logger import get_logger

log = get_logger(agent_name="TestDialog", trace_id="test-dialog")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clarification_flow():
    """
    测试澄清式导购流程：
    用户输入"想买个相机"时，因 category 被提取但 budget 缺失，
    系统应转入 DialogFlow 追问预算。
    """
    result = await app_graph.ainvoke({
        "messages": [HumanMessage(content="想买个相机")],
        "trace_id": "test-clarify-001",
        "thread_id": "test-thread-clarify",
        "user_id": "user-test",
    })

    intent = result.get("user_intent")
    response = result.get("response", "")

    assert intent == "search", f"Expected intent=search, got {intent}"
    assert response, "Response should not be empty"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_flow():
    """
    测试闲聊流程：用户输入"你好"时，意图为 chat，
    系统应通过 DialogFlow 生成闲聊回复。
    """
    result = await app_graph.ainvoke({
        "messages": [HumanMessage(content="你好，今天天气不错")],
        "trace_id": "test-chat-001",
        "thread_id": "test-thread-chat",
        "user_id": "user-test",
    })

    intent = result.get("user_intent")
    response = result.get("response", "")

    assert intent == "chat", f"Expected intent=chat, got {intent}"
    assert response, "Response should not be empty"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_turn_slot_filling():
    """
    测试多轮对话槽位填充：
    第1轮：用户说"想买个相机" -> 系统追问预算
    第2轮：用户补充"5000元以内，用来旅行拍照" -> 系统应识别到完整槽位
    """
    result1 = await app_graph.ainvoke({
        "messages": [HumanMessage(content="想买个相机")],
        "trace_id": "test-multi-001",
        "thread_id": "test-thread-multi",
        "user_id": "user-test",
    })

    slots1 = result1.get("slots", {})

    messages_round2 = list(result1.get("messages", []))
    messages_round2.append(HumanMessage(content="5000元以内，用来旅行拍照"))

    result2 = await app_graph.ainvoke({
        "messages": messages_round2,
        "trace_id": "test-multi-002",
        "thread_id": "test-thread-multi",
        "user_id": "user-test",
        "slots": slots1,
    })

    slots2 = result2.get("slots", {})

    assert slots2.get("budget"), "Budget should be filled after round 2"
    assert slots2.get("category"), "Category should be filled"

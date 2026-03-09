"""
tests/test_intent.py
IntentParserAgent 意图识别准确率测试。

使用标准 pytest 格式，Mock 掉真实 LLM 调用，避免消耗 Token 和网络依赖。
每个测试用例独立验证意图分类和槽位抽取能力。
"""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.models.intent import IntentResult
from app.agents.intent_parser import intent_parser_node


def _make_state(query: str, **overrides) -> dict:
    """构建最小化的 AgentState 供测试用。"""
    from langchain_core.messages import HumanMessage
    state = {
        "messages": [HumanMessage(content=query)],
        "trace_id": "test-trace",
        "thread_id": "test-thread",
        "user_id": "",
        "slots": {},
        "task_status": "pending",
        "_node_metrics": [],
        "_agent_route_path": [],
    }
    state.update(overrides)
    return state


def _mock_structured_llm(intent: str, **slot_kwargs):
    """构建一个 Mock LLM，模拟 with_structured_output 返回 IntentResult。"""
    result = IntentResult(intent=intent, **slot_kwargs)
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=result)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


# ─────────────── 意图分类测试 ───────────────


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_search_intent(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("search", category="相机")

    result = await intent_parser_node(_make_state("推荐一款相机"))

    assert result["user_intent"] == "search"
    assert result["slots"]["category"] == "相机"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_search_with_budget(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm(
        "search", budget="5000元以内", category="微单", scenario="夜拍",
    )

    result = await intent_parser_node(_make_state("5000元以内适合夜拍的微单"))

    assert result["user_intent"] == "search"
    assert result["slots"]["budget"] == "5000元以内"
    assert result["slots"]["category"] == "微单"
    assert result["slots"]["scenario"] == "夜拍"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_outfit_intent(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("outfit", scenario="通勤")

    result = await intent_parser_node(_make_state("男生通勤穿搭推荐"))

    assert result["user_intent"] == "outfit"
    assert result["slots"]["scenario"] == "通勤"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_qa_intent(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("qa")

    result = await intent_parser_node(_make_state("Sony A7M4 的夜拍效果怎么样"))

    assert result["user_intent"] == "qa"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_chat_intent(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("chat")

    result = await intent_parser_node(_make_state("你好"))

    assert result["user_intent"] == "chat"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_compare_intent(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("compare")

    result = await intent_parser_node(_make_state("iPhone 16 和 Pixel 9 哪个拍照好"))

    assert result["user_intent"] == "compare"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_plan_intent(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("plan", scenario="旅行")

    result = await intent_parser_node(_make_state("去西藏旅游需要准备哪些装备"))

    assert result["user_intent"] == "plan"
    assert result["slots"]["scenario"] == "旅行"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_search_running_shoes(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm(
        "search", budget="300左右", category="运动鞋", scenario="跑步",
    )

    result = await intent_parser_node(
        _make_state("有没有适合跑步的轻便运动鞋，预算300左右"),
    )

    assert result["user_intent"] == "search"
    assert result["slots"]["budget"]
    assert result["slots"]["category"]
    assert result["slots"]["scenario"]


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_outfit_date(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("outfit", scenario="约会")

    result = await intent_parser_node(_make_state("夏天约会穿什么好看"))

    assert result["user_intent"] == "outfit"
    assert result["slots"]["scenario"] == "约会"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_chat_farewell(mock_get_llm, mock_router):
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("chat")

    result = await intent_parser_node(_make_state("谢谢你的推荐，再见"))

    assert result["user_intent"] == "chat"


# ─────────────── 降级测试 ───────────────


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_fallback_on_primary_failure(mock_get_llm, mock_router):
    """首选模型失败时应降级到备用模型。"""
    mock_router.return_value = _make_mock_router()

    fail_llm = MagicMock()
    fail_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=RuntimeError("模拟失败"),
    )

    ok_llm = _mock_structured_llm("chat")

    mock_get_llm.side_effect = [fail_llm, ok_llm]

    result = await intent_parser_node(_make_state("你好"))

    assert result["user_intent"] == "chat"


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_total_failure_defaults_to_chat(mock_get_llm, mock_router):
    """所有模型都失败时应回退到 chat 意图。"""
    mock_router.return_value = _make_mock_router()

    fail_llm = MagicMock()
    fail_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=RuntimeError("全部失败"),
    )

    mock_get_llm.return_value = fail_llm

    result = await intent_parser_node(_make_state("推荐一款相机"))

    assert result["user_intent"] == "chat"


# ─────────────── slot 合并测试 ───────────────


@pytest.mark.asyncio
@patch("app.agents.intent_parser.get_model_router")
@patch("app.agents.intent_parser.get_llm")
async def test_slot_merge_with_existing(mock_get_llm, mock_router):
    """新识别的槽位应与已有槽位合并，已有槽位不被覆盖为空。"""
    mock_router.return_value = _make_mock_router()
    mock_get_llm.return_value = _mock_structured_llm("search", category="手机")

    state = _make_state("推荐一款手机", slots={"budget": "3000以内", "scenario": "日常"})
    result = await intent_parser_node(state)

    assert result["slots"]["category"] == "手机"
    assert result["slots"]["budget"] == "3000以内"
    assert result["slots"]["scenario"] == "日常"


# ─────────────── 辅助 ───────────────


def _make_mock_router():
    from app.agents.fallback import TaskComplexity

    router = MagicMock()
    router.classify_complexity.return_value = TaskComplexity.LIGHT
    router.select_model.return_value = "primary"
    return router

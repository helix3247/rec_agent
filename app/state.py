"""
app/state.py
全局 AgentState 定义 —— LangGraph 各节点间流转的状态容器。
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Slots(TypedDict, total=False):
    """用户需求槽位（预算/用途/风格等），由 IntentParser 抽取。"""
    budget: Optional[str]
    category: Optional[str]
    scenario: Optional[str]
    style: Optional[str]
    must_have: Optional[str]


class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态。

    字段说明:
        messages      : 对话历史，使用 add_messages reducer 自动追加。
        user_intent   : 意图识别结果 (search / outfit / qa / chat / compare / plan / unknown)。
        current_agent : 当前执勤 Agent 名称。
        task_status   : 任务状态 (pending / in_progress / completed / failed)。
        trace_id      : 本次请求的链路追踪 ID。
        thread_id     : 会话 ID，用于多轮对话。
        user_id       : 用户 ID，用于个性化推荐。
        selected_product_id : 用户选中的商品 ID（收藏夹场景）。
        slots         : 需求槽位（预算/用途/风格等）。
        response      : 最终返回给用户的文本。
        candidates    : 候选商品列表。
        suggested_questions : 推荐的后续问题。
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_intent: str
    current_agent: str
    task_status: str
    trace_id: str
    thread_id: str
    user_id: str
    selected_product_id: str
    slots: Slots
    response: str
    candidates: list[dict[str, Any]]
    suggested_questions: list[str]

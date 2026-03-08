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


class PlanStep(TypedDict, total=False):
    """Planner 拆解出的单个子任务步骤。"""
    step: int
    description: str
    agent: str
    params: dict[str, Any]
    status: str
    result: str


class NodeMetrics(TypedDict, total=False):
    """单个节点的执行指标。"""
    node_name: str
    start_time: float
    end_time: float
    latency_ms: float
    token_usage: dict[str, int]
    tool_calls: list[dict[str, Any]]
    success: bool
    error: str


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
        reflection_count    : Reflector 已重试次数。
        reflection_feedback : Reflector 给上游 Agent 的修正建议。
        plan_steps          : Planner 拆解出的子任务步骤列表。
        plan_current_step   : Planner 当前执行到的步骤索引。
        plan_results        : 各子任务执行结果的汇总。
        _request_start_time : 请求开始时间戳（秒），用于计算总耗时。
        _node_metrics       : 各节点执行指标列表，由 MonitorAgent 汇总输出。
        _agent_route_path   : Agent 路由路径记录（有序节点名列表）。
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
    reflection_count: int
    reflection_feedback: str
    plan_steps: list[PlanStep]
    plan_current_step: int
    plan_results: list[dict[str, Any]]
    _request_start_time: float
    _node_metrics: list[NodeMetrics]
    _agent_route_path: list[str]

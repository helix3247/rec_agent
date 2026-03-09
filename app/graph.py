"""
app/graph.py
使用 LangGraph StateGraph 串联 Multi-Agent 流程。

流程:
    START -> IntentParser -> Dispatcher --(条件边)-->

    普通模式:
        shopping -> reflector --(条件边)--> response_formatter / 重试 / dialog
        outfit   -> reflector --(条件边)--> response_formatter / 重试 / dialog
        rag      -> response_formatter
        tool_call -> response_formatter
        dialog   -> response_formatter

    规划模式 (plan):
        planner -> (生成计划/整合结果) --(条件边)-->
            shopping -> reflector -> planner (收集结果, 继续下一步)
            outfit   -> reflector -> planner
            rag      -> planner
            response_formatter (全部完成)

    --> ResponseFormatter -> Monitor -> END

流式模式:
    使用 build_pre_formatter_graph() 构建不含 ResponseFormatter 和 Monitor 的子图，
    在 SSE 端点中手动流式调用润色 + Monitor 上报。
"""

from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.agents.intent_parser import intent_parser_node
from app.agents.dispatcher import dispatcher_node, dispatch_route
from app.agents.shopping import shopping_node
from app.agents.dialog import dialog_node
from app.agents.outfit import outfit_node
from app.agents.rag import rag_node
from app.agents.tool_call import tool_call_node
from app.agents.planner import planner_node, planner_route
from app.agents.reflector import reflector_node, reflect_route
from app.agents.response_formatter import response_formatter_node
from app.agents.monitor import monitor_node


def _rag_route(state: AgentState) -> str:
    """RAG 条件边：判断是否处于 planner 子任务模式。"""
    plan_steps = state.get("plan_steps", [])
    if plan_steps:
        return "planner"
    return "__end__"


def _rag_route_full(state: AgentState) -> str:
    """RAG 条件边（完整图）：判断是否处于 planner 子任务模式。"""
    plan_steps = state.get("plan_steps", [])
    if plan_steps:
        return "planner"
    return "response_formatter"


def _reflect_route_pre(state: AgentState) -> str:
    """Reflector 条件边（前置图）：通过时到 END 而非 response_formatter。"""
    result = reflect_route(state)
    if result == "response_formatter":
        return "__end__"
    return result


def _planner_route_pre(state: AgentState) -> str:
    """Planner 条件边（前置图）：完成时到 END 而非 response_formatter。"""
    result = planner_route(state)
    if result == "response_formatter":
        return "__end__"
    return result


def _build_common_edges(graph: StateGraph, *, pre_formatter: bool = False):
    """添加 dispatcher 之后的公共边。pre_formatter=True 时终点为 END 而非 response_formatter。"""
    graph.add_edge(START, "intent_parser")
    graph.add_edge("intent_parser", "dispatcher")

    graph.add_conditional_edges(
        "dispatcher",
        dispatch_route,
        {
            "shopping": "shopping",
            "dialog": "dialog",
            "outfit": "outfit",
            "rag": "rag",
            "tool_call": "tool_call",
            "planner": "planner",
        },
    )

    graph.add_edge("shopping", "reflector")
    graph.add_edge("outfit", "reflector")

    if pre_formatter:
        graph.add_conditional_edges(
            "reflector",
            _reflect_route_pre,
            {
                "__end__": END,
                "shopping": "shopping",
                "outfit": "outfit",
                "dialog": "dialog",
                "planner": "planner",
            },
        )
        graph.add_edge("dialog", END)
        graph.add_edge("tool_call", END)

        graph.add_conditional_edges(
            "rag",
            _rag_route,
            {"planner": "planner", "__end__": END},
        )
        graph.add_conditional_edges(
            "planner",
            _planner_route_pre,
            {
                "shopping": "shopping",
                "outfit": "outfit",
                "rag": "rag",
                "__end__": END,
            },
        )
    else:
        graph.add_conditional_edges(
            "reflector",
            reflect_route,
            {
                "response_formatter": "response_formatter",
                "shopping": "shopping",
                "outfit": "outfit",
                "dialog": "dialog",
                "planner": "planner",
            },
        )
        graph.add_edge("dialog", "response_formatter")
        graph.add_edge("tool_call", "response_formatter")

        graph.add_conditional_edges(
            "rag",
            _rag_route_full,
            {"planner": "planner", "response_formatter": "response_formatter"},
        )
        graph.add_conditional_edges(
            "planner",
            planner_route,
            {
                "shopping": "shopping",
                "outfit": "outfit",
                "rag": "rag",
                "response_formatter": "response_formatter",
            },
        )

        graph.add_edge("response_formatter", "monitor")
        graph.add_edge("monitor", END)


def build_graph() -> StateGraph:
    """构建并编译完整的 Agent 工作流图。"""
    graph = StateGraph(AgentState)

    graph.add_node("intent_parser", intent_parser_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("shopping", shopping_node)
    graph.add_node("dialog", dialog_node)
    graph.add_node("outfit", outfit_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("response_formatter", response_formatter_node)
    graph.add_node("monitor", monitor_node)

    _build_common_edges(graph, pre_formatter=False)

    return graph.compile()


def build_pre_formatter_graph():
    """
    构建前置子图 —— 执行到 ResponseFormatter 之前的所有节点。

    用于流式模式：先通过此子图完成意图解析、路由、业务处理，
    再在 SSE 端点中手动流式调用 ResponseFormatter 润色。
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent_parser", intent_parser_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("shopping", shopping_node)
    graph.add_node("dialog", dialog_node)
    graph.add_node("outfit", outfit_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reflector", reflector_node)

    _build_common_edges(graph, pre_formatter=True)

    return graph.compile()


app_graph = build_graph()

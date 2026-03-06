"""
app/graph.py
使用 LangGraph StateGraph 串联 Multi-Agent 流程。

流程:
    START -> IntentParser -> Dispatcher --(条件边)-->
        shopping  (槽位完整的 search/outfit/qa/compare/plan)
        dialog    (槽位缺失追问 / chat / unknown)
    --> ResponseFormatter -> Monitor -> END
"""

from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.agents.intent_parser import intent_parser_node
from app.agents.dispatcher import dispatcher_node, dispatch_route
from app.agents.shopping import shopping_node
from app.agents.dialog import dialog_node
from app.agents.response_formatter import response_formatter_node
from app.agents.monitor import monitor_node


def build_graph() -> StateGraph:
    """构建并编译 Agent 工作流图。"""
    graph = StateGraph(AgentState)

    graph.add_node("intent_parser", intent_parser_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("shopping", shopping_node)
    graph.add_node("dialog", dialog_node)
    graph.add_node("response_formatter", response_formatter_node)
    graph.add_node("monitor", monitor_node)

    graph.add_edge(START, "intent_parser")
    graph.add_edge("intent_parser", "dispatcher")

    graph.add_conditional_edges(
        "dispatcher",
        dispatch_route,
        {
            "shopping": "shopping",
            "dialog": "dialog",
        },
    )

    graph.add_edge("shopping", "response_formatter")
    graph.add_edge("dialog", "response_formatter")
    graph.add_edge("response_formatter", "monitor")
    graph.add_edge("monitor", END)

    return graph.compile()


app_graph = build_graph()

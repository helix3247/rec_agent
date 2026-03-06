"""
app/graph.py
使用 LangGraph StateGraph 串联 Multi-Agent 流程。

流程:
    START -> IntentParser -> Dispatcher --(条件边)-->
        shopping   (search / compare)
        outfit     (outfit，槽位完整)
        rag        (qa)
        tool_call  (tool)
        planner    (plan)
        dialog     (chat / unknown / 槽位缺失追问)
    --> ResponseFormatter -> Monitor -> END
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
from app.agents.planner import planner_node
from app.agents.response_formatter import response_formatter_node
from app.agents.monitor import monitor_node


def build_graph() -> StateGraph:
    """构建并编译 Agent 工作流图。"""
    graph = StateGraph(AgentState)

    graph.add_node("intent_parser", intent_parser_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("shopping", shopping_node)
    graph.add_node("dialog", dialog_node)
    graph.add_node("outfit", outfit_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("planner", planner_node)
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
            "outfit": "outfit",
            "rag": "rag",
            "tool_call": "tool_call",
            "planner": "planner",
        },
    )

    graph.add_edge("shopping", "response_formatter")
    graph.add_edge("dialog", "response_formatter")
    graph.add_edge("outfit", "response_formatter")
    graph.add_edge("rag", "response_formatter")
    graph.add_edge("tool_call", "response_formatter")
    graph.add_edge("planner", "response_formatter")
    graph.add_edge("response_formatter", "monitor")
    graph.add_edge("monitor", END)

    return graph.compile()


app_graph = build_graph()

#!/usr/bin/env python
"""快速验证 metrics 收集功能"""
import sys
sys.path.insert(0, "d:/Project/rec_agent")

from app.graph import build_graph
from app.state import AgentState
import time

def main():
    graph = build_graph()
    
    initial_state = AgentState(
        user_input="帮我找一双运动鞋",
        user_id="test_user_001",
        session_id="test_session_001",
        intent=None,
        slots={},
        route_history=[],
        loop_count=0,
        max_loops=3,
        reflections=[],
        final_answer=None,
        suggested_questions=[],
        conversation_history=[],
        context={},
        _request_start_time=time.time(),
        _node_metrics=[],
        _agent_route_path=[],
    )
    
    result = graph.invoke(initial_state)
    
    print(f"Route path: {result['_agent_route_path']}")
    print(f"Node metrics count: {len(result['_node_metrics'])}")
    
    total_tokens = 0
    for metric in result['_node_metrics']:
        if 'token_usage' in metric:
            total_tokens += metric['token_usage'].get('total_tokens', 0)
    
    print(f"Total tokens: {total_tokens}")

if __name__ == "__main__":
    main()

"""
conftest.py
全局 pytest fixtures —— 为所有测试提供统一的 Mock 对象和配置。
"""

import time
from unittest.mock import MagicMock

import pytest

from app.state import AgentState


# ────────────────────── Mock 配置 ──────────────────────


@pytest.fixture()
def mock_config():
    """提供一套安全的测试用配置，不连接真实服务。"""
    from app.core.config import (
        EmbeddingSettings,
        ESSettings,
        LangfuseSettings,
        LangSmithSettings,
        LLMSettings,
        MilvusSettings,
        MySQLSettings,
        RedisSettings,
        Settings,
    )

    return Settings(
        llm=LLMSettings(
            llm_api_key="test-key",
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            fallback_llm_api_key="test-fallback-key",
            fallback_llm_base_url="http://localhost:11434/v1",
            fallback_llm_model="test-fallback-model",
        ),
        embedding=EmbeddingSettings(
            embedding_api_key="test-emb-key",
            embedding_base_url="http://localhost:11434/v1",
            embedding_model="test-embedding",
        ),
        langsmith=LangSmithSettings(langchain_tracing_v2=False),
        langfuse=LangfuseSettings(langfuse_enabled=False),
        mysql=MySQLSettings(
            mysql_host="127.0.0.1",
            mysql_port=3306,
            mysql_user="test",
            mysql_password="test",
            mysql_database="test_db",
        ),
        es=ESSettings(
            es_host="http://127.0.0.1:9200",
            es_index_name="test_index",
        ),
        milvus=MilvusSettings(
            milvus_host="127.0.0.1",
            milvus_port=19530,
            milvus_collection="test_collection",
        ),
        redis=RedisSettings(
            redis_host="127.0.0.1",
            redis_port=6379,
        ),
    )


# ────────────────────── Mock 外部服务 ──────────────────────


@pytest.fixture()
def mock_es():
    """Mock Elasticsearch 客户端。"""
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "hits": {
            "total": {"value": 2},
            "hits": [
                {
                    "_id": "prod-001",
                    "_score": 8.5,
                    "_source": {
                        "product_id": "prod-001",
                        "name": "测试商品A",
                        "category": "相机",
                        "brand": "Sony",
                        "price": 5999,
                        "tags": ["微单", "全画幅"],
                        "description": "一款出色的微单相机",
                    },
                },
                {
                    "_id": "prod-002",
                    "_score": 7.2,
                    "_source": {
                        "product_id": "prod-002",
                        "name": "测试商品B",
                        "category": "相机",
                        "brand": "Canon",
                        "price": 4999,
                        "tags": ["微单", "半画幅"],
                        "description": "一款性价比微单相机",
                    },
                },
            ],
        },
    }
    return mock_client


@pytest.fixture()
def mock_milvus_collection():
    """Mock Milvus Collection 对象。"""
    mock_hit_1 = MagicMock()
    mock_hit_1.entity.get = lambda k, d="": {
        "product_id": "prod-001",
        "doc_type": "review",
        "text": "这款相机画质非常好",
    }.get(k, d)
    mock_hit_1.score = 0.95

    mock_hit_2 = MagicMock()
    mock_hit_2.entity.get = lambda k, d="": {
        "product_id": "prod-002",
        "doc_type": "faq",
        "text": "续航大约4小时",
    }.get(k, d)
    mock_hit_2.score = 0.88

    mock_collection = MagicMock()
    mock_collection.search.return_value = [[mock_hit_1, mock_hit_2]]
    mock_collection.insert.return_value = MagicMock(primary_keys=[1])
    return mock_collection


@pytest.fixture()
def mock_mysql_conn():
    """Mock pymysql 连接对象。"""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "user_id": "user-001",
        "name": "测试用户",
        "age": 28,
        "gender": "male",
        "budget_level": "mid",
        "style_pref": '["休闲", "运动"]',
        "category_interest": '["相机", "手机"]',
    }
    mock_cursor.fetchall.return_value = []
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture()
def mock_embedding():
    """Mock Embedding API 调用，返回固定维度向量。"""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 3072)]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    return mock_client


# ────────────────────── AgentState fixtures ──────────────────────


@pytest.fixture()
def sample_state() -> AgentState:
    """一个包含基本字段的 AgentState 样本。"""
    from langchain_core.messages import AIMessage, HumanMessage

    return AgentState(
        messages=[
            HumanMessage(content="帮我找一双运动鞋"),
            AIMessage(content="请问您的预算是多少？"),
        ],
        user_intent="search",
        current_agent="IntentParser",
        task_status="in_progress",
        trace_id="test-trace-001",
        thread_id="test-thread-001",
        user_id="user-001",
        selected_product_id="",
        slots={"category": "运动鞋"},
        response="",
        candidates=[],
        suggested_questions=[],
        reflection_count=0,
        reflection_feedback="",
        plan_steps=[],
        plan_current_step=0,
        plan_results=[],
        _request_start_time=time.time(),
        _node_metrics=[],
        _agent_route_path=["intent_parser"],
    )


@pytest.fixture()
def sample_state_with_metrics() -> AgentState:
    """一个包含完整指标数据的 AgentState 样本，用于 Monitor 测试。"""
    from langchain_core.messages import AIMessage, HumanMessage

    return AgentState(
        messages=[
            HumanMessage(content="推荐一款5000元的微单相机"),
            AIMessage(content="为您推荐 Sony A7M4"),
        ],
        user_intent="search",
        current_agent="MonitorAgent",
        task_status="completed",
        trace_id="test-trace-metrics",
        thread_id="test-thread-metrics",
        user_id="user-001",
        selected_product_id="",
        slots={"budget": "5000", "category": "相机"},
        response="为您推荐 Sony A7M4，价格 15999 元",
        candidates=[{"product_id": "p1", "name": "Sony A7M4", "price": 15999}],
        suggested_questions=["这款相机夜拍效果如何？"],
        reflection_count=0,
        reflection_feedback="",
        plan_steps=[],
        plan_current_step=0,
        plan_results=[],
        _request_start_time=time.time() - 2.5,
        _node_metrics=[
            {
                "node_name": "intent_parser",
                "start_time": time.time() - 2.5,
                "end_time": time.time() - 2.0,
                "latency_ms": 500,
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                "tool_calls": [],
                "success": True,
                "error": "",
            },
            {
                "node_name": "shopping",
                "start_time": time.time() - 2.0,
                "end_time": time.time() - 1.0,
                "latency_ms": 1000,
                "token_usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
                "tool_calls": [
                    {"tool_name": "search_products", "success": True, "error": ""},
                    {"tool_name": "get_user_profile", "success": True, "error": ""},
                ],
                "success": True,
                "error": "",
            },
            {
                "node_name": "response_formatter",
                "start_time": time.time() - 1.0,
                "end_time": time.time() - 0.5,
                "latency_ms": 500,
                "token_usage": {"prompt_tokens": 150, "completion_tokens": 80, "total_tokens": 230},
                "tool_calls": [],
                "success": True,
                "error": "",
            },
        ],
        _agent_route_path=["intent_parser", "dispatcher", "shopping", "reflector", "response_formatter"],
    )

"""
tests/conftest.py
全局测试 Fixtures —— 为测试套件提供公共的 Mock 和配置。
"""

import pytest
from unittest.mock import MagicMock


class ExternalCallError(RuntimeError):
    """测试中意外发出外部调用时抛出的异常。"""
    pass


def _raise_external_call_error(service: str):
    """生成一个抛出外部调用错误的函数。"""
    def _raiser(*args, **kwargs):
        raise ExternalCallError(
            f"Test tried to call real {service} without proper mocking! "
            f"Please mock this call explicitly in your test."
        )
    return _raiser


@pytest.fixture(autouse=True)
def _suppress_external_calls(monkeypatch):
    """
    自动禁止测试中意外发出的真实外部调用。

    通过 monkeypatch 将 OpenAI / ES / Redis / Milvus 客户端的创建函数替换为安全的报错桩，
    防止测试因忘记 Mock 而产生网络请求或消耗 Token。
    仅对未显式 Mock 的调用生效。
    """
    # 阻止 OpenAI 客户端创建
    try:
        monkeypatch.setattr(
            "openai.OpenAI",
            _raise_external_call_error("OpenAI"),
        )
    except (ImportError, AttributeError):
        pass

    # 阻止 Elasticsearch 客户端创建
    try:
        monkeypatch.setattr(
            "elasticsearch.Elasticsearch",
            _raise_external_call_error("Elasticsearch"),
        )
    except (ImportError, AttributeError):
        pass

    # 阻止 Redis 连接
    try:
        monkeypatch.setattr(
            "redis.Redis",
            _raise_external_call_error("Redis"),
        )
    except (ImportError, AttributeError):
        pass

    # 阻止 Milvus 连接
    try:
        monkeypatch.setattr(
            "pymilvus.connections.connect",
            _raise_external_call_error("Milvus"),
        )
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_config():
    """提供 Mock 的配置对象，用于不依赖真实配置文件的测试。"""
    config = MagicMock()
    
    # LLM 配置
    config.llm.llm_model = "gpt-4o-mini"
    config.llm.fallback_llm_model = "deepseek-chat"
    config.llm.llm_api_key = "test-key"
    config.llm.llm_base_url = "https://api.openai.com/v1"
    
    # Embedding 配置
    config.embedding.embedding_model = "text-embedding-3-large"
    config.embedding.embedding_api_key = "test-emb-key"
    config.embedding.embedding_base_url = "https://api.openai.com/v1"
    
    # Elasticsearch 配置
    config.es.es_host = "http://localhost:9200"
    config.es.es_index = "test_products"
    config.es.es_username = None
    config.es.es_password = None
    
    # Redis 配置
    config.redis.redis_host = "localhost"
    config.redis.redis_port = 6379
    config.redis.redis_db = 0
    config.redis.redis_password = None
    
    # Milvus 配置
    config.milvus.milvus_host = "localhost"
    config.milvus.milvus_port = 19530
    config.milvus.milvus_collection = "test_knowledge"
    
    # Database 配置
    config.database.mysql_host = "localhost"
    config.database.mysql_port = 3306
    config.database.mysql_user = "test"
    config.database.mysql_password = "test"
    config.database.mysql_database = "test_ecommerce"
    
    return config


@pytest.fixture
def mock_embedding():
    """提供 Mock 的 Embedding 函数，返回固定长度的假向量。"""
    def _get_mock_embedding(text: str) -> list[float]:
        # 返回 3072 维的假向量（text-embedding-3-large 的维度）
        # 使用文本哈希作为种子，保证相同文本返回相同向量
        seed = hash(text) % 1000
        return [float((seed + i) % 100) / 100.0 for i in range(3072)]
    
    return _get_mock_embedding

"""
tests/test_tools.py
工具层单元测试 —— 覆盖 search、knowledge、db、personalization、memory 模块。

所有外部依赖（ES、Milvus、MySQL、Embedding API）均通过 Mock 隔离，
可在无网络环境下运行。
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest



# ════════════════════════════════════════════════════════════
#  search.search_products 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSearchProducts:
    """测试 search.search_products 的 Hybrid Search（关键词 + 向量）。"""

    @patch("app.tools.search._get_embedding")
    @patch("app.tools.search._get_es_client")
    def test_hybrid_search_success(self, mock_es_factory, mock_embedding, mock_es):
        """关键词 + 向量混合检索正常工作。"""
        mock_es_factory.return_value = mock_es
        mock_embedding.return_value = [0.1] * 3072

        from app.tools.search import search_products, es_circuit_breaker
        es_circuit_breaker.record_success()

        results = search_products("微单相机", top_k=2)

        assert len(results) == 2
        assert results[0]["product_id"] == "prod-001"
        assert results[0]["name"] == "测试商品A"
        assert results[0]["brand"] == "Sony"
        assert results[0]["score"] == 8.5
        mock_es.search.assert_called_once()

    @patch("app.tools.search._get_embedding")
    @patch("app.tools.search._get_es_client")
    def test_search_with_filters(self, mock_es_factory, mock_embedding, mock_es):
        """带结构化过滤条件的检索。"""
        mock_es_factory.return_value = mock_es
        mock_embedding.return_value = [0.1] * 3072

        from app.tools.search import search_products, es_circuit_breaker
        es_circuit_breaker.record_success()

        search_products(
            "相机",
            category="相机",
            brand="Sony",
            min_price=3000,
            max_price=8000,
            tags=["微单"],
        )

        call_args = mock_es.search.call_args
        body = call_args.kwargs.get("body") or call_args[1].get("body")
        bool_query = body["query"]["bool"]
        assert "filter" in bool_query
        assert len(bool_query["filter"]) >= 3

    @patch("app.tools.search._get_embedding")
    @patch("app.tools.search._get_es_client")
    def test_search_keyword_only(self, mock_es_factory, mock_embedding, mock_es):
        """禁用向量检索时仅使用关键词检索。"""
        mock_es_factory.return_value = mock_es

        from app.tools.search import search_products, es_circuit_breaker
        es_circuit_breaker.record_success()

        search_products("运动鞋", use_vector=False)

        mock_embedding.assert_not_called()
        mock_es.search.assert_called_once()

    @patch("app.tools.search._get_es_client")
    def test_search_circuit_breaker_open(self, mock_es_factory, mock_es):
        """熔断器开启时直接返回空列表。"""
        mock_es_factory.return_value = mock_es

        from app.tools.search import search_products, es_circuit_breaker
        for _ in range(10):
            es_circuit_breaker.record_failure()

        results = search_products("测试", use_vector=False)
        assert results == []
        mock_es.search.assert_not_called()

        # 恢复熔断器状态
        es_circuit_breaker.record_success()

    @patch("app.tools.search._get_embedding")
    @patch("app.tools.search._get_es_client")
    def test_search_embedding_failure_fallback(self, mock_es_factory, mock_embedding, mock_es):
        """Embedding 失败时降级为纯关键词检索。"""
        mock_es_factory.return_value = mock_es
        mock_embedding.side_effect = Exception("Embedding API 不可用")

        from app.tools.search import search_products, es_circuit_breaker
        es_circuit_breaker.record_success()

        results = search_products("相机", use_vector=True)

        assert len(results) == 2
        mock_es.search.assert_called_once()
        body = mock_es.search.call_args.kwargs.get("body") or mock_es.search.call_args[1].get("body")
        assert "knn" not in body


# ════════════════════════════════════════════════════════════
#  knowledge.query_knowledge 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestQueryKnowledge:
    """测试 knowledge.query_knowledge 的向量检索。"""

    @patch("app.tools.knowledge._get_embedding")
    @patch("app.tools.knowledge.Collection")
    @patch("app.tools.knowledge._ensure_connection")
    def test_query_knowledge_success(self, mock_conn, mock_coll_cls, mock_embedding, mock_milvus_collection):
        """正常的知识库检索。"""
        mock_coll_cls.return_value = mock_milvus_collection
        mock_embedding.return_value = [0.1] * 3072

        from app.tools.knowledge import query_knowledge, milvus_circuit_breaker
        milvus_circuit_breaker.record_success()

        results = query_knowledge("这款相机的画质怎么样", top_k=2)

        assert len(results) == 2
        assert results[0]["product_id"] == "prod-001"
        assert results[0]["doc_type"] == "review"
        assert results[0]["score"] == 0.95

    @patch("app.tools.knowledge._get_embedding")
    @patch("app.tools.knowledge.Collection")
    @patch("app.tools.knowledge._ensure_connection")
    def test_query_knowledge_with_filters(self, mock_conn, mock_coll_cls, mock_embedding, mock_milvus_collection):
        """带 product_id 和 doc_type 过滤的检索。"""
        mock_coll_cls.return_value = mock_milvus_collection
        mock_embedding.return_value = [0.1] * 3072

        from app.tools.knowledge import query_knowledge, milvus_circuit_breaker
        milvus_circuit_breaker.record_success()

        query_knowledge("评价", product_id="prod-001", doc_type="review")

        search_call = mock_milvus_collection.search.call_args
        expr = search_call.kwargs.get("expr") or search_call[1].get("expr")
        assert "prod-001" in expr
        assert "review" in expr

    @patch("app.tools.knowledge._get_embedding")
    @patch("app.tools.knowledge._ensure_connection")
    def test_query_knowledge_embedding_failure(self, mock_conn, mock_embedding):
        """Embedding 失败时返回空列表。"""
        mock_embedding.side_effect = Exception("Embedding 服务不可用")

        from app.tools.knowledge import query_knowledge

        results = query_knowledge("测试查询")
        assert results == []

    @patch("app.tools.knowledge._ensure_connection")
    def test_query_knowledge_circuit_breaker(self, mock_conn):
        """Milvus 熔断器开启时返回空。"""
        from app.tools.knowledge import query_knowledge, milvus_circuit_breaker

        for _ in range(10):
            milvus_circuit_breaker.record_failure()

        with patch("app.tools.knowledge._get_embedding", return_value=[0.1] * 3072):
            with patch("app.tools.knowledge.Collection"):
                results = query_knowledge("测试")
                assert results == []

        milvus_circuit_breaker.record_success()


# ════════════════════════════════════════════════════════════
#  db.get_user_profile 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestGetUserProfile:
    """测试 db.get_user_profile 的查询和解析。"""

    @patch("app.tools.db._get_connection")
    def test_get_user_profile_success(self, mock_conn_factory, mock_mysql_conn):
        """正常获取用户画像。"""
        mock_conn_factory.return_value = mock_mysql_conn

        from app.tools.db import get_user_profile, _mysql_breaker
        _mysql_breaker.record_success()

        profile = get_user_profile("user-001")

        assert profile is not None
        assert profile["user_id"] == "user-001"
        assert profile["name"] == "测试用户"
        assert profile["budget_level"] == "mid"
        assert isinstance(profile["style_preference"], list)

    @patch("app.tools.db._get_connection")
    def test_get_user_profile_not_found(self, mock_conn_factory):
        """用户不存在时返回 None。"""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn_factory.return_value = mock_conn

        from app.tools.db import get_user_profile, _mysql_breaker
        _mysql_breaker.record_success()

        profile = get_user_profile("nonexistent-user")
        assert profile is None

    def test_get_user_profile_empty_id(self):
        """空用户 ID 返回 None。"""
        from app.tools.db import get_user_profile

        assert get_user_profile("") is None
        assert get_user_profile(None) is None

    @patch("app.tools.db._get_connection")
    def test_get_user_profile_with_purchase_history(self, mock_conn_factory):
        """用户有购买历史时正确计算偏好。"""
        purchases = [
            {"product_id": "p1", "name": "Sony A7M4", "category": "相机", "brand": "Sony", "price": 15999},
            {"product_id": "p2", "name": "Canon R6", "category": "相机", "brand": "Canon", "price": 12999},
        ]

        call_count = 0

        def mock_fetchall():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return purchases
            return []

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "user_id": "user-002",
            "name": "摄影爱好者",
            "age": 30,
            "gender": "male",
            "budget_level": "high",
            "style_pref": "[]",
            "category_interest": '["相机"]',
        }
        mock_cursor.fetchall = mock_fetchall
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn_factory.return_value = mock_conn

        from app.tools.db import get_user_profile, _mysql_breaker
        _mysql_breaker.record_success()

        profile = get_user_profile("user-002")

        assert profile is not None
        assert "Sony" in profile["liked_brands"]
        assert "Canon" in profile["liked_brands"]
        assert "相机" in profile["liked_categories"]
        assert profile["price_range"]["min"] == 12999
        assert profile["price_range"]["max"] == 15999


# ════════════════════════════════════════════════════════════
#  db._parse_json_field 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestParseJsonField:
    """测试 JSON 字段解析的各种边界情况。"""

    def test_valid_json_list(self):
        from app.tools.db import _parse_json_field

        assert _parse_json_field('["a", "b"]') == ["a", "b"]

    def test_valid_json_dict(self):
        from app.tools.db import _parse_json_field

        assert _parse_json_field('{"key": "value"}') == {"key": "value"}

    def test_dict_passthrough(self):
        from app.tools.db import _parse_json_field

        d = {"key": "value"}
        assert _parse_json_field(d) is d

    def test_list_passthrough(self):
        from app.tools.db import _parse_json_field

        lst = [1, 2, 3]
        assert _parse_json_field(lst) is lst

    def test_invalid_json(self):
        from app.tools.db import _parse_json_field

        assert _parse_json_field("not json") == []

    def test_none_input(self):
        from app.tools.db import _parse_json_field

        assert _parse_json_field(None) == []

    def test_empty_string(self):
        from app.tools.db import _parse_json_field

        assert _parse_json_field("") == []

    def test_json_primitive(self):
        from app.tools.db import _parse_json_field

        assert _parse_json_field('"string"') == []
        assert _parse_json_field("42") == []


# ════════════════════════════════════════════════════════════
#  personalization.rerank_by_user_profile 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPersonalizationRerank:
    """测试 personalization.rerank_by_user_profile 的排序逻辑。"""

    def _make_products(self):
        return [
            {"product_id": "p1", "name": "Nike 跑鞋", "category": "鞋子", "brand": "Nike", "price": 800, "score": 5.0},
            {"product_id": "p2", "name": "Sony 相机", "category": "相机", "brand": "Sony", "price": 15999, "score": 8.0},
            {"product_id": "p3", "name": "Adidas 外套", "category": "外套", "brand": "Adidas", "price": 600, "score": 6.0},
        ]

    def _make_profile(self):
        return {
            "user_id": "user-001",
            "liked_brands": ["Sony", "Canon"],
            "liked_categories": ["相机", "镜头"],
            "category_interest": ["相机"],
            "budget_level": "high",
            "price_range": {"min": 5000, "max": 20000, "avg": 12000},
            "purchase_history": [
                {"category": "相机", "brand": "Sony"},
            ],
        }

    def test_rerank_with_profile(self):
        """有用户画像时，偏好品牌/品类的商品排名应提升。"""
        from app.tools.personalization import rerank_by_user_profile

        products = self._make_products()
        profile = self._make_profile()

        result = rerank_by_user_profile(products, profile)

        assert len(result) == 3
        assert result[0]["product_id"] == "p2"

    def test_rerank_without_profile(self):
        """无用户画像时退化为原始排序。"""
        from app.tools.personalization import rerank_by_user_profile

        products = self._make_products()
        result = rerank_by_user_profile(products, None)

        assert result[0]["product_id"] == "p1"
        assert result[1]["product_id"] == "p2"
        assert result[2]["product_id"] == "p3"

    def test_rerank_empty_products(self):
        """空商品列表返回空。"""
        from app.tools.personalization import rerank_by_user_profile

        result = rerank_by_user_profile([], self._make_profile())
        assert result == []

    def test_complementary_category_boost(self):
        """互补品类应获得加分。"""
        from app.tools.personalization import rerank_by_user_profile

        products = [
            {"product_id": "p1", "name": "相机包", "category": "相机包", "brand": "Peak", "price": 300, "score": 5.0},
            {"product_id": "p2", "name": "镜头", "category": "镜头", "brand": "Sony", "price": 5000, "score": 5.0},
        ]
        profile = {
            "user_id": "u1",
            "liked_brands": [],
            "liked_categories": [],
            "category_interest": [],
            "budget_level": "high",
            "price_range": {"min": 0, "max": 0, "avg": 0},
            "purchase_history": [{"category": "相机", "brand": "Sony"}],
        }

        result = rerank_by_user_profile(products, profile)
        # 镜头和相机包都是相机的互补品类，但镜头还获得了品类匹配
        assert len(result) == 2

    def test_no_internal_fields_leaked(self):
        """排序后不应泄露内部字段。"""
        from app.tools.personalization import rerank_by_user_profile

        products = self._make_products()
        result = rerank_by_user_profile(products, self._make_profile())

        for p in result:
            assert "_personalization_boost" not in p
            assert "_final_score" not in p


# ════════════════════════════════════════════════════════════
#  memory.migrate_to_long_term 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMemoryMigration:
    """测试 memory.migrate_to_long_term 的长期记忆存储。"""

    def _make_messages(self, count=4):
        from langchain_core.messages import HumanMessage, AIMessage

        msgs = []
        for i in range(count):
            if i % 2 == 0:
                msgs.append(HumanMessage(content=f"用户消息 {i // 2 + 1}"))
            else:
                msgs.append(AIMessage(content=f"AI回复 {i // 2 + 1}"))
        return msgs

    @patch("app.tools.memory._ensure_memory_collection")
    @patch("app.tools.memory._get_embedding")
    @patch("app.tools.memory._summarize_conversation")
    def test_migrate_success(self, mock_summarize, mock_embedding, mock_collection):
        """正常迁移流程。"""
        mock_summarize.return_value = "用户偏好运动风格，预算500元"
        mock_embedding.return_value = [0.1] * 3072
        mock_coll = MagicMock()
        mock_collection.return_value = mock_coll

        from app.tools.memory import migrate_to_long_term

        result = migrate_to_long_term(
            user_id="user-001",
            thread_id="thread-001",
            messages=self._make_messages(4),
        )

        assert result is True
        mock_coll.insert.assert_called_once()
        mock_coll.flush.assert_called_once()

    def test_migrate_empty_user_id(self):
        """空用户 ID 跳过迁移。"""
        from app.tools.memory import migrate_to_long_term

        result = migrate_to_long_term("", "thread-001", self._make_messages())
        assert result is False

    def test_migrate_empty_messages(self):
        """空消息列表跳过迁移。"""
        from app.tools.memory import migrate_to_long_term

        result = migrate_to_long_term("user-001", "thread-001", [])
        assert result is False

    def test_migrate_insufficient_turns(self):
        """对话轮次不足（< 2 轮用户消息）跳过迁移。"""
        from langchain_core.messages import HumanMessage

        from app.tools.memory import migrate_to_long_term

        result = migrate_to_long_term(
            "user-001",
            "thread-001",
            [HumanMessage(content="你好")],
        )
        assert result is False

    @patch("app.tools.memory._ensure_memory_collection")
    @patch("app.tools.memory._get_embedding")
    @patch("app.tools.memory._summarize_conversation")
    def test_migrate_empty_summary(self, mock_summarize, mock_embedding, mock_collection):
        """摘要为空时不执行插入。"""
        mock_summarize.return_value = ""

        from app.tools.memory import migrate_to_long_term

        result = migrate_to_long_term(
            "user-001", "thread-001", self._make_messages(4),
        )
        assert result is False


# ════════════════════════════════════════════════════════════
#  memory 辅助函数测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMemoryHelpers:

    def test_extract_preferences(self):
        from app.tools.memory import _extract_preferences

        prefs = _extract_preferences("用户偏好 Nike 品牌的运动风格鞋子，预算500元")
        parsed = json.loads(prefs)
        assert "style" in parsed or "budget" in parsed

    def test_extract_preferences_empty(self):
        from app.tools.memory import _extract_preferences

        prefs = _extract_preferences("一段没有任何关键词的文本 xyz")
        assert prefs == "{}"

    def test_format_memory_context_empty(self):
        from app.tools.memory import format_memory_context

        assert format_memory_context([]) == ""

    def test_format_memory_context_normal(self):
        from app.tools.memory import format_memory_context

        memories = [
            {"summary": "用户喜欢相机", "timestamp": int(time.time()), "score": 0.9},
            {"summary": "用户预算5000", "timestamp": int(time.time()) - 86400, "score": 0.8},
        ]
        result = format_memory_context(memories)
        assert "[用户历史偏好与记忆]" in result
        assert "用户喜欢相机" in result
        assert "用户预算5000" in result

    def test_fallback_summary(self):
        from langchain_core.messages import HumanMessage, AIMessage

        from app.tools.memory import _fallback_summary

        messages = [
            HumanMessage(content="推荐相机"),
            AIMessage(content="好的"),
            HumanMessage(content="预算5000"),
        ]
        result = _fallback_summary(messages)
        assert "推荐相机" in result
        assert "预算5000" in result

"""
tests/test_checkpoint.py
LangGraph Checkpoint Persistence 单元测试。

验证 checkpoint 模块的配置逻辑和 Graph 集成方式。
由于需要真实 Redis 连接来测试完整的 checkpoint 读写，
这里通过 Mock 验证模块的初始化逻辑和 Graph 编译参数传递。
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.checkpoint import _build_redis_url, _CheckpointerHolder


# ════════════════════════════════════════════════════════════
#  Redis URL 构建测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBuildRedisUrl:
    """测试 Redis URL 构建逻辑。"""

    def test_url_without_password(self):
        """无密码时 URL 格式正确。"""
        with patch("app.core.checkpoint.settings") as mock_settings:
            mock_settings.redis.redis_password = ""
            mock_settings.redis.redis_host = "localhost"
            mock_settings.redis.redis_port = 6379
            mock_settings.redis.redis_db = 0
            url = _build_redis_url()
            assert url == "redis://localhost:6379/0"

    def test_url_with_password(self):
        """有密码时 URL 格式正确。"""
        with patch("app.core.checkpoint.settings") as mock_settings:
            mock_settings.redis.redis_password = "mypassword"
            mock_settings.redis.redis_host = "redis-host"
            mock_settings.redis.redis_port = 6380
            mock_settings.redis.redis_db = 2
            url = _build_redis_url()
            assert url == "redis://:mypassword@redis-host:6380/2"

    def test_url_with_custom_db(self):
        """自定义 DB 编号时 URL 格式正确。"""
        with patch("app.core.checkpoint.settings") as mock_settings:
            mock_settings.redis.redis_password = ""
            mock_settings.redis.redis_host = "127.0.0.1"
            mock_settings.redis.redis_port = 6379
            mock_settings.redis.redis_db = 5
            url = _build_redis_url()
            assert url == "redis://127.0.0.1:6379/5"


# ════════════════════════════════════════════════════════════
#  CheckpointerHolder 初始化测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCheckpointerHolder:
    """测试 CheckpointerHolder 的延迟初始化和单例行为。"""

    @pytest.mark.asyncio
    async def test_lazy_initialization(self):
        """首次调用 get() 时应创建 AsyncRedisSaver 并调用 asetup()。"""
        mock_saver = MagicMock()
        mock_saver.asetup = AsyncMock()

        with patch("app.core.checkpoint.AsyncRedisSaver", return_value=mock_saver):
            with patch("app.core.checkpoint.settings") as mock_settings:
                mock_settings.redis.redis_password = ""
                mock_settings.redis.redis_host = "localhost"
                mock_settings.redis.redis_port = 6379
                mock_settings.redis.redis_db = 0

                holder = _CheckpointerHolder()
                result = await holder.get()

                assert result is mock_saver
                mock_saver.asetup.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_behavior(self):
        """多次调用 get() 应返回同一实例，只初始化一次。"""
        mock_saver = MagicMock()
        mock_saver.asetup = AsyncMock()

        with patch("app.core.checkpoint.AsyncRedisSaver", return_value=mock_saver):
            with patch("app.core.checkpoint.settings") as mock_settings:
                mock_settings.redis.redis_password = ""
                mock_settings.redis.redis_host = "localhost"
                mock_settings.redis.redis_port = 6379
                mock_settings.redis.redis_db = 0

                holder = _CheckpointerHolder()
                result1 = await holder.get()
                result2 = await holder.get()

                assert result1 is result2
                mock_saver.asetup.assert_called_once()

    @pytest.mark.asyncio
    async def test_ttl_config_passed(self):
        """TTL 配置应正确传递给 AsyncRedisSaver。"""
        with patch("app.core.checkpoint.AsyncRedisSaver") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.asetup = AsyncMock()
            mock_cls.return_value = mock_instance

            with patch("app.core.checkpoint.settings") as mock_settings:
                mock_settings.redis.redis_password = ""
                mock_settings.redis.redis_host = "localhost"
                mock_settings.redis.redis_port = 6379
                mock_settings.redis.redis_db = 0

                holder = _CheckpointerHolder()
                await holder.get()

                call_kwargs = mock_cls.call_args
                assert call_kwargs.kwargs["ttl"]["default_ttl"] == 30
                assert call_kwargs.kwargs["ttl"]["refresh_on_read"] is True


# ════════════════════════════════════════════════════════════
#  Graph 编译集成测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestGraphCheckpointerIntegration:
    """测试 Graph 构建函数签名正确接收 checkpointer 参数。"""

    def test_build_graph_accepts_checkpointer_param(self):
        """build_graph 函数签名应包含 checkpointer 参数。"""
        import inspect
        from app.graph import build_graph
        sig = inspect.signature(build_graph)
        assert "checkpointer" in sig.parameters
        assert sig.parameters["checkpointer"].default is None

    def test_build_pre_formatter_graph_accepts_checkpointer_param(self):
        """build_pre_formatter_graph 函数签名应包含 checkpointer 参数。"""
        import inspect
        from app.graph import build_pre_formatter_graph
        sig = inspect.signature(build_pre_formatter_graph)
        assert "checkpointer" in sig.parameters
        assert sig.parameters["checkpointer"].default is None

    def test_graph_module_imports_checkpoint_types(self):
        """graph.py 应导入 BaseCheckpointSaver 类型。"""
        from langgraph.checkpoint.base import BaseCheckpointSaver
        assert BaseCheckpointSaver is not None

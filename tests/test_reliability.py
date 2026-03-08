"""
tests/test_reliability.py
可靠性机制单元测试 —— 覆盖 retry_with_backoff、CircuitBreaker、IdempotencyGuard。

所有测试无外部依赖，直接验证可靠性模块的核心逻辑。
"""

import asyncio
import time
import pytest

from app.core.reliability import (
    CircuitBreaker,
    IdempotencyGuard,
    retry_with_backoff,
    sync_timeout_call,
    timeout_call,
)


# ════════════════════════════════════════════════════════════
#  retry_with_backoff 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRetryWithBackoff:
    """测试 retry_with_backoff 装饰器的重试次数和退避时间。"""

    def test_sync_success_no_retry(self):
        """首次成功不触发重试。"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_sync_retry_then_success(self):
        """前几次失败后成功。"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("模拟连接失败")
            return "recovered"

        result = fail_then_succeed()
        assert result == "recovered"
        assert call_count == 3

    def test_sync_exhaust_retries(self):
        """重试耗尽后抛出原始异常。"""

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            raise ValueError("持续失败")

        with pytest.raises(ValueError, match="持续失败"):
            always_fail()

    def test_sync_retry_count(self):
        """验证重试次数精确匹配 max_retries。"""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def count_calls():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("失败")

        with pytest.raises(RuntimeError):
            count_calls()

        # 1 次初始 + 2 次重试 = 3 次
        assert call_count == 3

    def test_sync_retryable_exceptions_filter(self):
        """仅对指定异常类型重试。"""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("不可重试")

        with pytest.raises(ValueError):
            raise_value_error()

        assert call_count == 1

    def test_sync_on_retry_callback(self):
        """验证 on_retry 回调被正确调用。"""
        retries = []

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            on_retry=lambda attempt, exc, delay: retries.append((attempt, str(exc))),
        )
        def fail_twice():
            if len(retries) < 2:
                raise RuntimeError(f"失败 {len(retries) + 1}")
            return "ok"

        result = fail_twice()
        assert result == "ok"
        assert len(retries) == 2
        assert retries[0][0] == 1
        assert retries[1][0] == 2

    @pytest.mark.asyncio
    async def test_async_retry_then_success(self):
        """异步函数的重试。"""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def async_fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("异步失败")
            return "async_ok"

        result = await async_fail_then_succeed()
        assert result == "async_ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_exhaust_retries(self):
        """异步函数重试耗尽。"""

        @retry_with_backoff(max_retries=1, base_delay=0.01)
        async def async_always_fail():
            raise TimeoutError("超时")

        with pytest.raises(TimeoutError):
            await async_always_fail()

    def test_backoff_timing(self):
        """验证退避时间大致遵循指数增长。"""
        timestamps = []

        @retry_with_backoff(max_retries=2, base_delay=0.05, backoff_factor=2.0)
        def timed_fail():
            timestamps.append(time.time())
            raise RuntimeError("定时失败")

        with pytest.raises(RuntimeError):
            timed_fail()

        assert len(timestamps) == 3
        delay_1 = timestamps[1] - timestamps[0]
        delay_2 = timestamps[2] - timestamps[1]
        # base_delay=0.05, 第1次退避 ~0.05s, 第2次退避 ~0.10s
        assert delay_1 >= 0.04
        assert delay_2 >= 0.08


# ════════════════════════════════════════════════════════════
#  timeout_call 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTimeoutCall:
    """测试超时控制机制。"""

    @pytest.mark.asyncio
    async def test_async_timeout_success(self):
        """在超时时间内完成的异步函数。"""

        @timeout_call(timeout_seconds=1.0)
        async def fast_func():
            return "快速完成"

        result = await fast_func()
        assert result == "快速完成"

    @pytest.mark.asyncio
    async def test_async_timeout_exceeded(self):
        """超时的异步函数。"""

        @timeout_call(timeout_seconds=0.1)
        async def slow_func():
            await asyncio.sleep(1.0)
            return "不会到达"

        with pytest.raises(TimeoutError):
            await slow_func()

    def test_sync_timeout_success(self):
        """同步超时控制 - 正常完成。"""

        def fast():
            return 42

        result = sync_timeout_call(fast, 1.0)
        assert result == 42

    def test_sync_timeout_exceeded(self):
        """同步超时控制 - 超时。"""

        def slow():
            time.sleep(2.0)
            return "不会到达"

        with pytest.raises(TimeoutError):
            sync_timeout_call(slow, 0.1)

    def test_sync_timeout_propagates_exception(self):
        """同步超时控制 - 传播内部异常。"""

        def raise_error():
            raise ValueError("内部错误")

        with pytest.raises(ValueError, match="内部错误"):
            sync_timeout_call(raise_error, 1.0)


# ════════════════════════════════════════════════════════════
#  CircuitBreaker 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCircuitBreaker:
    """测试 CircuitBreaker 的熔断和恢复逻辑。"""

    def test_initial_state_closed(self):
        """初始状态为 CLOSED。"""
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=0.1)
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold(self):
        """连续失败达到阈值后进入 OPEN 状态。"""
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)

        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.allow_request() is False

    def test_success_resets_count(self):
        """成功调用重置失败计数。"""
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()

        # 成功后重置，所以只有 2 次连续失败
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_after_recovery_timeout(self):
        """冷却期过后转入 HALF_OPEN 状态。"""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitBreaker.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_success_closes(self):
        """HALF_OPEN 下成功调用恢复到 CLOSED。"""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_failure_reopens(self):
        """HALF_OPEN 下失败重新进入 OPEN。"""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)

        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_multiple_breakers_independent(self):
        """不同实例互不影响。"""
        cb1 = CircuitBreaker("service_a", failure_threshold=2, recovery_timeout=10)
        cb2 = CircuitBreaker("service_b", failure_threshold=2, recovery_timeout=10)

        cb1.record_failure()
        cb1.record_failure()
        assert cb1.state == CircuitBreaker.OPEN
        assert cb2.state == CircuitBreaker.CLOSED


# ════════════════════════════════════════════════════════════
#  IdempotencyGuard 测试
# ════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIdempotencyGuard:
    """测试 IdempotencyGuard 的幂等保护。"""

    def test_first_call_not_duplicate(self):
        """首次操作不视为重复。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)

        is_dup, result = guard.check_and_set("trace-1", "step-1", result="first")
        assert is_dup is False
        assert result == "first"

    def test_second_call_is_duplicate(self):
        """相同 trace_id + step 的二次操作视为重复。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)

        guard.check_and_set("trace-1", "step-1", result="original")
        is_dup, result = guard.check_and_set("trace-1", "step-1", result="should_be_ignored")

        assert is_dup is True
        assert result == "original"

    def test_different_trace_not_duplicate(self):
        """不同 trace_id 的操作不视为重复。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)

        guard.check_and_set("trace-1", "step-1", result="r1")
        is_dup, result = guard.check_and_set("trace-2", "step-1", result="r2")

        assert is_dup is False
        assert result == "r2"

    def test_different_step_not_duplicate(self):
        """相同 trace_id 但不同 step 不视为重复。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)

        guard.check_and_set("trace-1", "step-1", result="r1")
        is_dup, result = guard.check_and_set("trace-1", "step-2", result="r2")

        assert is_dup is False
        assert result == "r2"

    def test_params_hash_differentiation(self):
        """params_hash 不同时不视为重复。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)

        guard.check_and_set("trace-1", "step-1", params_hash="abc", result="r1")
        is_dup, result = guard.check_and_set("trace-1", "step-1", params_hash="xyz", result="r2")

        assert is_dup is False
        assert result == "r2"

    def test_ttl_expiration(self):
        """TTL 过期后缓存失效。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=0.1)

        guard.check_and_set("trace-1", "step-1", result="old")
        time.sleep(0.15)
        is_dup, result = guard.check_and_set("trace-1", "step-1", result="new")

        assert is_dup is False
        assert result == "new"

    def test_max_size_eviction(self):
        """超过 max_size 后最早的缓存被淘汰。"""
        guard = IdempotencyGuard(max_size=3, ttl_seconds=60)

        guard.check_and_set("t1", "s1", result="r1")
        guard.check_and_set("t2", "s1", result="r2")
        guard.check_and_set("t3", "s1", result="r3")
        guard.check_and_set("t4", "s1", result="r4")

        # t1 应该被淘汰
        is_dup, _ = guard.check_and_set("t1", "s1", result="r1_new")
        assert is_dup is False

        # t4 应该还在
        is_dup, result = guard.check_and_set("t4", "s1", result="ignored")
        assert is_dup is True
        assert result == "r4"

    def test_invalidate(self):
        """手动使缓存失效。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)

        guard.check_and_set("trace-1", "step-1", result="original")
        guard.invalidate("trace-1", "step-1")

        is_dup, result = guard.check_and_set("trace-1", "step-1", result="after_invalidate")
        assert is_dup is False
        assert result == "after_invalidate"

    def test_invalidate_nonexistent(self):
        """使不存在的缓存失效不报错。"""
        guard = IdempotencyGuard(max_size=100, ttl_seconds=60)
        guard.invalidate("nonexistent", "step")  # 不应抛异常

"""
app/core/reliability.py
调度可靠性模块 —— 统一的超时、重试、退避与幂等策略。

为 LLM 调用与工具调用（ES/Milvus/Redis）提供一致的容错能力:
    - 超时控制: 可配置的调用超时时间
    - 重试策略: 指数退避重试，支持自定义重试条件
    - 幂等保护: 基于 trace_id + step 的去重，防止重复写入/重复调用
    - 熔断器: 错误率超阈值时短路调用，快速失败
"""

import asyncio
import functools
import hashlib
import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, TypeVar

from app.core.logger import get_logger

_logger = get_logger(agent_name="Reliability")

T = TypeVar("T")

# ─────────────────────────── 重试装饰器 ───────────────────────────


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    带指数退避的重试装饰器（支持同步和异步函数）。

    Args:
        max_retries: 最大重试次数。
        base_delay: 初始延迟（秒）。
        max_delay: 最大延迟（秒）。
        backoff_factor: 退避倍数。
        retryable_exceptions: 可重试的异常类型。
        on_retry: 每次重试时的回调函数 (attempt, exception, delay)。
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception: Exception = RuntimeError("未知重试错误")
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    _logger.warning(
                        "重试 {}/{} | func={} | delay={:.1f}s | error={}",
                        attempt + 1, max_retries, func.__name__, delay, str(e),
                    )
                    await asyncio.sleep(delay)
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception: Exception = RuntimeError("未知重试错误")
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    _logger.warning(
                        "重试 {}/{} | func={} | delay={:.1f}s | error={}",
                        attempt + 1, max_retries, func.__name__, delay, str(e),
                    )
                    time.sleep(delay)
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ─────────────────────────── 超时控制 ───────────────────────────


def timeout_call(timeout_seconds: float):
    """
    超时控制装饰器（仅支持异步函数）。
    超时后抛出 asyncio.TimeoutError。
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                _logger.error(
                    "调用超时 | func={} | timeout={}s",
                    func.__name__, timeout_seconds,
                )
                raise TimeoutError(
                    f"{func.__name__} 调用超时（{timeout_seconds}s）"
                ) from exc
        return wrapper
    return decorator


def sync_timeout_call(func: Callable[..., T], timeout_seconds: float, *args, **kwargs) -> T:
    """
    同步函数的超时控制（基于线程实现）。
    """
    result: list[Any] = [None]
    exception: list[BaseException | None] = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"{func.__name__} 调用超时（{timeout_seconds}s）")
    exc = exception[0]
    if exc is not None:
        raise exc
    return result[0]


# ─────────────────────────── 幂等保护 ───────────────────────────


class IdempotencyGuard:
    """
    幂等保护器 —— 基于 trace_id + step 的去重。

    使用 LRU 缓存存储已执行的操作结果，避免重复写入/重复扣费。
    线程安全。
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _make_key(self, trace_id: str, step: str, params_hash: str = "") -> str:
        raw = f"{trace_id}:{step}:{params_hash}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _evict_expired(self):
        now = time.time()
        expired_keys = [
            k for k, (_, ts) in self._cache.items()
            if now - ts > self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]

    def check_and_set(
        self,
        trace_id: str,
        step: str,
        params_hash: str = "",
        result: Any = None,
    ) -> tuple[bool, Any]:
        """
        检查操作是否已执行过。

        Args:
            trace_id: 请求链路 ID。
            step: 操作步骤标识。
            params_hash: 参数哈希（可选，用于更精确的去重）。
            result: 本次操作结果（仅在首次调用时缓存）。

        Returns:
            (is_duplicate, cached_result):
                - is_duplicate=True 表示重复操作，返回缓存结果
                - is_duplicate=False 表示首次操作，结果已缓存
        """
        key = self._make_key(trace_id, step, params_hash)

        with self._lock:
            self._evict_expired()

            if key in self._cache:
                cached_result, _ = self._cache[key]
                self._cache.move_to_end(key)
                _logger.debug("幂等命中 | trace_id={} | step={}", trace_id, step)
                return True, cached_result

            self._cache[key] = (result, time.time())

            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

            return False, result

    def invalidate(self, trace_id: str, step: str, params_hash: str = ""):
        """手动使某条缓存失效。"""
        key = self._make_key(trace_id, step, params_hash)
        with self._lock:
            self._cache.pop(key, None)


# 全局幂等保护器
idempotency_guard = IdempotencyGuard()


# ─────────────────────────── 熔断器 ───────────────────────────


class CircuitBreaker:
    """
    简易熔断器。

    状态机: CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN
        - CLOSED: 正常通过请求
        - OPEN: 快速失败（错误率超过阈值后触发）
        - HALF_OPEN: 冷却后允许试探性通过一个请求
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        """检查是否允许通过请求。"""
        current = self.state
        if current == self.CLOSED:
            return True
        if current == self.HALF_OPEN:
            return True
        _logger.warning("熔断器 [{}] 开启，拒绝请求", self.name)
        return False

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = self.OPEN
                _logger.error(
                    "熔断器 [{}] 触发 | failures={}/{}",
                    self.name, self._failure_count, self._failure_threshold,
                )


# 预置各服务的熔断器
es_circuit_breaker = CircuitBreaker("elasticsearch", failure_threshold=5, recovery_timeout=30)
milvus_circuit_breaker = CircuitBreaker("milvus", failure_threshold=5, recovery_timeout=30)
llm_circuit_breaker = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60)

"""
app/core/cache.py
缓存层 —— 基于 Redis 的 Embedding 向量缓存与用户画像缓存。

设计思路:
    - Embedding 缓存: 相同文本的向量结果缓存 24h，避免重复调用 Embedding API
    - 画像缓存: 基于 user_id 缓存 MySQL 画像数据，TTL 5 分钟
    - 缓存命中率统计: 线程安全的命中/未命中计数器，供 MonitorAgent 上报

缓存 key 规范:
    - Embedding: cache:emb:<md5(text)>
    - 画像:     cache:profile:<user_id>
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Optional

import redis

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="Cache")

_EMBEDDING_TTL = 86400      # 24 小时
_PROFILE_TTL = 300           # 5 分钟

_KEY_PREFIX_EMBEDDING = "cache:emb:"
_KEY_PREFIX_PROFILE = "cache:profile:"

# ── Redis 客户端（单例，线程安全） ──────────────────────────

_redis_client: redis.Redis | None = None
_redis_lock = threading.Lock()
_redis_available = True      # 首次连接失败后标记，避免反复重试


def _get_redis() -> redis.Redis | None:
    """
    获取 Redis 客户端单例。

    如果 Redis 不可达，静默降级（返回 None），不影响主流程。
    """
    global _redis_client, _redis_available

    if not _redis_available:
        return None

    if _redis_client is not None:
        return _redis_client

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client

        try:
            r = redis.Redis(
                host=settings.redis.redis_host,
                port=settings.redis.redis_port,
                db=settings.redis.redis_db,
                password=settings.redis.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
            )
            r.ping()
            _redis_client = r
            _logger.info("缓存层 Redis 连接成功")
            return _redis_client
        except Exception as e:
            _redis_available = False
            _logger.warning("缓存层 Redis 不可达，降级为无缓存模式 | error={}", str(e))
            return None


# ── 缓存命中率统计 ──────────────────────────────────────

class CacheStats:
    """线程安全的缓存命中率统计器。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def record_hit(self) -> None:
        with self._lock:
            self._hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self._misses += 1

    def get_stats(self) -> dict[str, float | int]:
        """返回命中率统计快照。"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0


embedding_cache_stats = CacheStats()
profile_cache_stats = CacheStats()


# ── Embedding 缓存 ──────────────────────────────────────

def _embedding_key(text: str) -> str:
    """基于文本 MD5 构建缓存 key，避免过长 key。"""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX_EMBEDDING}{digest}"


def get_cached_embedding(text: str) -> Optional[list[float]]:
    """
    从 Redis 读取已缓存的 Embedding 向量。

    Returns:
        向量列表，缓存未命中或 Redis 不可用时返回 None。
    """
    r = _get_redis()
    if r is None:
        return None

    try:
        raw = r.get(_embedding_key(text))
        if raw is not None:
            embedding_cache_stats.record_hit()
            return json.loads(raw)
        embedding_cache_stats.record_miss()
        return None
    except Exception as e:
        _logger.debug("Embedding 缓存读取失败 | error={}", str(e))
        embedding_cache_stats.record_miss()
        return None


def set_cached_embedding(text: str, vector: list[float]) -> None:
    """
    将 Embedding 向量写入 Redis 缓存。

    失败时静默降级，不影响主流程。
    """
    r = _get_redis()
    if r is None:
        return

    try:
        r.setex(
            _embedding_key(text),
            _EMBEDDING_TTL,
            json.dumps(vector),
        )
    except Exception as e:
        _logger.debug("Embedding 缓存写入失败 | error={}", str(e))


# ── 用户画像缓存 ──────────────────────────────────────

def _profile_key(user_id: str) -> str:
    return f"{_KEY_PREFIX_PROFILE}{user_id}"


def get_cached_profile(user_id: str) -> Optional[dict]:
    """
    从 Redis 读取已缓存的用户画像。

    Returns:
        画像字典，缓存未命中或 Redis 不可用时返回 None。
    """
    r = _get_redis()
    if r is None:
        return None

    try:
        raw = r.get(_profile_key(user_id))
        if raw is not None:
            profile_cache_stats.record_hit()
            return json.loads(raw)
        profile_cache_stats.record_miss()
        return None
    except Exception as e:
        _logger.debug("画像缓存读取失败 | error={}", str(e))
        profile_cache_stats.record_miss()
        return None


def set_cached_profile(user_id: str, profile: dict) -> None:
    """
    将用户画像写入 Redis 缓存。

    对 Decimal/datetime 等不可序列化类型做安全转换。
    失败时静默降级，不影响主流程。
    """
    r = _get_redis()
    if r is None:
        return

    try:
        r.setex(
            _profile_key(user_id),
            _PROFILE_TTL,
            json.dumps(profile, ensure_ascii=False, default=str),
        )
    except Exception as e:
        _logger.debug("画像缓存写入失败 | error={}", str(e))


def invalidate_profile(user_id: str) -> None:
    """主动失效指定用户的画像缓存（用于用户数据变更后）。"""
    r = _get_redis()
    if r is None:
        return

    try:
        r.delete(_profile_key(user_id))
    except Exception:
        pass


# ── 聚合统计 ────────────────────────────────────────────

def get_all_cache_stats() -> dict[str, dict]:
    """
    返回所有缓存类型的命中率统计。

    MonitorAgent 通过此函数获取数据并上报到 Langfuse。
    """
    return {
        "embedding": embedding_cache_stats.get_stats(),
        "profile": profile_cache_stats.get_stats(),
    }

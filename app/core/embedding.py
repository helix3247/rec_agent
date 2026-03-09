"""
app/core/embedding.py
统一的 Embedding 客户端 —— 提供线程安全的单例 OpenAI 客户端和向量获取接口。

消除 search.py / knowledge.py / memory.py 中完全重复的 _get_embedding() 实现，
同时复用 OpenAI 客户端避免每次调用都新建连接。
集成 Redis 缓存：相同文本的向量结果缓存 24h，避免重复调用 Embedding API。
"""

from __future__ import annotations

import threading

from openai import OpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.core.cache import get_cached_embedding, set_cached_embedding

_logger = get_logger(agent_name="Embedding")

_client: OpenAI | None = None
_client_lock = threading.Lock()


def _get_client() -> OpenAI:
    """获取全局单例 OpenAI 客户端（线程安全）。"""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        emb_cfg = settings.embedding
        _client = OpenAI(
            api_key=emb_cfg.embedding_api_key or "dummy",
            base_url=emb_cfg.embedding_base_url,
        )
        return _client


def get_embedding(text: str) -> list[float]:
    """
    调用 Embedding 模型获取文本的向量表示。

    优先从 Redis 缓存读取，缓存未命中时调用 API 并回写缓存（TTL 24h）。
    缓存层不可用时透明降级，不影响功能。
    """
    cached = get_cached_embedding(text)
    if cached is not None:
        _logger.debug("Embedding 缓存命中 | text_len={}", len(text))
        return cached

    client = _get_client()
    emb_cfg = settings.embedding
    response = client.embeddings.create(
        model=emb_cfg.embedding_model,
        input=[text],
    )
    vector = response.data[0].embedding

    set_cached_embedding(text, vector)
    return vector

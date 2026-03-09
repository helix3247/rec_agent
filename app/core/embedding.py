"""
app/core/embedding.py
统一的 Embedding 客户端 —— 提供线程安全的单例 OpenAI 客户端和向量获取接口。

消除 search.py / knowledge.py / memory.py 中完全重复的 _get_embedding() 实现，
同时复用 OpenAI 客户端避免每次调用都新建连接。
"""

import threading

from openai import OpenAI

from app.core.config import settings
from app.core.logger import get_logger

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

    使用全局单例客户端，避免每次调用创建新连接。
    """
    client = _get_client()
    emb_cfg = settings.embedding
    response = client.embeddings.create(
        model=emb_cfg.embedding_model,
        input=[text],
    )
    return response.data[0].embedding

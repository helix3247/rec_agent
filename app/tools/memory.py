"""
app/tools/memory.py
长期记忆管理 —— 实现 Redis 短期记忆到 Milvus 长期记忆的迁移策略。

迁移策略:
    - 会话结束（显式结束或 Redis TTL 即将过期）时，将对话摘要向量化后写入 Milvus，关联 user_id。
    - 新会话开启时，从 Milvus 检索该用户近期对话摘要，注入上下文实现跨会话记忆。

Milvus 集合: user_memory（独立于知识库 collection）
"""

import json
import time
from typing import Optional

from openai import OpenAI
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.core.config import settings
from app.core.llm import get_llm
from app.core.logger import get_logger

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)

_logger = get_logger(agent_name="MemoryTool")
_MEMORY_COLLECTION = "user_memory"
_MEMORY_VECTOR_DIM = 3072
_MAX_SUMMARY_LEN = 800
_milvus_connected = False

_SUMMARY_PROMPT = """你是一个对话摘要助手。请将以下用户与AI助手的对话精炼为一段简洁的摘要，
重点保留用户的偏好、需求、购物意向和反馈信息。摘要应便于在未来的对话中回忆用户的个人偏好。

要求：
- 保留用户明确表达的偏好（如品牌、风格、预算、场景等）
- 保留用户对推荐结果的反馈（满意/不满意、原因）
- 忽略纯寒暄和无实质内容的对话
- 摘要长度不超过300字
- 使用第三人称描述用户

对话记录：
{conversation}
"""


def _ensure_milvus_connection():
    """确保 Milvus 连接已建立。"""
    global _milvus_connected
    if not _milvus_connected:
        milvus_cfg = settings.milvus
        connections.connect(
            alias="default",
            host=milvus_cfg.milvus_host,
            port=str(milvus_cfg.milvus_port),
        )
        _milvus_connected = True


def _ensure_memory_collection() -> Collection:
    """确保长期记忆 Milvus 集合存在，不存在则创建。"""
    _ensure_milvus_connection()

    if not utility.has_collection(_MEMORY_COLLECTION):
        fields = [
            FieldSchema(name="memory_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="thread_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="preferences", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_MEMORY_VECTOR_DIM),
        ]
        schema = CollectionSchema(fields=fields, description="用户长期对话记忆")
        collection = Collection(name=_MEMORY_COLLECTION, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        _logger.info("创建长期记忆集合 | collection={}", _MEMORY_COLLECTION)
    else:
        collection = Collection(_MEMORY_COLLECTION)

    return collection


def _get_embedding(text: str) -> list[float]:
    """调用 Embedding 模型获取向量表示。"""
    emb_cfg = settings.embedding
    client = OpenAI(
        api_key=emb_cfg.embedding_api_key or "dummy",
        base_url=emb_cfg.embedding_base_url,
    )
    response = client.embeddings.create(
        model=emb_cfg.embedding_model,
        input=[text],
    )
    return response.data[0].embedding


def _summarize_conversation(messages: list[BaseMessage]) -> str:
    """使用 LLM 生成对话摘要。"""
    if not messages:
        return ""

    conversation_text = ""
    for msg in messages:
        role = "用户" if getattr(msg, "type", "") == "human" else "AI助手"
        conversation_text += f"{role}: {msg.content}\n"

    if len(conversation_text) < 20:
        return ""

    try:
        llm = get_llm("primary", temperature=0.3)
        prompt = _SUMMARY_PROMPT.format(conversation=conversation_text[:3000])
        response = llm.invoke([SystemMessage(content=prompt)])
        summary = response.content.strip()
        return summary[:_MAX_SUMMARY_LEN]
    except Exception as e:
        _logger.warning("对话摘要生成失败(primary)，尝试 fallback | error={}", str(e))
        try:
            llm = get_llm("fallback", temperature=0.3)
            prompt = _SUMMARY_PROMPT.format(conversation=conversation_text[:3000])
            response = llm.invoke([SystemMessage(content=prompt)])
            summary = response.content.strip()
            return summary[:_MAX_SUMMARY_LEN]
        except Exception:
            _logger.error("对话摘要生成完全失败")
            return _fallback_summary(messages)


def _fallback_summary(messages: list[BaseMessage]) -> str:
    """LLM 不可用时的降级摘要：提取用户消息关键内容。"""
    user_msgs = [
        msg.content for msg in messages
        if getattr(msg, "type", "") == "human"
    ]
    if not user_msgs:
        return ""
    return "用户曾询问: " + "; ".join(user_msgs[-5:])


def _extract_preferences(summary: str) -> str:
    """从摘要中提取结构化偏好（简单实现，可后续用 LLM 增强）。"""
    preference_keywords = {
        "brand": ["品牌", "牌子"],
        "style": ["风格", "款式", "穿搭"],
        "budget": ["预算", "价格", "元", "块钱"],
        "scenario": ["场景", "用途", "通勤", "约会", "休闲", "运动"],
        "category": ["品类", "类目", "相机", "手机", "衣服", "鞋"],
    }

    found = {}
    for key, keywords in preference_keywords.items():
        for kw in keywords:
            if kw in summary:
                found[key] = kw
                break

    return json.dumps(found, ensure_ascii=False) if found else "{}"


def migrate_to_long_term(
    user_id: str,
    thread_id: str,
    messages: list[BaseMessage],
) -> bool:
    """
    将会话对话迁移到 Milvus 长期记忆。

    Args:
        user_id: 用户 ID。
        thread_id: 会话 ID。
        messages: 该会话的完整对话历史。

    Returns:
        是否迁移成功。
    """
    if not user_id or not messages:
        _logger.debug("跳过记忆迁移 | user_id={} | messages_count={}", user_id, len(messages) if messages else 0)
        return False

    user_messages = [m for m in messages if getattr(m, "type", "") == "human"]
    if len(user_messages) < 2:
        _logger.debug("对话轮次不足，跳过迁移 | user_messages={}", len(user_messages))
        return False

    try:
        summary = _summarize_conversation(messages)
        if not summary:
            return False

        preferences = _extract_preferences(summary)
        embedding = _get_embedding(summary)

        collection = _ensure_memory_collection()
        collection.insert([
            [user_id],
            [thread_id],
            [summary],
            [preferences],
            [int(time.time())],
            [embedding],
        ])
        collection.flush()

        _logger.info(
            "记忆迁移成功 | user_id={} | thread_id={} | summary_len={}",
            user_id, thread_id, len(summary),
        )
        return True

    except Exception as e:
        _logger.error("记忆迁移失败 | user_id={} | error={}", user_id, str(e))
        return False


def recall_long_term_memory(
    user_id: str,
    query: Optional[str] = None,
    top_k: int = 3,
) -> list[dict]:
    """
    从 Milvus 检索用户的长期记忆。

    Args:
        user_id: 用户 ID。
        query: 可选的查询文本，用于语义相似检索；为空则按时间倒序获取最近记忆。
        top_k: 返回数量。

    Returns:
        记忆列表，每条包含 summary, preferences, timestamp, score。
    """
    if not user_id:
        return []

    try:
        collection = _ensure_memory_collection()
        collection.load()
    except Exception as e:
        _logger.warning("长期记忆集合加载失败 | error={}", str(e))
        return []

    search_text = query or f"用户 {user_id} 的偏好和历史"

    try:
        query_vector = _get_embedding(search_text)
    except Exception as e:
        _logger.warning("记忆检索 Embedding 失败 | error={}", str(e))
        return []

    expr = f'user_id == "{user_id}"'
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 16},
    }

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["summary", "preferences", "timestamp", "thread_id"],
        )
    except Exception as e:
        _logger.warning("长期记忆检索失败 | error={}", str(e))
        return []

    memories = []
    for hit in results[0]:
        memories.append({
            "summary": hit.entity.get("summary", ""),
            "preferences": hit.entity.get("preferences", "{}"),
            "timestamp": hit.entity.get("timestamp", 0),
            "thread_id": hit.entity.get("thread_id", ""),
            "score": round(hit.score, 4),
        })

    _logger.info("长期记忆检索完成 | user_id={} | hits={}", user_id, len(memories))
    return memories


def format_memory_context(memories: list[dict]) -> str:
    """将检索到的长期记忆格式化为可注入 Prompt 的上下文文本。"""
    if not memories:
        return ""

    lines = ["[用户历史偏好与记忆]"]
    for i, mem in enumerate(memories, 1):
        ts = mem.get("timestamp", 0)
        time_str = time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "未知时间"
        lines.append(f"{i}. ({time_str}) {mem['summary']}")

    return "\n".join(lines)

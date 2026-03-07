"""
app/tools/knowledge.py
知识库检索工具 —— 封装 Milvus 向量检索逻辑，支持按 product_id 和 doc_type 过滤。
"""

from typing import Optional

from openai import OpenAI
from pymilvus import Collection, connections

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="KnowledgeTool")

_connected = False


def _ensure_connection():
    """确保 Milvus 连接已建立。"""
    global _connected
    if not _connected:
        milvus_cfg = settings.milvus
        connections.connect(
            alias="default",
            host=milvus_cfg.milvus_host,
            port=str(milvus_cfg.milvus_port),
        )
        _connected = True


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


def query_knowledge(
    query: str,
    product_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """
    从 Milvus 知识库中检索与 query 语义相关的文档 chunk。

    Args:
        query: 用户查询文本。
        product_id: 按商品 ID 过滤（精确匹配）。
        doc_type: 按文档类型过滤（review / faq / manual）。
        top_k: 返回数量。

    Returns:
        命中的文档列表，每个元素包含 product_id, doc_type, text, score。
    """
    _ensure_connection()
    collection_name = settings.milvus.milvus_collection

    try:
        collection = Collection(collection_name)
        collection.load()
    except Exception as e:
        _logger.error("Milvus 集合加载失败 | collection={} | error={}", collection_name, str(e))
        return []

    try:
        query_vector = _get_embedding(query)
    except Exception as e:
        _logger.error("知识库检索 Embedding 失败 | error={}", str(e))
        return []

    # 构建过滤表达式
    filters = []
    if product_id:
        filters.append(f'product_id == "{product_id}"')
    if doc_type:
        filters.append(f'doc_type == "{doc_type}"')
    expr = " and ".join(filters) if filters else ""

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
            expr=expr or None,
            output_fields=["product_id", "doc_type", "text"],
        )
        _logger.info("Milvus 检索完成 | hits={} | expr={}", len(results[0]), expr or "无")
    except Exception as e:
        _logger.error("Milvus 检索失败 | error={}", str(e))
        return []

    chunks = []
    for hit in results[0]:
        chunks.append({
            "product_id": hit.entity.get("product_id", ""),
            "doc_type": hit.entity.get("doc_type", ""),
            "text": hit.entity.get("text", ""),
            "score": round(hit.score, 4),
        })

    return chunks

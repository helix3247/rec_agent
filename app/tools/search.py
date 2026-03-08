"""
app/tools/search.py
商品检索工具 —— 封装 Elasticsearch Hybrid Search（关键词 + KNN 向量）与结构化过滤。
集成可靠性机制：超时控制、熔断保护、重试。
"""

import time
from typing import Optional

from openai import OpenAI
from elasticsearch import Elasticsearch

from app.core.config import settings
from app.core.logger import get_logger
from app.core.reliability import (
    es_circuit_breaker,
    retry_with_backoff,
)

_logger = get_logger(agent_name="SearchTool")


def _get_es_client() -> Elasticsearch:
    """获取 Elasticsearch 客户端实例。"""
    es_cfg = settings.es
    es_kwargs: dict = {"hosts": es_cfg.es_host}

    if es_cfg.es_host.startswith("https"):
        es_kwargs["verify_certs"] = False
        es_kwargs["ssl_show_warn"] = False

    if es_cfg.es_username and es_cfg.es_password:
        es_kwargs["basic_auth"] = (es_cfg.es_username, es_cfg.es_password)

    return Elasticsearch(**es_kwargs)


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


def search_products(
    query: str,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    tags: Optional[list[str]] = None,
    top_k: int = 10,
    use_vector: bool = True,
) -> list[dict]:
    """
    ES Hybrid Search：结合关键词匹配、KNN 向量检索和结构化过滤。

    Args:
        query: 用户查询文本。
        category: 品类过滤（精确匹配）。
        brand: 品牌过滤（精确匹配）。
        min_price: 最低价格。
        max_price: 最高价格。
        tags: 标签过滤（任一匹配）。
        top_k: 返回数量。
        use_vector: 是否启用向量检索。

    Returns:
        命中商品列表，每个元素包含 product_id, name, category, brand, price, tags, description, score。
    """
    es = _get_es_client()
    index_name = settings.es.es_index_name

    # ── 构建 bool query（关键词 + 结构化过滤） ──
    must_clauses = []
    filter_clauses = []

    if query:
        must_clauses.append({
            "multi_match": {
                "query": query,
                "fields": ["name^3", "description", "tags^2", "brand^2", "category^2"],
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        })

    if category:
        filter_clauses.append({"term": {"category": category}})
    if brand:
        filter_clauses.append({"term": {"brand": brand}})
    if min_price is not None or max_price is not None:
        price_range: dict = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        filter_clauses.append({"range": {"price": price_range}})
    if tags:
        filter_clauses.append({"terms": {"tags": tags}})

    body: dict = {"size": top_k}

    bool_query: dict = {}
    if must_clauses:
        bool_query["must"] = must_clauses
    if filter_clauses:
        bool_query["filter"] = filter_clauses

    if bool_query:
        body["query"] = {"bool": bool_query}

    # ── KNN 向量检索（Hybrid） ──
    if use_vector and query:
        try:
            query_vector = _get_embedding(query)
            body["knn"] = {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 5,
            }
            if filter_clauses:
                body["knn"]["filter"] = {"bool": {"filter": filter_clauses}}
        except Exception as e:
            _logger.warning("向量检索 Embedding 失败，仅使用关键词检索 | error={}", str(e))

    if not es_circuit_breaker.allow_request():
        _logger.warning("ES 熔断器开启，跳过检索")
        return []

    try:
        t0 = time.time()
        resp = es.search(index=index_name, body=body, request_timeout=15)
        elapsed = round((time.time() - t0) * 1000)
        es_circuit_breaker.record_success()
        _logger.info("ES 检索完成 | hits={} | took={}ms", resp["hits"]["total"]["value"], elapsed)
    except Exception as e:
        es_circuit_breaker.record_failure()
        _logger.error("ES 检索失败 | error={}", str(e))
        return []

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "product_id": src.get("product_id", hit["_id"]),
            "name": src.get("name", ""),
            "category": src.get("category", ""),
            "brand": src.get("brand", ""),
            "price": src.get("price", 0),
            "tags": src.get("tags", []),
            "description": src.get("description", ""),
            "score": hit.get("_score", 0),
        })

    return results

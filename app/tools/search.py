"""
app/tools/search.py
商品检索工具 —— 封装 Elasticsearch Hybrid Search（BM25 + KNN 向量 + 显式 RRF 融合）
与结构化过滤。集成 IK 中文分词、同义词扩展、可靠性机制。
"""

import threading
import time
from typing import Optional

from elasticsearch import (
    ConnectionError as ESConnectionError,
    ConnectionTimeout,
    Elasticsearch,
)

from app.core.config import settings
from app.core.embedding import get_embedding
from app.core.logger import get_logger
from app.core.reliability import (
    es_circuit_breaker,
    retry_with_backoff,
)

_TRANSIENT_ES_EXCEPTIONS = (ESConnectionError, ConnectionTimeout, TimeoutError, OSError)

_logger = get_logger(agent_name="SearchTool")

_es_client: Elasticsearch | None = None
_es_client_lock = threading.Lock()

# ── RRF 融合参数（可通过环境变量覆盖） ──
RRF_RANK_CONSTANT = 60
RRF_BM25_WEIGHT = 0.4
RRF_KNN_WEIGHT = 0.6

# ── IK 分词 Index Mapping ──
PRODUCT_INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "filter": {
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms_path": "analysis/synonym.txt",
                    "updateable": True,
                }
            },
            "analyzer": {
                "ik_max_index": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase"],
                },
                "ik_smart_search": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase", "synonym_filter"],
                },
            },
        }
    },
    "mappings": {
        "properties": {
            "product_id": {"type": "keyword"},
            "name": {
                "type": "text",
                "analyzer": "ik_max_index",
                "search_analyzer": "ik_smart_search",
            },
            "category": {"type": "keyword"},
            "brand": {"type": "keyword"},
            "price": {"type": "float"},
            "tags": {"type": "keyword"},
            "description": {
                "type": "text",
                "analyzer": "ik_max_index",
                "search_analyzer": "ik_smart_search",
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 3072,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def _get_es_client() -> Elasticsearch:
    """获取全局单例 Elasticsearch 客户端（线程安全，复用连接池）。"""
    global _es_client
    if _es_client is not None:
        return _es_client
    with _es_client_lock:
        if _es_client is not None:
            return _es_client
        es_cfg = settings.es
        es_kwargs: dict = {"hosts": es_cfg.es_host}

        if es_cfg.es_host.startswith("https"):
            es_kwargs["verify_certs"] = False
            es_kwargs["ssl_show_warn"] = False

        if es_cfg.es_username and es_cfg.es_password:
            es_kwargs["basic_auth"] = (es_cfg.es_username, es_cfg.es_password)

        _es_client = Elasticsearch(**es_kwargs)
        return _es_client


def ensure_index_with_ik() -> None:
    """确保 product_index 使用 IK 分词器 mapping；如已存在则跳过。"""
    es = _get_es_client()
    index_name = settings.es.es_index_name
    if es.indices.exists(index=index_name):
        _logger.info("索引 {} 已存在，跳过创建", index_name)
        return
    es.indices.create(index=index_name, body=PRODUCT_INDEX_SETTINGS)
    _logger.info("索引 {} 已创建（IK 分词 + 同义词）", index_name)


def reindex_with_ik(old_index: str | None = None) -> dict:
    """
    将旧索引数据迁移到使用 IK 分词器的新 mapping。

    流程：旧索引 → 创建新索引 → reindex → 切换别名 / 删除旧索引。
    """
    es = _get_es_client()
    src_index = old_index or settings.es.es_index_name
    tmp_index = f"{src_index}_ik_tmp"

    if not es.indices.exists(index=src_index):
        return {"status": "error", "message": f"源索引 {src_index} 不存在"}

    if es.indices.exists(index=tmp_index):
        es.indices.delete(index=tmp_index)

    es.indices.create(index=tmp_index, body=PRODUCT_INDEX_SETTINGS)

    resp = es.reindex(
        body={"source": {"index": src_index}, "dest": {"index": tmp_index}},
        wait_for_completion=True,
        timeout="300s",
    )
    created = resp.get("created", 0)
    _logger.info("Reindex 完成 | {} → {} | created={}", src_index, tmp_index, created)

    es.indices.delete(index=src_index)
    es.indices.create(index=src_index, body=PRODUCT_INDEX_SETTINGS)

    resp2 = es.reindex(
        body={"source": {"index": tmp_index}, "dest": {"index": src_index}},
        wait_for_completion=True,
        timeout="300s",
    )
    es.indices.delete(index=tmp_index)

    return {
        "status": "ok",
        "index": src_index,
        "docs_migrated": resp2.get("created", 0),
    }


def _build_filter_clauses(
    category: Optional[str],
    brand: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
    tags: Optional[list[str]],
) -> list[dict]:
    """构建 ES 结构化过滤条件。"""
    clauses: list[dict] = []
    if category:
        clauses.append({"term": {"category": category}})
    if brand:
        clauses.append({"term": {"brand": brand}})
    if min_price is not None or max_price is not None:
        price_range: dict = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        clauses.append({"range": {"price": price_range}})
    if tags:
        clauses.append({"terms": {"tags": tags}})
    return clauses


def _rrf_merge(
    bm25_hits: list[dict],
    knn_hits: list[dict],
    top_k: int,
    k: int = RRF_RANK_CONSTANT,
    bm25_weight: float = RRF_BM25_WEIGHT,
    knn_weight: float = RRF_KNN_WEIGHT,
) -> list[dict]:
    """
    显式 Reciprocal Rank Fusion 融合两路检索结果。

    RRF_score(d) = bm25_weight / (k + rank_bm25(d)) + knn_weight / (k + rank_knn(d))

    Args:
        bm25_hits: BM25 检索结果（已按分数排序）。
        knn_hits: KNN 向量检索结果（已按分数排序）。
        top_k: 最终返回数量。
        k: RRF 常数（默认 60，值越大长尾文档权重越高）。
        bm25_weight: BM25 路权重。
        knn_weight: KNN 路权重。
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        pid = hit["product_id"]
        scores[pid] = scores.get(pid, 0.0) + bm25_weight / (k + rank)
        docs[pid] = hit

    for rank, hit in enumerate(knn_hits, start=1):
        pid = hit["product_id"]
        scores[pid] = scores.get(pid, 0.0) + knn_weight / (k + rank)
        if pid not in docs:
            docs[pid] = hit

    sorted_pids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [{**docs[pid], "score": round(scores[pid], 6)} for pid in sorted_pids]


def _parse_hits(hits: list[dict]) -> list[dict]:
    """从 ES 原始 hits 提取标准化商品字典。"""
    results = []
    for hit in hits:
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


def _do_bm25_search(
    es: Elasticsearch,
    index_name: str,
    query: str,
    filter_clauses: list[dict],
    top_k: int,
) -> list[dict]:
    """执行纯 BM25 关键词检索。"""
    body: dict = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "name^3", "description", "tags^2",
                            "brand^2", "category^2",
                        ],
                        "type": "best_fields",
                    }
                }],
            }
        },
    }
    if filter_clauses:
        body["query"]["bool"]["filter"] = filter_clauses

    resp = es.search(index=index_name, body=body, timeout="15s")
    return _parse_hits(resp["hits"]["hits"])


def _do_knn_search(
    es: Elasticsearch,
    index_name: str,
    query_vector: list[float],
    filter_clauses: list[dict],
    top_k: int,
) -> list[dict]:
    """执行纯 KNN 向量检索。"""
    knn_body: dict = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": top_k * 5,
    }
    if filter_clauses:
        knn_body["filter"] = {"bool": {"filter": filter_clauses}}

    body: dict = {"size": top_k, "knn": knn_body}
    resp = es.search(index=index_name, body=body, timeout="15s")
    return _parse_hits(resp["hits"]["hits"])


def search_products(
    query: str,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    tags: Optional[list[str]] = None,
    top_k: int = 10,
    use_vector: bool = True,
    strategy: str = "hybrid_rrf",
) -> list[dict]:
    """
    商品检索入口，支持三种检索策略。

    Args:
        query: 用户查询文本。
        category: 品类过滤（精确匹配）。
        brand: 品牌过滤（精确匹配）。
        min_price: 最低价格。
        max_price: 最高价格。
        tags: 标签过滤（任一匹配）。
        top_k: 返回数量。
        use_vector: 是否启用向量检索（仅 hybrid_rrf 策略有效）。
        strategy: 检索策略，可选 "bm25" | "knn" | "hybrid_rrf"（默认）。

    Returns:
        命中商品列表，每个元素包含 product_id, name, category, brand, price, tags, description, score。
    """
    es = _get_es_client()
    index_name = settings.es.es_index_name
    filter_clauses = _build_filter_clauses(category, brand, min_price, max_price, tags)

    if not es_circuit_breaker.allow_request():
        _logger.warning("ES 熔断器开启，跳过检索")
        return []

    @retry_with_backoff(
        max_retries=1,
        base_delay=0.5,
        max_delay=2.0,
        retryable_exceptions=_TRANSIENT_ES_EXCEPTIONS,
    )
    def _execute():
        if strategy == "bm25":
            return _do_bm25_search(es, index_name, query, filter_clauses, top_k)

        if strategy == "knn":
            query_vector = get_embedding(query)
            return _do_knn_search(es, index_name, query_vector, filter_clauses, top_k)

        # hybrid_rrf: 两路独立检索 → 显式 RRF 融合
        bm25_results = _do_bm25_search(es, index_name, query, filter_clauses, top_k)

        if not use_vector:
            return bm25_results

        try:
            query_vector = get_embedding(query)
            knn_results = _do_knn_search(
                es, index_name, query_vector, filter_clauses, top_k,
            )
        except Exception as e:
            _logger.warning("向量检索失败，降级为纯 BM25 | error={}", str(e))
            return bm25_results

        merged = _rrf_merge(bm25_results, knn_results, top_k)
        _logger.info(
            "RRF 融合完成 | bm25_hits={} | knn_hits={} | merged={}",
            len(bm25_results), len(knn_results), len(merged),
        )
        return merged

    try:
        t0 = time.time()
        results = _execute()
        elapsed = round((time.time() - t0) * 1000)
        es_circuit_breaker.record_success()
        _logger.info(
            "ES 检索完成 | strategy={} | hits={} | took={}ms",
            strategy, len(results), elapsed,
        )
        return results
    except _TRANSIENT_ES_EXCEPTIONS as e:
        es_circuit_breaker.record_failure()
        _logger.error("ES 检索失败（瞬时故障，已重试） | error={}", str(e))
        return []
    except Exception as e:
        es_circuit_breaker.record_failure()
        _logger.error("ES 检索失败（非瞬时故障） | error={}", str(e))
        return []

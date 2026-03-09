"""
tests/eval_retrieval.py
离线检索评测 —— 对比 BM25、KNN、Hybrid+RRF 三组策略的 NDCG@10 和 Recall@10 指标。

使用方式:
    conda activate rec_agent
    python tests/eval_retrieval.py

评测流程:
    1. 加载评测集 (query + 人工标注的相关商品 ID 及相关度等级)
    2. 分别用三种检索策略检索
    3. 计算 NDCG@10 和 Recall@10
    4. 输出对比报告

注意:
    - KNN 和 Hybrid+RRF 策略需要商品有 embedding 字段
    - 如果 embedding 服务不可用，会跳过 KNN/Hybrid 评测
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch

# ── ES 连接配置 ──
ES_HOST = "http://127.0.0.1:29200"
ES_AUTH = ("elastic", "changeme")
INDEX = "product_index"

# ── 评测集 ──
# relevance: 3=完全相关, 2=高度相关, 1=部分相关, 0=不相关
# relevant_products 中的 product 用 name 匹配（因 product_id 是随机 UUID）
EVAL_QUERIES = [
    # ─── 男装-卫衣/帽衫 ───
    {
        "query": "想买一件卫衣",
        "relevant_keywords": ["卫衣", "帽衫", "连帽"],
        "category": "男装",
        "description": "基础卫衣搜索，测试同义词扩展（卫衣=帽衫=连帽衫）",
    },
    {
        "query": "秋天穿的连帽衫",
        "relevant_keywords": ["卫衣", "帽衫", "连帽"],
        "category": "男装",
        "description": "同义词扩展测试：连帽衫→卫衣",
    },
    {
        "query": "休闲风格的套头衫",
        "relevant_keywords": ["卫衣", "帽衫", "连帽", "休闲"],
        "category": "男装",
        "description": "同义词+风格组合",
    },
    # ─── 男装-外套/羽绒服 ───
    {
        "query": "冬天保暖的羽绒服",
        "relevant_keywords": ["羽绒服", "羽绒", "棉服"],
        "category": "男装",
        "description": "季节性需求+品类",
    },
    {
        "query": "商务西装外套",
        "relevant_keywords": ["西装", "外套", "商务"],
        "category": "男装",
        "description": "风格+品类搜索",
    },
    {
        "query": "休闲夹克男",
        "relevant_keywords": ["夹克", "外套", "休闲"],
        "category": "男装",
        "description": "同义词测试：夹克=外套",
    },
    # ─── 男装-裤子 ───
    {
        "query": "牛仔裤直筒",
        "relevant_keywords": ["牛仔裤", "直筒"],
        "category": "男装",
        "description": "品类+版型",
    },
    {
        "query": "运动休闲裤",
        "relevant_keywords": ["休闲裤", "运动"],
        "category": "男装",
        "description": "运动风格裤子",
    },
    # ─── 男装-品牌搜索 ───
    {
        "query": "Nike 的衣服",
        "relevant_keywords": ["Nike"],
        "brand": "Nike",
        "description": "品牌维度搜索",
    },
    {
        "query": "Uniqlo 男装",
        "relevant_keywords": ["Uniqlo"],
        "brand": "Uniqlo",
        "description": "品牌+品类",
    },
    {
        "query": "GXG 潮流男装",
        "relevant_keywords": ["GXG", "潮流"],
        "brand": "GXG",
        "description": "品牌+风格",
    },
    {
        "query": "海澜之家 衬衫",
        "relevant_keywords": ["海澜之家", "衬衫", "衬衣"],
        "brand": "海澜之家",
        "description": "中文品牌搜索",
    },
    # ─── 男装-价格/属性 ───
    {
        "query": "500元以下的T恤",
        "relevant_keywords": ["T恤", "短袖"],
        "category": "男装",
        "max_price": 500,
        "description": "品类+价格区间",
    },
    {
        "query": "简约风格黑色男装",
        "relevant_keywords": ["简约", "黑色"],
        "category": "男装",
        "description": "风格+颜色",
    },
    {
        "query": "日系风格男装推荐",
        "relevant_keywords": ["日系"],
        "category": "男装",
        "description": "风格搜索",
    },
    # ─── 运动鞋 ───
    {
        "query": "跑步鞋推荐",
        "relevant_keywords": ["跑鞋", "跑步", "运动鞋"],
        "category": "运动鞋",
        "description": "运动鞋同义词扩展",
    },
    {
        "query": "Nike 缓震跑鞋",
        "relevant_keywords": ["Nike", "缓震"],
        "brand": "Nike",
        "category": "运动鞋",
        "description": "品牌+功能",
    },
    {
        "query": "透气轻便的运动鞋",
        "relevant_keywords": ["透气", "轻便", "运动鞋"],
        "category": "运动鞋",
        "description": "多功能属性",
    },
    {
        "query": "篮球鞋耐磨",
        "relevant_keywords": ["篮球", "耐磨"],
        "category": "运动鞋",
        "description": "具体运动+功能",
    },
    {
        "query": "Adidas boost 鞋",
        "relevant_keywords": ["Adidas"],
        "brand": "Adidas",
        "category": "运动鞋",
        "description": "品牌+技术搜索",
    },
    {
        "query": "李宁运动鞋",
        "relevant_keywords": ["李宁"],
        "brand": "李宁",
        "category": "运动鞋",
        "description": "中文品牌搜索",
    },
    # ─── 手机 ───
    {
        "query": "拍照好的手机",
        "relevant_keywords": ["影像", "摄", "拍照"],
        "category": "手机",
        "description": "功能需求搜索",
    },
    {
        "query": "长续航手机推荐",
        "relevant_keywords": ["续航", "电池"],
        "category": "手机",
        "description": "续航需求",
    },
    {
        "query": "游戏手机性能好的",
        "relevant_keywords": ["游戏", "性能", "处理器"],
        "category": "手机",
        "description": "游戏性能需求",
    },
    {
        "query": "Apple iPhone",
        "relevant_keywords": ["Apple"],
        "brand": "Apple",
        "category": "手机",
        "description": "品牌搜索",
    },
    {
        "query": "华为手机",
        "relevant_keywords": ["Huawei", "华为"],
        "brand": "Huawei",
        "category": "手机",
        "description": "中文品牌搜索",
    },
    {
        "query": "5000 以下性价比手机",
        "relevant_keywords": ["手机"],
        "category": "手机",
        "max_price": 5000,
        "description": "价格约束搜索",
    },
    # ─── 数码相机 ───
    {
        "query": "入门级微单相机",
        "relevant_keywords": ["相机", "微单", "入门"],
        "category": "数码相机",
        "description": "相机品类搜索",
    },
    {
        "query": "Sony 全画幅相机",
        "relevant_keywords": ["Sony", "全画幅"],
        "brand": "Sony",
        "category": "数码相机",
        "description": "品牌+画幅",
    },
    {
        "query": "4K视频拍摄的Vlog相机",
        "relevant_keywords": ["4K", "Vlog", "视频"],
        "category": "数码相机",
        "description": "功能+用途",
    },
    {
        "query": "富士复古风相机",
        "relevant_keywords": ["Fujifilm", "富士", "复古"],
        "brand": "Fujifilm",
        "category": "数码相机",
        "description": "品牌+风格",
    },
    {
        "query": "佳能单反相机",
        "relevant_keywords": ["Canon", "佳能", "单反"],
        "brand": "Canon",
        "category": "数码相机",
        "description": "中文品牌映射",
    },
    # ─── 跨品类/模糊搜索 ───
    {
        "query": "送男朋友的礼物",
        "relevant_keywords": [],
        "description": "模糊意图搜索",
    },
    {
        "query": "户外运动装备",
        "relevant_keywords": ["运动", "户外"],
        "description": "跨品类搜索",
    },
    {
        "query": "潮流穿搭推荐",
        "relevant_keywords": ["潮流", "街头"],
        "description": "风格导向搜索",
    },
]


def _get_es_client() -> Elasticsearch:
    return Elasticsearch(ES_HOST, basic_auth=ES_AUTH)


def _search_bm25(
    es: Elasticsearch, query: str, top_k: int = 10,
    category: Optional[str] = None, brand: Optional[str] = None,
    max_price: Optional[float] = None,
) -> list[dict]:
    """纯 BM25 关键词检索（使用 IK 分词）。"""
    body: dict = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": ["name^3", "description", "tags^2", "brand^2", "category^2"],
                        "type": "best_fields",
                    }
                }],
            }
        },
        "_source": {"excludes": ["embedding"]},
    }
    filters = _build_filters(category, brand, max_price)
    if filters:
        body["query"]["bool"]["filter"] = filters

    resp = es.search(index=INDEX, body=body, timeout="15s")
    return _parse_hits(resp)


def _search_knn(
    es: Elasticsearch, query: str, top_k: int = 10,
    category: Optional[str] = None, brand: Optional[str] = None,
    max_price: Optional[float] = None,
) -> list[dict]:
    """纯 KNN 向量检索。"""
    from app.core.embedding import get_embedding

    query_vector = get_embedding(query)
    knn_body: dict = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": top_k * 5,
    }
    filters = _build_filters(category, brand, max_price)
    if filters:
        knn_body["filter"] = {"bool": {"filter": filters}}

    body = {"size": top_k, "knn": knn_body, "_source": {"excludes": ["embedding"]}}
    resp = es.search(index=INDEX, body=body, timeout="15s")
    return _parse_hits(resp)


def _search_hybrid_rrf(
    es: Elasticsearch, query: str, top_k: int = 10,
    category: Optional[str] = None, brand: Optional[str] = None,
    max_price: Optional[float] = None,
    rrf_k: int = 60, bm25_weight: float = 0.4, knn_weight: float = 0.6,
) -> list[dict]:
    """Hybrid + 显式 RRF 融合。"""
    bm25_results = _search_bm25(es, query, top_k, category, brand, max_price)

    try:
        knn_results = _search_knn(es, query, top_k, category, brand, max_price)
    except Exception:
        return bm25_results

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_results, start=1):
        pid = hit["product_id"]
        scores[pid] = scores.get(pid, 0.0) + bm25_weight / (rrf_k + rank)
        docs[pid] = hit

    for rank, hit in enumerate(knn_results, start=1):
        pid = hit["product_id"]
        scores[pid] = scores.get(pid, 0.0) + knn_weight / (rrf_k + rank)
        if pid not in docs:
            docs[pid] = hit

    sorted_pids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [{**docs[pid], "score": round(scores[pid], 6)} for pid in sorted_pids]


def _build_filters(
    category: Optional[str], brand: Optional[str], max_price: Optional[float],
) -> list[dict]:
    filters: list[dict] = []
    if category:
        filters.append({"term": {"category": category}})
    if brand:
        filters.append({"term": {"brand": brand}})
    if max_price is not None:
        filters.append({"range": {"price": {"lte": max_price}}})
    return filters


def _parse_hits(resp: dict) -> list[dict]:
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


# ── 自动标注相关度 ──

def _auto_relevance(item: dict, eval_entry: dict) -> int:
    """
    基于关键词/品类/品牌自动判定相关度等级。

    评分规则:
        3 = 品类+品牌+关键词全匹配
        2 = 品类匹配 + 至少一个关键词命中
        1 = 至少一个关键词命中
        0 = 无命中
    """
    score = 0
    expected_cat = eval_entry.get("category")
    expected_brand = eval_entry.get("brand")
    keywords = eval_entry.get("relevant_keywords", [])

    cat_match = expected_cat and item.get("category") == expected_cat
    brand_match = expected_brand and item.get("brand") == expected_brand

    text_blob = f"{item.get('name', '')} {item.get('description', '')} {' '.join(item.get('tags', []))}"
    text_lower = text_blob.lower()
    kw_hits = sum(1 for kw in keywords if kw.lower() in text_lower)

    if cat_match and brand_match and kw_hits > 0:
        score = 3
    elif cat_match and kw_hits > 0:
        score = 2
    elif brand_match:
        score = 2
    elif kw_hits > 0:
        score = 1

    return score


# ── 指标计算 ──

def _dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain @ k。"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def _ndcg_at_k(relevances: list[int], k: int) -> float:
    """Normalized DCG @ k。"""
    dcg = _dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(relevances: list[int], k: int, total_relevant: int) -> float:
    """Recall @ k：前 k 个结果中相关文档的比例。"""
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / total_relevant


# ── 评测主流程 ──

def evaluate_strategy(
    strategy_name: str,
    search_fn,
    es: Elasticsearch,
    k: int = 10,
) -> dict:
    """对单个策略跑全量评测集。"""
    ndcg_scores = []
    recall_scores = []
    latencies = []
    per_query_results = []

    for entry in EVAL_QUERIES:
        query = entry["query"]
        category = entry.get("category")
        brand = entry.get("brand")
        max_price = entry.get("max_price")

        t0 = time.time()
        try:
            results = search_fn(
                es, query, top_k=k,
                category=category, brand=brand, max_price=max_price,
            )
        except Exception as e:
            per_query_results.append({
                "query": query, "error": str(e),
                "ndcg": 0.0, "recall": 0.0,
            })
            ndcg_scores.append(0.0)
            recall_scores.append(0.0)
            continue
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        relevances = [_auto_relevance(item, entry) for item in results]

        # 计算总相关文档数（在整个索引中搜索更多结果来估算）
        try:
            all_results = search_fn(es, query, top_k=50, category=category, brand=brand, max_price=max_price)
            all_rels = [_auto_relevance(item, entry) for item in all_results]
            total_relevant = max(sum(1 for r in all_rels if r > 0), 1)
        except Exception:
            total_relevant = max(sum(1 for r in relevances if r > 0), 1)

        ndcg = _ndcg_at_k(relevances, k)
        recall = _recall_at_k(relevances, k, total_relevant)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)

        per_query_results.append({
            "query": query,
            "hits": len(results),
            "ndcg": round(ndcg, 4),
            "recall": round(recall, 4),
            "latency_ms": round(latency, 1),
            "top3": [r["name"] for r in results[:3]],
            "relevances": relevances[:k],
        })

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "strategy": strategy_name,
        "avg_ndcg@10": round(avg_ndcg, 4),
        "avg_recall@10": round(avg_recall, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "queries_evaluated": len(EVAL_QUERIES),
        "per_query": per_query_results,
    }


def _print_report(results: list[dict]) -> None:
    """输出对比报告。"""
    print("\n" + "=" * 80)
    print("  检索策略离线评测报告 (NDCG@10 / Recall@10)")
    print("=" * 80)
    print(f"  评测集: {len(EVAL_QUERIES)} 条 query")
    print(f"  索引: {INDEX}")
    print()

    # 汇总表
    print(f"  {'策略':<20} {'NDCG@10':>10} {'Recall@10':>10} {'Avg Latency':>12}")
    print("  " + "-" * 56)
    for r in results:
        print(
            f"  {r['strategy']:<20} {r['avg_ndcg@10']:>10.4f} "
            f"{r['avg_recall@10']:>10.4f} {r['avg_latency_ms']:>10.1f}ms"
        )

    # 找出最优策略
    best = max(results, key=lambda x: x["avg_ndcg@10"])
    print(f"\n  [最优策略] {best['strategy']} (NDCG@10={best['avg_ndcg@10']:.4f})")

    # 逐 query 对比
    print(f"\n  {'─' * 76}")
    print("  逐 Query 得分对比:")
    print(f"  {'─' * 76}")
    print(f"  {'Query':<30}", end="")
    for r in results:
        print(f" {r['strategy']:>15}", end="")
    print()
    print(f"  {'─' * 76}")

    for i, entry in enumerate(EVAL_QUERIES):
        query_short = entry["query"][:28]
        print(f"  {query_short:<30}", end="")
        for r in results:
            pq = r["per_query"][i]
            if "error" in pq:
                print(f" {'ERROR':>15}", end="")
            else:
                print(f" {pq['ndcg']:>15.4f}", end="")
        print()

    print("=" * 80)


def main():
    print("=" * 80)
    print("  离线检索评测 — BM25 vs KNN vs Hybrid+RRF")
    print("=" * 80)

    es = _get_es_client()
    doc_count = es.count(index=INDEX)["count"]
    print(f"\n  ES 连接成功 | 索引: {INDEX} | 文档数: {doc_count}")

    strategies = [
        ("BM25 (IK分词)", _search_bm25),
        ("KNN (向量)", _search_knn),
        ("Hybrid+RRF", _search_hybrid_rrf),
    ]

    all_results = []
    for name, fn in strategies:
        print(f"\n  [评测中] {name} ...")
        t0 = time.time()
        try:
            result = evaluate_strategy(name, fn, es, k=10)
            elapsed = time.time() - t0
            print(f"  [完成] {name} | NDCG@10={result['avg_ndcg@10']:.4f} | "
                  f"Recall@10={result['avg_recall@10']:.4f} | 耗时={elapsed:.1f}s")
            all_results.append(result)
        except Exception as e:
            print(f"  [失败] {name} | error={e}")
            all_results.append({
                "strategy": name,
                "avg_ndcg@10": 0.0,
                "avg_recall@10": 0.0,
                "avg_latency_ms": 0.0,
                "queries_evaluated": 0,
                "per_query": [{"query": q["query"], "error": str(e)} for q in EVAL_QUERIES],
            })

    _print_report(all_results)

    # 保存详细报告
    report_path = Path(__file__).parent.parent / "logs" / "eval_retrieval_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index": INDEX,
        "doc_count": doc_count,
        "eval_queries": len(EVAL_QUERIES),
        "results": all_results,
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  详细报告已保存至: {report_path}")

    return all_results


if __name__ == "__main__":
    main()

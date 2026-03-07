"""
scripts/sync_to_vector_db.py
将商品数据同步到 Elasticsearch（商品检索），将评论/文档同步到 Milvus（RAG知识库）。

执行方式:
    python scripts/sync_to_vector_db.py              # 同步 ES + Milvus
    python scripts/sync_to_vector_db.py --target es  # 仅同步 ES
    python scripts/sync_to_vector_db.py --target milvus  # 仅同步 Milvus
    python scripts/sync_to_vector_db.py --drop-existing  # 重建索引/集合
"""

import argparse
import json
import sys
import time
from typing import Generator

from config import (
    EMBEDDING_CONFIG,
    ES_CONFIG,
    MILVUS_CONFIG,
    MOCK_DATA_FILE,
)

# ─────────────────────────── Embedding 客户端 ───────────────────────────

def _get_openai_client():
    """返回兼容 OpenAI 接口的客户端。"""
    try:
        from openai import OpenAI
    except ImportError:
        print("[错误] 请安装 openai 包: pip install openai")
        sys.exit(1)

    return OpenAI(
        api_key=EMBEDDING_CONFIG["api_key"] or "dummy",
        base_url=EMBEDDING_CONFIG["base_url"],
    ), EMBEDDING_CONFIG["model"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量获取文本 Embedding。
    按 EMBEDDING_CONFIG.batch_size 分批请求，批次间休眠防止限流。
    """
    client, model = _get_openai_client()
    batch_size = EMBEDDING_CONFIG["batch_size"]
    interval = EMBEDDING_CONFIG["request_interval"]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        if i + batch_size < len(texts):
            time.sleep(interval)

    return all_embeddings


def detect_embedding_dim() -> int:
    """发送一条探测请求，获取当前模型实际返回的向量维度。"""
    result = embed_texts(["probe"])
    dim = len(result[0])
    print(f"  [Embedding] 探测到向量维度: {dim}")
    return dim


# ─────────────────────────── 文本切片（Chunking） ───────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符数切片，相邻 Chunk 有 overlap 重叠。"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ─────────────────────────── Elasticsearch 同步 ───────────────────────────

def _build_es_mapping(vector_dim: int) -> dict:
    return {
        "mappings": {
            "properties": {
                "product_id":  {"type": "keyword"},
                "name":        {"type": "text", "analyzer": "standard"},
                "category":    {"type": "keyword"},
                "brand":       {"type": "keyword"},
                "price":       {"type": "float"},
                "tags":        {"type": "keyword"},
                "description": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": vector_dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }


def _get_es_client():
    try:
        from elasticsearch import Elasticsearch, helpers
        es_kwargs = {"hosts": ES_CONFIG["hosts"]}
        # 如果是 HTTPS 连接，跳过证书验证
        if ES_CONFIG["hosts"].startswith("https"):
            es_kwargs["verify_certs"] = False
            es_kwargs["ssl_show_warn"] = False
        
        if ES_CONFIG.get("username") and ES_CONFIG.get("password"):
            es_kwargs["basic_auth"] = (ES_CONFIG["username"], ES_CONFIG["password"])
        return Elasticsearch(**es_kwargs), helpers
    except ImportError:
        print("[错误] 请安装 elasticsearch 包: pip install elasticsearch")
        sys.exit(1)


def _es_bulk_actions(
    products: list,
    embeddings: list[list[float]],
    index_name: str,
) -> Generator:
    for product, embedding in zip(products, embeddings):
        yield {
            "_index": index_name,
            "_id": product["id"],
            "_source": {
                "product_id":  product["id"],
                "name":        product["name"],
                "category":    product["category"],
                "brand":       product["brand"],
                "price":       product["price"],
                "tags":        product["tags"],
                "description": product["description"],
                "embedding":   embedding,
            },
        }


def sync_elasticsearch(products: list, drop_existing: bool, vector_dim: int) -> None:
    print("\n[ES] 开始同步 Elasticsearch ...")
    es, helpers = _get_es_client()
    index_name = ES_CONFIG["index_name"]

    if drop_existing and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"  [ES] 已删除旧索引 {index_name}")

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=_build_es_mapping(vector_dim))
        print(f"  [ES] 已创建索引 {index_name}（vector_dim={vector_dim}）")

    print(f"  [ES] 对 {len(products)} 个商品描述进行 Embedding ...")
    texts = [p["description"] for p in products]
    embeddings = embed_texts(texts)

    bulk_size = ES_CONFIG["bulk_size"]
    total_ok = 0
    for i in range(0, len(products), bulk_size):
        batch_products = products[i: i + bulk_size]
        batch_embeddings = embeddings[i: i + bulk_size]
        ok, errors = helpers.bulk(
            es,
            _es_bulk_actions(batch_products, batch_embeddings, index_name),
            raise_on_error=False,
        )
        if errors:
            print(f"  [ES] 批次写入有 {len(errors)} 条失败: {errors[0]}")
        total_ok += ok
        print(f"  [ES] 已写入 {total_ok}/{len(products)} 条")

    print(f"✓ ES 同步完成，共写入 {total_ok} 个商品文档")


# ─────────────────────────── Milvus 同步 ───────────────────────────

def _get_milvus():
    try:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )
        return Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    except ImportError:
        print("[错误] 请安装 pymilvus 包: pip install pymilvus")
        sys.exit(1)


def _build_milvus_schema(Collection, CollectionSchema, DataType, FieldSchema, vector_dim: int):
    dim = vector_dim
    fields = [
        FieldSchema(name="chunk_id",    dtype=DataType.INT64,    is_primary=True, auto_id=True),
        FieldSchema(name="product_id",  dtype=DataType.VARCHAR,  max_length=36),
        FieldSchema(name="doc_type",    dtype=DataType.VARCHAR,  max_length=20),
        FieldSchema(name="text",        dtype=DataType.VARCHAR,  max_length=2000),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="商品评论与说明知识库")
    return schema


def sync_milvus(docs: list, drop_existing: bool, vector_dim: int) -> None:
    print("\n[Milvus] 开始同步 Milvus ...")
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility = _get_milvus()

    connections.connect(
        host=MILVUS_CONFIG["host"],
        port=str(MILVUS_CONFIG["port"]),
    )
    collection_name = MILVUS_CONFIG["collection_name"]

    if drop_existing and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"  [Milvus] 已删除旧集合 {collection_name}")

    if not utility.has_collection(collection_name):
        schema = _build_milvus_schema(Collection, CollectionSchema, DataType, FieldSchema, vector_dim)
        collection = Collection(name=collection_name, schema=schema)
        # 创建 IVF_FLAT 向量索引
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128},
            },
        )
        print(f"  [Milvus] 已创建集合 {collection_name}")
    else:
        collection = Collection(collection_name)

    # 文本切片
    chunk_size = MILVUS_CONFIG["chunk_size"]
    chunk_overlap = MILVUS_CONFIG["chunk_overlap"]

    all_product_ids: list[str] = []
    all_doc_types: list[str] = []
    all_texts: list[str] = []

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for chunk in chunks:
            all_product_ids.append(doc["product_id"])
            all_doc_types.append(doc.get("type", "review"))
            all_texts.append(chunk)

    print(f"  [Milvus] 共切出 {len(all_texts)} 个文本块，开始 Embedding ...")
    all_embeddings = embed_texts(all_texts)

    batch_size = MILVUS_CONFIG["batch_size"]
    total_inserted = 0
    for i in range(0, len(all_texts), batch_size):
        batch_data = [
            all_product_ids[i: i + batch_size],
            all_doc_types[i:  i + batch_size],
            all_texts[i:      i + batch_size],
            all_embeddings[i: i + batch_size],
        ]
        mr = collection.insert(batch_data)
        total_inserted += len(mr.primary_keys)
        print(f"  [Milvus] 已写入 {total_inserted}/{len(all_texts)} 个文本块")

    collection.flush()
    collection.load()
    print(f"✓ Milvus 同步完成，共写入 {total_inserted} 个文本块")


# ─────────────────────────── 主流程 ───────────────────────────

def main(target: str, drop_existing: bool) -> None:
    if not MOCK_DATA_FILE.exists():
        print(f"[错误] 未找到数据文件: {MOCK_DATA_FILE}")
        print("请先运行 python scripts/generate_mock_data.py 生成数据。")
        sys.exit(1)

    if not EMBEDDING_CONFIG["api_key"]:
        print("[警告] EMBEDDING_API_KEY 未设置，请在环境变量或 config.py 中配置。")

    print(f"[1/3] 读取数据文件: {MOCK_DATA_FILE}")
    with open(MOCK_DATA_FILE, "r", encoding="utf-8") as f:
        mock_data = json.load(f)
    products = mock_data.get("products", [])
    docs = mock_data.get("docs", [])
    print(f"  商品 {len(products)} 条，文档/评论 {len(docs)} 条")

    # 优先使用 config 中已配置的维度，否则自动探测
    cfg_dim = ES_CONFIG["vector_dim"] or MILVUS_CONFIG["vector_dim"]
    if cfg_dim:
        vector_dim = cfg_dim
        print(f"[2/3] 使用配置中的向量维度: {vector_dim}")
    else:
        print("[2/3] 探测 Embedding 模型实际维度 ...")
        vector_dim = detect_embedding_dim()

    if target in ("es", "all"):
        sync_elasticsearch(products, drop_existing, vector_dim)

    if target in ("milvus", "all"):
        sync_milvus(docs, drop_existing, vector_dim)

    print("\n全部同步任务完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步数据到 ES 和 Milvus")
    parser.add_argument(
        "--target",
        choices=["es", "milvus", "all"],
        default="all",
        help="同步目标: es / milvus / all（默认 all）",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="同步前先删除已有索引/集合（用于重置数据）",
    )
    args = parser.parse_args()
    main(target=args.target, drop_existing=args.drop_existing)

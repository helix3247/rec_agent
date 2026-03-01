"""
scripts/config.py
统一管理所有造数脚本的可配置项。
修改此文件中的参数即可控制数据规模、数据库连接和 API 设置。
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 明确指定加载项目根目录的 .env，避免因工作目录不同而找不到
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

# ─────────────────────────── 路径配置 ───────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MOCK_DATA_FILE = DATA_DIR / "mock_data.json"

# ─────────────────────────── 数据规模配置 ───────────────────────────
DATA_CONFIG = {
    # 商品数量（50~100 之间）
    "num_products": 80,
    # 用户数量
    "num_users": 10,
    # 每个商品生成的评论 / 说明书条数（3~5 之间随机取）
    "reviews_per_product_min": 3,
    "reviews_per_product_max": 5,
    # 随机种子（设为 None 则每次结果不同）
    "random_seed": 42,
}

# ─────────────────────────── MySQL 配置 ───────────────────────────
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "root123"),
    "database": os.getenv("MYSQL_DATABASE", "rec_agent"),
    "charset": "utf8mb4",
}

# ─────────────────────────── Elasticsearch 配置 ───────────────────────────
ES_CONFIG = {
    "hosts": os.getenv("ES_HOST", "http://127.0.0.1:9200"),
    "index_name": os.getenv("ES_INDEX_NAME", "product_index"),
    "username": os.getenv("ES_USERNAME", ""),
    "password": os.getenv("ES_PASSWORD", ""),
    # 向量维度（与 Embedding 模型保持一致，0 表示运行时自动探测）
    "vector_dim": int(os.getenv("ES_VECTOR_DIM", 0)),
    # 批量写入大小
    "bulk_size": int(os.getenv("ES_BULK_SIZE", 50)),
}

# ─────────────────────────── Milvus 配置 ───────────────────────────
MILVUS_CONFIG = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": int(os.getenv("MILVUS_PORT", 19530)),
    "collection_name": os.getenv("MILVUS_COLLECTION", "knowledge_base"),
    # 向量维度（与 Embedding 模型保持一致，0 表示运行时自动探测）
    "vector_dim": int(os.getenv("MILVUS_VECTOR_DIM", 0)),
    # 文本切片最大长度（字符数）
    "chunk_size": int(os.getenv("MILVUS_CHUNK_SIZE", 300)),
    # 相邻 Chunk 的重叠长度
    "chunk_overlap": int(os.getenv("MILVUS_CHUNK_OVERLAP", 50)),
    # 批量写入大小
    "batch_size": int(os.getenv("MILVUS_BATCH_SIZE", 100)),
}

# ─────────────────────────── Embedding API 配置 ───────────────────────────
EMBEDDING_CONFIG = {
    "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
    "api_key": os.getenv("API_KEY", ""),
    "base_url": os.getenv("BASE_URL", "https://api.openai.com/v1"),
    "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    "request_interval": float(os.getenv("EMBEDDING_REQUEST_INTERVAL", 0.1)),
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", 32)),
}

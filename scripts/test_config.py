"""
scripts/test_config.py
验证配置加载是否正常。运行后打印脱敏后的配置项。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings


def mask(value: str, show: int = 4) -> str:
    """脱敏处理：仅展示前 show 个字符"""
    if not value:
        return "(空)"
    if len(value) <= show:
        return value
    return value[:show] + "****"


def main():
    print("=" * 60)
    print("  配置加载验证")
    print("=" * 60)

    print("\n[LLM 配置]")
    print(f"  主模型 API Key  : {mask(settings.llm.llm_api_key)}")
    print(f"  主模型 Base URL : {settings.llm.llm_base_url}")
    print(f"  主模型 Model    : {settings.llm.llm_model}")
    print(f"  备用 API Key    : {mask(settings.llm.fallback_llm_api_key)}")
    print(f"  备用 Base URL   : {settings.llm.fallback_llm_base_url}")
    print(f"  备用 Model      : {settings.llm.fallback_llm_model}")

    print("\n[Embedding 配置]")
    print(f"  Provider  : {settings.embedding.embedding_provider}")
    print(f"  API Key   : {mask(settings.embedding.embedding_api_key)}")
    print(f"  Base URL  : {settings.embedding.embedding_base_url}")
    print(f"  Model     : {settings.embedding.embedding_model}")

    print("\n[LangSmith 配置]")
    print(f"  Tracing   : {settings.langsmith.langchain_tracing_v2}")
    print(f"  API Key   : {mask(settings.langsmith.langchain_api_key)}")
    print(f"  Project   : {settings.langsmith.langchain_project}")

    print("\n[MySQL 配置]")
    print(f"  Host      : {settings.mysql.mysql_host}")
    print(f"  Port      : {settings.mysql.mysql_port}")
    print(f"  User      : {settings.mysql.mysql_user}")
    print(f"  Password  : {mask(settings.mysql.mysql_password)}")
    print(f"  Database  : {settings.mysql.mysql_database}")

    print("\n[Elasticsearch 配置]")
    print(f"  Host      : {settings.es.es_host}")
    print(f"  Index     : {settings.es.es_index_name}")

    print("\n[Milvus 配置]")
    print(f"  Host       : {settings.milvus.milvus_host}")
    print(f"  Port       : {settings.milvus.milvus_port}")
    print(f"  Collection : {settings.milvus.milvus_collection}")

    print("\n[Redis 配置]")
    print(f"  Host      : {settings.redis.redis_host}")
    print(f"  Port      : {settings.redis.redis_port}")
    print(f"  DB        : {settings.redis.redis_db}")
    print(f"  Password  : {mask(settings.redis.redis_password)}")

    print("\n" + "=" * 60)
    print("  [OK] 配置加载验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
app/core/config.py
使用 pydantic-settings 统一管理应用配置，从 .env 文件和环境变量加载。
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent.parent


class LLMSettings(BaseSettings):
    """主模型 & 备用模型配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 主模型 (DeepSeek)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "gemini-2.5-flash-lite"

    # 备用模型 (OpenAI)
    fallback_llm_api_key: str = ""
    fallback_llm_base_url: str = "https://api.openai.com/v1"
    fallback_llm_model: str = "gpt-4o-mini"



class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_provider: str = "openai"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-large"


class LangSmithSettings(BaseSettings):
    """LangSmith 观测平台配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rec-agent"


class MySQLSettings(BaseSettings):
    """MySQL 数据库配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "root123"
    mysql_database: str = "rec_agent"


class ESSettings(BaseSettings):
    """Elasticsearch 配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    es_host: str = "http://127.0.0.1:9200"
    es_index_name: str = "product_index"
    es_username: str = ""
    es_password: str = ""


class MilvusSettings(BaseSettings):
    """Milvus 向量数据库配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"


class RedisSettings(BaseSettings):
    """Redis 缓存配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""


class Settings(BaseSettings):
    """聚合所有子配置的全局配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    langsmith: LangSmithSettings = LangSmithSettings()
    mysql: MySQLSettings = MySQLSettings()
    es: ESSettings = ESSettings()
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()

    # 应用级配置
    app_name: str = "Rec Agent"
    debug: bool = False
    log_level: str = "INFO"


# 全局单例
settings = Settings()

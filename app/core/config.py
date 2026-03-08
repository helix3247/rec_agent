"""
app/core/config.py
使用 pydantic-settings 统一管理应用配置，从 .env 文件和环境变量加载。
包含配置校验：API key 非空检查、端口范围校验、URL 格式校验。
"""

import re
from pathlib import Path
from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent.parent

_URL_PATTERN = re.compile(r"^https?://[^\s]+$")
_EMBEDDING_DIM_MAP = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def _validate_port(port: int, field_name: str) -> None:
    """校验端口号范围。"""
    if not 1 <= port <= 65535:
        raise ValueError(f"{field_name} 端口号 {port} 不在有效范围 (1-65535)")


def _validate_url(url: str, field_name: str) -> None:
    """校验 URL 格式。"""
    if url and not _URL_PATTERN.match(url):
        raise ValueError(f"{field_name} URL 格式无效: {url}")


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

    @model_validator(mode="after")
    def validate_llm_config(self) -> "LLMSettings":
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY 不能为空，请在 .env 中配置主模型 API Key")
        _validate_url(self.llm_base_url, "LLM_BASE_URL")
        if self.fallback_llm_base_url:
            _validate_url(self.fallback_llm_base_url, "FALLBACK_LLM_BASE_URL")
        return self



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

    @model_validator(mode="after")
    def validate_embedding_config(self) -> "EmbeddingSettings":
        _validate_url(self.embedding_base_url, "EMBEDDING_BASE_URL")
        return self

    @property
    def expected_dim(self) -> int | None:
        """根据模型名推断预期向量维度，用于与 Milvus 配置交叉校验。"""
        return _EMBEDDING_DIM_MAP.get(self.embedding_model)


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
    mysql_password: str = ""
    mysql_database: str = "rec_agent"

    @model_validator(mode="after")
    def validate_mysql_config(self) -> "MySQLSettings":
        _validate_port(self.mysql_port, "MYSQL_PORT")
        return self


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

    @model_validator(mode="after")
    def validate_es_config(self) -> "ESSettings":
        _validate_url(self.es_host, "ES_HOST")
        return self


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

    @model_validator(mode="after")
    def validate_milvus_config(self) -> "MilvusSettings":
        _validate_port(self.milvus_port, "MILVUS_PORT")
        return self


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

    @model_validator(mode="after")
    def validate_redis_config(self) -> "RedisSettings":
        _validate_port(self.redis_port, "REDIS_PORT")
        return self


class LangfuseSettings(BaseSettings):
    """Langfuse 可观测性平台配置"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    @model_validator(mode="after")
    def validate_langfuse_config(self) -> "LangfuseSettings":
        if self.langfuse_enabled:
            if not self.langfuse_public_key or not self.langfuse_secret_key:
                raise ValueError(
                    "Langfuse 已启用但密钥未配置: "
                    "请设置 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY"
                )
        _validate_url(self.langfuse_host, "LANGFUSE_HOST")
        return self


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
    langfuse: LangfuseSettings = LangfuseSettings()
    mysql: MySQLSettings = MySQLSettings()
    es: ESSettings = ESSettings()
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()

    # 应用级配置
    app_name: str = "Rec Agent"
    debug: bool = False
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_cross_settings(self) -> "Settings":
        """跨模块配置一致性校验：Embedding 维度 vs Milvus 集合配置。"""
        expected_dim = self.embedding.expected_dim
        if expected_dim:
            # memory.py 中 _MEMORY_VECTOR_DIM = 3072，此处直接用常量避免循环导入
            milvus_vector_dim = 3072
            if expected_dim != milvus_vector_dim:
                import warnings
                warnings.warn(
                    f"Embedding 模型 {self.embedding.embedding_model} 输出维度 {expected_dim}，"
                    f"但 Milvus user_memory 集合配置为 {milvus_vector_dim} 维。"
                    f"请确保两者一致，否则会导致向量插入失败。",
                    UserWarning,
                    stacklevel=2,
                )
        return self


def _create_settings() -> Settings:
    """
    安全创建配置单例。

    在子配置校验失败时（如 .env 缺少必要配置），降级为不校验模式，
    避免 import 阶段因配置不全而阻断整个模块加载。
    """
    try:
        return Settings()
    except Exception:
        import warnings
        warnings.warn(
            "配置校验失败，使用默认配置启动。请检查 .env 文件。",
            UserWarning,
            stacklevel=2,
        )

        class _PermissiveLLM(LLMSettings):
            @model_validator(mode="after")
            def validate_llm_config(self) -> "_PermissiveLLM":
                return self

        class _PermissiveLangfuse(LangfuseSettings):
            @model_validator(mode="after")
            def validate_langfuse_config(self) -> "_PermissiveLangfuse":
                return self

        class _PermissiveSettings(Settings):
            @model_validator(mode="after")
            def validate_cross_settings(self) -> "_PermissiveSettings":
                return self

        return _PermissiveSettings(
            llm=_PermissiveLLM(),
            langfuse=_PermissiveLangfuse(),
        )


# 全局单例
settings = _create_settings()

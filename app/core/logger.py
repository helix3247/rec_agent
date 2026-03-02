"""
app/core/logger.py
基于 loguru 的日志系统配置。
日志格式包含时间、级别、TraceID、AgentName，支持控制台 + 文件双输出，按天轮转。
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings, PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[trace_id]}</cyan> | "
    "<magenta>{extra[agent_name]}</magenta> | "
    "<level>{message}</level>"
)

_LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[trace_id]} | "
    "{extra[agent_name]} | "
    "{message}"
)


def setup_logger() -> None:
    """初始化 loguru 配置：移除默认 handler，添加控制台和文件 handler。"""
    logger.remove()

    logger.configure(extra={"trace_id": "-", "agent_name": "-"})

    logger.add(
        sys.stderr,
        format=_LOG_FORMAT,
        level=settings.log_level,
        colorize=True,
    )

    logger.add(
        str(_LOG_DIR / "app.log"),
        format=_LOG_FORMAT_FILE,
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )


def get_logger(agent_name: str = "-", trace_id: str = "-"):
    """获取带有 trace_id 和 agent_name 上下文的 logger 实例。"""
    return logger.bind(trace_id=trace_id, agent_name=agent_name)


setup_logger()

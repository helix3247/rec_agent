"""
scripts/test_logger.py
验证 loguru 日志系统配置是否正常：控制台输出 + 文件输出。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logger import get_logger


def main():
    print("=" * 60)
    print("  日志系统验证")
    print("=" * 60)

    log = get_logger(agent_name="TestLogger", trace_id="trace-test-001")

    log.debug("这是一条 DEBUG 日志")
    log.info("这是一条 INFO 日志")
    log.warning("这是一条 WARNING 日志")
    log.error("这是一条 ERROR 日志")

    log2 = get_logger(agent_name="ShoppingAgent", trace_id="trace-test-002")
    log2.info("模拟 ShoppingAgent 日志输出")

    log_file = Path(__file__).parent.parent / "logs" / "app.log"
    if log_file.exists():
        print(f"\n  [OK] 日志文件已创建: {log_file}")
        print(f"  [OK] 日志文件大小: {log_file.stat().st_size} bytes")
    else:
        print(f"\n  [FAIL] 日志文件未找到: {log_file}")

    print("\n" + "=" * 60)
    print("  日志系统验证完成！请检查控制台输出和 logs/app.log 文件。")
    print("=" * 60)


if __name__ == "__main__":
    main()

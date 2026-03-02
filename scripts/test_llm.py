"""
scripts/test_llm.py
分别测试主模型 (DeepSeek) 和备用模型 (OpenAI) 的连通性，
以及降级逻辑是否正常工作。
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from app.core.llm import get_llm, invoke_with_fallback
from app.core.logger import get_logger

_logger = get_logger(agent_name="TestLLM", trace_id="test-001")


def test_primary():
    """测试主模型连通性"""
    print("\n" + "=" * 60)
    print("  测试主模型 (primary) 连通性")
    print("=" * 60)
    try:
        llm = get_llm("primary")
        response = llm.invoke([HumanMessage(content="你好，请用一句话介绍你自己。")])
        print(f"  [OK] 主模型响应: {response.content[:100]}")
        _logger.info("主模型测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 主模型调用失败: {e}")
        _logger.error("主模型测试失败: {}", str(e))
        return False


def test_fallback():
    """测试备用模型连通性"""
    print("\n" + "=" * 60)
    print("  测试备用模型 (fallback) 连通性")
    print("=" * 60)
    try:
        llm = get_llm("fallback")
        response = llm.invoke([HumanMessage(content="你好，请用一句话介绍你自己。")])
        print(f"  [OK] 备用模型响应: {response.content[:100]}")
        _logger.info("备用模型测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 备用模型调用失败: {e}")
        _logger.error("备用模型测试失败: {}", str(e))
        return False


async def test_fallback_mechanism():
    """测试降级机制：调用 invoke_with_fallback"""
    print("\n" + "=" * 60)
    print("  测试降级机制 (invoke_with_fallback)")
    print("=" * 60)
    try:
        messages = [HumanMessage(content="请用一句话说一个有趣的事实。")]
        result = await invoke_with_fallback(messages)
        print(f"  [OK] 降级调用响应: {result[:100]}")
        _logger.info("降级机制测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 降级调用失败: {e}")
        _logger.error("降级机制测试失败: {}", str(e))
        return False


def main():
    print("=" * 60)
    print("  LLM 连通性测试")
    print("=" * 60)

    results = {}
    results["primary"] = test_primary()
    results["fallback"] = test_fallback()
    results["fallback_mechanism"] = asyncio.run(test_fallback_mechanism())

    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "[OK] 通过" if passed else "[FAIL] 失败"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n  所有测试通过!")
    else:
        print("\n  部分测试失败, 请检查配置。")


if __name__ == "__main__":
    main()

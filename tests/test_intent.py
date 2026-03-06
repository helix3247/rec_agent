"""
tests/test_intent.py
IntentParserAgent 意图识别准确率测试。
包含 10 个典型 Query 测试用例，验证意图分类和槽位抽取能力。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from app.core.llm import get_llm
from app.agents.intent_parser import _parse_intent_json
from app.prompts.intent import INTENT_SYSTEM_PROMPT


TEST_CASES = [
    # (query, 期望意图, 期望槽位检查)
    {
        "query": "推荐一款相机",
        "expected_intent": "search",
        "check_slots": {"category": "相机"},
    },
    {
        "query": "5000元以内适合夜拍的微单",
        "expected_intent": "search",
        "check_slots": {"budget": True, "category": True, "scenario": True},
    },
    {
        "query": "男生通勤穿搭推荐",
        "expected_intent": "outfit",
        "check_slots": {"scenario": True},
    },
    {
        "query": "Sony A7M4 的夜拍效果怎么样",
        "expected_intent": "qa",
        "check_slots": {},
    },
    {
        "query": "你好",
        "expected_intent": "chat",
        "check_slots": {},
    },
    {
        "query": "iPhone 16 和 Pixel 9 哪个拍照好",
        "expected_intent": "compare",
        "check_slots": {},
    },
    {
        "query": "去西藏旅游需要准备哪些装备",
        "expected_intent": "plan",
        "check_slots": {"scenario": True},
    },
    {
        "query": "有没有适合跑步的轻便运动鞋，预算300左右",
        "expected_intent": "search",
        "check_slots": {"budget": True, "category": True, "scenario": True},
    },
    {
        "query": "夏天约会穿什么好看",
        "expected_intent": "outfit",
        "check_slots": {"scenario": True},
    },
    {
        "query": "谢谢你的推荐，再见",
        "expected_intent": "chat",
        "check_slots": {},
    },
]


def run_intent_test():
    """运行 10 个意图识别测试用例。"""
    print("=" * 60)
    print("  IntentParser 意图识别准确率测试 (10 cases)")
    print("=" * 60)

    llm = get_llm("primary")

    passed = 0
    failed_cases = []

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected = case["expected_intent"]
        check_slots = case["check_slots"]

        try:
            response = llm.invoke([
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ])
            result = _parse_intent_json(response.content)

            intent_ok = result.intent == expected

            slots_ok = True
            slot_details = []
            for slot_key, expected_val in check_slots.items():
                actual = getattr(result, slot_key, None)
                if expected_val is True:
                    if not actual:
                        slots_ok = False
                        slot_details.append(f"{slot_key}: (empty)")
                elif isinstance(expected_val, str):
                    if not actual or expected_val not in actual:
                        slots_ok = False
                        slot_details.append(f"{slot_key}: expected contains '{expected_val}', got '{actual}'")

            if intent_ok and slots_ok:
                passed += 1
                status = "[OK]"
            else:
                failed_cases.append(i)
                status = "[FAIL]"
                if not intent_ok:
                    slot_details.insert(0, f"intent: expected={expected}, got={result.intent}")

            print(f"  {status} Case {i:2d} | query=\"{query}\"")
            print(f"           intent={result.intent} | budget={result.budget} | "
                  f"category={result.category} | scenario={result.scenario} | "
                  f"style={result.style} | must_have={result.must_have}")
            if slot_details:
                print(f"           issues: {', '.join(slot_details)}")

        except Exception as e:
            failed_cases.append(i)
            print(f"  [ERR]  Case {i:2d} | query=\"{query}\" | error={e}")

    print("\n" + "=" * 60)
    accuracy = passed / len(TEST_CASES) * 100
    print(f"  结果: {passed}/{len(TEST_CASES)} 通过 | 准确率: {accuracy:.0f}%")
    if failed_cases:
        print(f"  失败用例: {failed_cases}")
    if accuracy >= 90:
        print("  [OK] 准确率 >= 90%, 测试通过!")
    else:
        print("  [FAIL] 准确率 < 90%, 需要优化 Prompt")
    print("=" * 60)

    return accuracy >= 90


if __name__ == "__main__":
    run_intent_test()

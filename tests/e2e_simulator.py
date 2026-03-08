"""
tests/e2e_simulator.py
端到端回归测试 —— User Simulator + Judge 打分。

使用方式:
    conda activate rec_agent
    python tests/e2e_simulator.py

功能:
    1. 用固定用例集模拟用户对话
    2. 通过 /chat API (内部直接调用 Graph) 获取系统响应
    3. 用 LLM Judge 或规则打分，评估维度:
        - 是否先澄清再推荐（clarity_first）
        - 是否返回 ≥3 个候选（enough_candidates）
        - 是否引用检索证据（evidence_based）
        - 是否给出相关问题（has_suggestions）
    4. 输出评分与失败用例列表，可用于迭代对比
"""

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import SystemMessage

from app.core.logger import get_logger
from app.core.llm import get_llm

_log = get_logger(agent_name="E2ESimulator", trace_id="e2e-test")


# ─────────────────── 测试用例定义 ───────────────────


@dataclass
class SimulatorTestCase:
    """端到端测试用例。"""
    name: str
    conversations: list[dict]
    expected_intent: str
    checks: list[str] = field(default_factory=list)
    description: str = ""


TEST_CASES = [
    SimulatorTestCase(
        name="澄清式导购_模糊查询",
        description="用户输入模糊需求，系统应先澄清再推荐",
        conversations=[
            {"query": "想买个相机", "user_id": "test_user_01"},
        ],
        expected_intent="search",
        checks=["clarity_first"],
    ),
    SimulatorTestCase(
        name="导购推荐_完整查询",
        description="用户给出完整需求，应返回>=3个候选商品",
        conversations=[
            {"query": "推荐一款5000元以内适合拍风景的微单相机", "user_id": "test_user_02"},
        ],
        expected_intent="search",
        checks=["enough_candidates", "evidence_based", "has_suggestions"],
    ),
    SimulatorTestCase(
        name="穿搭推荐",
        description="穿搭需求应返回搭配方案",
        conversations=[
            {"query": "男生通勤穿搭推荐，预算2000元", "user_id": "test_user_03"},
        ],
        expected_intent="outfit",
        checks=["evidence_based", "has_suggestions"],
    ),
    SimulatorTestCase(
        name="商品问答_RAG",
        description="商品问答应基于知识库回答",
        conversations=[
            {
                "query": "Sony A7M4 夜拍效果怎么样？",
                "user_id": "test_user_04",
                "selected_product_id": "P001",
            },
        ],
        expected_intent="qa",
        checks=["evidence_based", "has_suggestions"],
    ),
    SimulatorTestCase(
        name="商品对比",
        description="对比查询应给出多商品对比分析",
        conversations=[
            {"query": "iPhone 16 和 Pixel 9 哪个拍照更好", "user_id": "test_user_05"},
        ],
        expected_intent="compare",
        checks=["evidence_based", "has_suggestions"],
    ),
    SimulatorTestCase(
        name="复杂规划_旅行装备",
        description="复杂需求应拆解为多步子任务",
        conversations=[
            {"query": "去西藏旅游需要准备哪些装备", "user_id": "test_user_06"},
        ],
        expected_intent="plan",
        checks=["evidence_based", "has_suggestions"],
    ),
    SimulatorTestCase(
        name="闲聊_打招呼",
        description="闲聊应友善回应并引导购物",
        conversations=[
            {"query": "你好，今天天气不错", "user_id": "test_user_07"},
        ],
        expected_intent="chat",
        checks=["has_suggestions"],
    ),
    SimulatorTestCase(
        name="多轮对话_澄清后推荐",
        description="两轮对话：先澄清后推荐",
        conversations=[
            {"query": "想买个相机", "user_id": "test_user_08"},
            {"query": "预算5000，主要拍风景", "user_id": "test_user_08"},
        ],
        expected_intent="search",
        checks=["enough_candidates", "has_suggestions"],
    ),
]


# ─────────────────── Graph 调用 ───────────────────


def _invoke_graph(query: str, thread_id: str, user_id: str = "", selected_product_id: str = "") -> dict:
    """直接调用 Graph 获取结果（绕过 HTTP 层）。"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from app.graph import app_graph
    from app.agents.dialog import load_history, load_slots

    trace_id = f"e2e-{uuid.uuid4().hex[:8]}"

    history = load_history(thread_id)
    stored_slots = load_slots(thread_id)
    messages = list(history)
    messages.append(HumanMessage(content=query))

    initial_state = {
        "messages": messages,
        "trace_id": trace_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "selected_product_id": selected_product_id or "",
        "task_status": "pending",
        "slots": stored_slots,
        "_request_start_time": time.time(),
        "_node_metrics": [],
        "_agent_route_path": [],
    }

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(app_graph.ainvoke(initial_state))
        loop.close()
    except RuntimeError:
        result = asyncio.get_event_loop().run_until_complete(app_graph.ainvoke(initial_state))

    return {
        "response": result.get("response", ""),
        "trace_id": trace_id,
        "thread_id": thread_id,
        "candidates": result.get("candidates", []),
        "suggested_questions": result.get("suggested_questions", []),
        "user_intent": result.get("user_intent", ""),
        "task_status": result.get("task_status", ""),
        "agent_route_path": result.get("_agent_route_path", []),
    }


# ─────────────────── Judge 评分 ───────────────────


JUDGE_SYSTEM_PROMPT = """你是一个AI系统评测专家。请对以下电商导购AI助手的回复进行评分。

## 评估维度

根据指定的检查项进行评估，每项得分为 0（不满足）或 1（满足）：

1. **clarity_first（澄清优先）**: 当用户查询信息不足时（如未指定预算、品类、用途等），AI是否先追问澄清，而非直接给出推荐。
2. **enough_candidates（候选充足）**: AI是否返回了3个或以上的候选商品供用户选择。
3. **evidence_based（基于证据）**: AI的回答是否引用了具体的商品参数、评论内容或检索数据，而非凭空编造。
4. **has_suggestions（有后续问题）**: AI是否给出了相关的后续问题建议，引导用户继续提问。

## 输入信息

- 用户查询: {query}
- AI回复: {response}
- 候选商品数: {candidates_count}
- 推荐问题: {suggested_questions}
- 检查项: {checks}

## 输出格式

请严格输出 JSON 格式，不要添加其他内容:
{{"scores": {{"check_name": 0_or_1, ...}}, "reason": "简要评分理由"}}
"""


def _judge_with_llm(
    query: str,
    response: str,
    candidates_count: int,
    suggested_questions: list[str],
    checks: list[str],
) -> dict[str, int]:
    """使用 LLM 作为 Judge 进行打分。"""
    prompt = JUDGE_SYSTEM_PROMPT.format(
        query=query,
        response=response[:500],
        candidates_count=candidates_count,
        suggested_questions=json.dumps(suggested_questions[:5], ensure_ascii=False),
        checks=json.dumps(checks, ensure_ascii=False),
    )

    try:
        llm = get_llm("primary", temperature=0)
        result = llm.invoke([SystemMessage(content=prompt)])
        content = result.content.strip()

        if "```json" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        return parsed.get("scores", {})
    except Exception as e:
        _log.warning("LLM Judge 打分失败，回退到规则打分 | error={}", str(e))
        return _judge_with_rules(query, response, candidates_count, suggested_questions, checks)


def _judge_with_rules(
    query: str,  # noqa: ARG001
    response: str,
    candidates_count: int,
    suggested_questions: list[str],
    checks: list[str],
) -> dict[str, int]:
    """规则兜底的 Judge 打分（query 保留以对齐 LLM Judge 签名）。"""
    scores = {}

    for check in checks:
        if check == "clarity_first":
            clarify_keywords = ["预算", "价格", "用途", "场景", "品牌", "请问", "您", "什么",
                                "哪种", "需求", "想要", "？", "需要了解"]
            has_clarify = any(kw in response for kw in clarify_keywords)
            has_no_product = candidates_count == 0
            scores[check] = 1 if (has_clarify and has_no_product) else 0

        elif check == "enough_candidates":
            scores[check] = 1 if candidates_count >= 3 else 0

        elif check == "evidence_based":
            evidence_signals = ["¥", "元", "评论", "评价", "用户", "参数", "规格",
                                "品牌", "型号", "ISO", "分辨率", "续航", "材质",
                                "根据", "显示"]
            has_evidence = sum(1 for kw in evidence_signals if kw in response) >= 2
            scores[check] = 1 if has_evidence else 0

        elif check == "has_suggestions":
            scores[check] = 1 if len(suggested_questions) >= 1 else 0

    return scores


# ─────────────────── 测试执行器 ───────────────────


@dataclass
class TestResult:
    """单个测试用例结果。"""
    case_name: str
    passed: bool
    scores: dict[str, int]
    total_checks: int
    passed_checks: int
    response_preview: str
    latency_ms: float
    details: dict


def run_single_test(case: SimulatorTestCase, use_llm_judge: bool = True) -> TestResult:
    """执行单个测试用例。"""
    _log.info("执行测试: {} | 对话轮数={}", case.name, len(case.conversations))

    thread_id = f"e2e-{uuid.uuid4().hex[:8]}"
    last_result = None

    t0 = time.time()
    for turn in case.conversations:
        last_result = _invoke_graph(
            query=turn["query"],
            thread_id=thread_id,
            user_id=turn.get("user_id", ""),
            selected_product_id=turn.get("selected_product_id", ""),
        )
    latency_ms = round((time.time() - t0) * 1000, 1)

    if last_result is None:
        return TestResult(
            case_name=case.name,
            passed=False,
            scores={},
            total_checks=len(case.checks),
            passed_checks=0,
            response_preview="(无结果)",
            latency_ms=latency_ms,
            details={"error": "Graph 未返回结果"},
        )

    response = last_result["response"]
    candidates = last_result["candidates"]
    suggested_questions = last_result["suggested_questions"]

    last_query = case.conversations[-1]["query"]

    if use_llm_judge:
        scores = _judge_with_llm(last_query, response, len(candidates), suggested_questions, case.checks)
    else:
        scores = _judge_with_rules(last_query, response, len(candidates), suggested_questions, case.checks)

    passed_checks = sum(scores.get(c, 0) for c in case.checks)
    all_passed = passed_checks == len(case.checks)

    return TestResult(
        case_name=case.name,
        passed=all_passed,
        scores=scores,
        total_checks=len(case.checks),
        passed_checks=passed_checks,
        response_preview=response[:120] if response else "(空回复)",
        latency_ms=latency_ms,
        details={
            "user_intent": last_result.get("user_intent", ""),
            "expected_intent": case.expected_intent,
            "candidates_count": len(candidates),
            "suggestions_count": len(suggested_questions),
            "agent_route_path": last_result.get("agent_route_path", []),
        },
    )


def run_all_tests(use_llm_judge: bool = True) -> list[TestResult]:
    """执行所有测试用例。"""
    results = []
    for case in TEST_CASES:
        try:
            result = run_single_test(case, use_llm_judge=use_llm_judge)
            results.append(result)
        except Exception as e:
            _log.error("测试用例执行异常 | case={} | error={}", case.name, str(e))
            results.append(TestResult(
                case_name=case.name,
                passed=False,
                scores={},
                total_checks=len(case.checks),
                passed_checks=0,
                response_preview=f"(异常: {str(e)[:80]})",
                latency_ms=0,
                details={"error": str(e)},
            ))
    return results


def print_report(results: list[TestResult]):
    """输出测试报告。"""
    print("\n" + "=" * 80)
    print("  E2E 回归测试报告 (User Simulator + Judge)")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    total_checks = sum(r.total_checks for r in results)
    passed_checks = sum(r.passed_checks for r in results)

    print(f"\n  用例总数: {total} | 通过: {passed} | 失败: {total - passed}")
    print(f"  检查项总数: {total_checks} | 通过: {passed_checks} | 失败: {total_checks - passed_checks}")
    if total_checks > 0:
        print(f"  总体得分: {passed_checks / total_checks * 100:.1f}%")

    print("\n  " + "-" * 76)
    print(f"  {'状态':<6} {'用例名':<25} {'得分':<12} {'耗时':<10} {'意图':<8}")
    print("  " + "-" * 76)

    failed_cases = []

    for r in results:
        status = "[OK]" if r.passed else "[FAIL]"
        score_str = f"{r.passed_checks}/{r.total_checks}"
        intent = r.details.get("user_intent", "-")
        print(f"  {status:<6} {r.case_name:<25} {score_str:<12} {r.latency_ms:>6.0f}ms   {intent}")

        if not r.passed:
            failed_cases.append(r)
            # 显示失败详情
            for check, score in r.scores.items():
                if score == 0:
                    print(f"         -> [FAIL] {check}")
            print(f"         -> 回复: {r.response_preview[:70]}...")

    print("\n  " + "-" * 76)

    # 各维度汇总
    print("\n  评估维度汇总:")
    all_checks = set()
    for r in results:
        all_checks.update(r.scores.keys())

    for check in sorted(all_checks):
        count = sum(1 for r in results if check in r.scores)
        score = sum(r.scores.get(check, 0) for r in results if check in r.scores)
        print(f"    {check:<25} {score}/{count} ({score / count * 100:.0f}%)" if count else f"    {check:<25} N/A")

    print("\n" + "=" * 80)

    if failed_cases:
        print(f"  [WARN] 共 {len(failed_cases)} 个用例未通过，需关注:")
        for r in failed_cases:
            print(f"    - {r.case_name}: {r.response_preview[:60]}...")
    else:
        print("  [PASS] 所有用例通过!")

    print("=" * 80)

    # 保存报告到文件
    report_path = Path(__file__).parent.parent / "logs" / "e2e_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": total,
        "passed_cases": passed,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "score_percent": round(passed_checks / total_checks * 100, 1) if total_checks else 0,
        "results": [
            {
                "case": r.case_name,
                "passed": r.passed,
                "scores": r.scores,
                "latency_ms": r.latency_ms,
                "details": r.details,
            }
            for r in results
        ],
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存至: {report_path}")

    return passed == total


def main():
    """E2E 回归测试入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="E2E 回归测试")
    parser.add_argument("--no-llm-judge", action="store_true", help="使用规则打分代替 LLM Judge")
    args = parser.parse_args()

    use_llm_judge = not args.no_llm_judge

    print("=" * 80)
    print("  E2E 回归测试 (User Simulator + Judge)")
    print(f"  Judge 模式: {'LLM' if use_llm_judge else '规则'}")
    print(f"  测试用例数: {len(TEST_CASES)}")
    print("=" * 80)

    t0 = time.time()
    results = run_all_tests(use_llm_judge=use_llm_judge)
    total_time = time.time() - t0

    all_passed = print_report(results)

    print(f"\n  总耗时: {total_time:.1f}s")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

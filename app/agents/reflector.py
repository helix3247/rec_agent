"""
app/agents/reflector.py
Reflector Node —— 反思与自我修正。
在 ShoppingAgent / OutfitAgent 输出后检查推荐结果质量，
若不符合用户约束则生成修正建议并回退给上游 Agent 重试。

循环控制：最大重试次数 ≤ 3。
重试策略变化：relax_filter -> rewrite_query -> clarify / adjust_budget。
"""

import json
import re

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, invoke_with_fallback_sync
from app.core.logger import get_logger
from app.prompts.reflector import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_BUDGET_ADVICE_PROMPT

MAX_RETRIES = 3


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        return json.loads(m.group())
    return json.loads(cleaned)


def _check_basic_issues(
    response: str,
    candidates: list[dict],
    slots: dict,
) -> dict | None:
    """规则前置检查：在调用 LLM 之前快速识别明显问题。"""
    if not candidates and not response:
        return {
            "passed": False,
            "reason": "推荐结果为空",
            "strategy": "relax_filter",
            "suggestion": "放宽检索过滤条件，扩大价格区间或去掉品类限制",
        }

    budget_str = slots.get("budget", "")
    if budget_str and candidates:
        budget_nums = re.findall(r"\d+", budget_str.replace(",", ""))
        if budget_nums:
            max_budget = max(float(n) for n in budget_nums)
            if "以上" not in budget_str:
                over_budget = [
                    c for c in candidates
                    if c.get("price", 0) > max_budget * 1.3
                ]
                if len(over_budget) > len(candidates) * 0.5:
                    return {
                        "passed": False,
                        "reason": f"超过半数商品价格超出用户预算 {budget_str}",
                        "strategy": "relax_filter",
                        "suggestion": f"收紧价格过滤至 {max_budget} 以内",
                    }

    return None


def _generate_budget_advice(
    slots: dict,
    log,
) -> str:
    """当用户预算不合理时，生成友好的预算调整建议回复。"""
    budget = slots.get("budget", "")
    category = slots.get("category", "")

    prompt = REFLECTOR_BUDGET_ADVICE_PROMPT.format(
        budget=budget or "未指定",
        category=category or "该类商品",
    )

    try:
        return invoke_with_fallback_sync(
            [SystemMessage(content=prompt)],
            temperature=0.5,
        )
    except Exception as e:
        log.warning("预算建议生成失败 | error={}", str(e))
        return (
            f"抱歉，以 {budget} 的预算购买 {category} 可能比较困难。"
            f"建议您适当提高预算，或者考虑一些性价比更高的替代方案。"
        )


def reflector_node(state: AgentState) -> dict:
    """
    Reflector 反思节点。

    检查上游 Agent（Shopping/Outfit）的推荐质量：
    - 通过 -> 直接放行到 ResponseFormatter
    - 不通过 -> 生成修正建议，回退给上游重试
    - 达到最大重试次数 -> 转入澄清 / 给出预算调整建议
    """
    trace_id = state.get("trace_id", "-")
    retry_count = state.get("reflection_count", 0)
    response = state.get("response", "")
    candidates = state.get("candidates", [])
    slots = state.get("slots", {})
    messages = state.get("messages", [])
    current_agent = state.get("current_agent", "")
    log = get_logger(agent_name="Reflector", trace_id=trace_id)

    log.info(
        "反思检查开始 | upstream={} | retry={}/{} | candidates_count={}",
        current_agent, retry_count, MAX_RETRIES, len(candidates),
    )

    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # ── 1. 规则前置检查 ──
    basic_issue = _check_basic_issues(response, candidates, slots)

    if basic_issue and not basic_issue["passed"]:
        log.info("规则检查不通过 | reason={}", basic_issue["reason"])
        reflection = basic_issue
    else:
        # ── 2. LLM 深度检查 ──
        candidates_json = json.dumps(candidates, ensure_ascii=False, default=str)
        system_prompt = REFLECTOR_SYSTEM_PROMPT.format(
            query=query,
            slots=json.dumps(slots, ensure_ascii=False),
            response=response[:500],
            candidates=candidates_json[:1000],
            retry_count=retry_count,
            max_retries=MAX_RETRIES,
        )

        try:
            llm = get_llm("primary", temperature=0.1)
            llm_result = llm.invoke([SystemMessage(content=system_prompt)])
            reflection = _extract_json(llm_result.content)
            log.info(
                "LLM 反思结果 | passed={} | strategy={}",
                reflection.get("passed"), reflection.get("strategy"),
            )
        except Exception as e:
            log.warning("反思 LLM 调用失败，默认放行 | error={}", str(e))
            reflection = {"passed": True, "reason": "反思模块异常，默认放行", "strategy": "none"}

    # ── 3. 根据反思结果决定下一步 ──
    if reflection.get("passed", True):
        log.info("反思通过，放行到 ResponseFormatter")
        return {
            "current_agent": "Reflector",
            "reflection_count": 0,
            "reflection_feedback": "",
        }

    new_retry_count = retry_count + 1
    strategy = reflection.get("strategy", "relax_filter")

    if new_retry_count > MAX_RETRIES or strategy == "clarify":
        log.info("达到最大重试次数或需要澄清，转入对话模式")
        return {
            "current_agent": "Reflector",
            "task_status": "needs_clarify",
            "reflection_count": new_retry_count,
            "reflection_feedback": reflection.get("suggestion", ""),
        }

    if strategy == "adjust_budget":
        log.info("需求不合理，生成预算调整建议")
        advice = _generate_budget_advice(slots, log)
        return {
            "current_agent": "Reflector",
            "task_status": "completed",
            "response": advice,
            "messages": [AIMessage(content=advice)],
            "reflection_count": 0,
            "reflection_feedback": "",
        }

    # 需要重试：根据策略调整 slots 和 feedback
    feedback = reflection.get("suggestion", "请扩大检索范围重试")

    adjusted_slots = dict(slots)
    if strategy == "relax_filter":
        budget = adjusted_slots.get("budget", "")
        if budget:
            nums = re.findall(r"\d+", budget.replace(",", ""))
            if nums:
                max_val = max(float(n) for n in nums)
                adjusted_slots["budget"] = f"{int(max_val * 1.5)}以内"
                feedback += f"（已将预算上限放宽至 {int(max_val * 1.5)} 元）"

    elif strategy == "rewrite_query":
        adjusted_query = reflection.get("adjusted_query", "")
        if adjusted_query:
            feedback = f"改写查询为: {adjusted_query}"

    log.info(
        "触发重试 | strategy={} | retry={}/{} | feedback={}",
        strategy, new_retry_count, MAX_RETRIES, feedback[:100],
    )

    return {
        "current_agent": "Reflector",
        "task_status": "retrying",
        "reflection_count": new_retry_count,
        "reflection_feedback": feedback,
        "slots": adjusted_slots,
    }


def reflect_route(state: AgentState) -> str:
    """
    Reflector 条件边路由函数。

    路由规则：
    - 需要重试 -> 回退到原 Agent（shopping / outfit）
    - 需要澄清 -> dialog
    - 通过且处于 planner 子任务模式 -> planner（收集结果继续下一步）
    - 通过（普通模式）-> response_formatter
    - 预算调整建议（已生成回复）-> response_formatter / planner
    """
    task_status = state.get("task_status", "")
    plan_steps = state.get("plan_steps", [])

    if task_status == "retrying":
        intent = state.get("user_intent", "search")
        if intent == "outfit":
            return "outfit"
        return "shopping"

    if task_status == "needs_clarify":
        return "dialog"

    if plan_steps:
        return "planner"

    return "response_formatter"

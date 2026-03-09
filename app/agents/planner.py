"""
app/agents/planner.py
PlannerNode —— 复杂任务拆解与规划执行。

使用 COT（Chain of Thought）思维链将复杂 Query 拆解为子任务序列，
每个子任务绑定到具体的 Agent/Tool，按序执行并整合结果。

流程:
    1. 首次进入: LLM 生成计划 -> 输出 plan_steps -> 条件边分发第一个子任务
    2. 子任务执行完毕后回到 planner -> 收集结果 -> 分发下一个子任务
    3. 所有子任务完成 -> LLM 整合结果 -> 输出最终回答
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.state import AgentState
from app.core.llm import get_llm, invoke_with_smart_routing, get_model_router
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, extract_token_usage
from app.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_INTEGRATE_PROMPT

MAX_PLAN_STEPS = 5


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


def _should_clarify_step(result: dict) -> bool:
    response = (result.get("result") or "").strip()
    candidates = result.get("candidates", [])
    if candidates:
        return False
    if not response:
        return True
    return any(
        key in response
        for key in ("未检索到", "没有找到", "暂时没有找到", "暂无相关信息", "无结果")
    )


async def _generate_plan(query: str, slots: dict, log) -> dict:
    """调用 LLM 生成任务计划。"""
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        query=query,
        slots=json.dumps(slots, ensure_ascii=False),
    )

    try:
        plan_text = await invoke_with_smart_routing(
            [SystemMessage(content=system_prompt)],
            agent_name="PlannerAgent",
            temperature=0.3,
        )
        plan = _extract_json(plan_text)
        log.info("任务规划生成成功 | steps_count={}", len(plan.get("steps", [])))
        return plan
    except Exception as e:
        log.error("任务规划彻底失败 | error={}", str(e))
        return {"plan_summary": "规划失败", "steps": []}


async def _integrate_results(
    query: str,
    plan_summary: str,
    plan_results: list[dict[str, Any]],
    log,
) -> str:
    """调用 LLM 整合所有子任务的执行结果。"""
    step_results_text = ""
    for i, result in enumerate(plan_results, 1):
        step_results_text += (
            f"\n### 步骤 {i}: {result.get('description', '')}\n"
            f"执行 Agent: {result.get('agent', '')}\n"
            f"结果:\n{result.get('result', '无结果')[:300]}\n"
        )
        if result.get("candidates"):
            step_results_text += "候选商品:\n"
            for c in result["candidates"][:3]:
                step_results_text += f"  - {c.get('title', '')} ¥{c.get('price', 0)}\n"

    prompt = PLANNER_INTEGRATE_PROMPT.format(
        query=query,
        plan_summary=plan_summary,
        step_results=step_results_text,
    )

    try:
        return await invoke_with_smart_routing(
            [SystemMessage(content=prompt)],
            agent_name="PlannerAgent",
            temperature=0.5,
        )
    except Exception as e:
        log.warning("结果整合 LLM 调用失败，使用拼接结果 | error={}", str(e))
        lines = [f"为您规划了 {plan_summary}：\n"]
        for i, result in enumerate(plan_results, 1):
            lines.append(f"{i}. {result.get('description', '')}：{result.get('result', '无结果')[:100]}")
        return "\n".join(lines)


async def planner_node(state: AgentState) -> dict:
    """
    PlannerNode 主节点。

    执行模式:
    A) 首次进入（无 plan_steps）-> 生成计划 -> 准备执行第一个子任务
    B) 子任务返回（有 plan_steps 且 plan_current_step < len）-> 收集结果 -> 准备下一个子任务
    C) 所有子任务完成 -> 整合结果 -> 输出最终回答
    """
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    plan_steps = state.get("plan_steps", [])
    plan_current = state.get("plan_current_step", 0)
    plan_results = list(state.get("plan_results", []))
    messages = state.get("messages", [])
    slots = state.get("slots", {})
    log = get_logger(agent_name="PlannerNode", trace_id=trace_id)

    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    # ── 模式 A: 首次进入，生成计划 ──
    if not plan_steps:
        log.info("任务规划开始 | query={}", query)
        plan = await _generate_plan(query, slots, log)

        steps = plan.get("steps", [])[:MAX_PLAN_STEPS]
        if not steps:
            log.warning("规划结果为空，回退到通用回答")
            fallback_reply = f"关于「{query}」，我建议您可以分步骤来选购。请告诉我您最关心的品类，我来为您推荐。"
            node_result = {
                "current_agent": "PlannerNode",
                "response": fallback_reply,
                "messages": [AIMessage(content=fallback_reply)],
                "task_status": "completed",
                "plan_steps": [],
                "plan_current_step": 0,
                "plan_results": [],
            }
            return {**node_result, **record_node_metrics(state, "PlannerNode", t0)}

        formatted_steps = []
        for s in steps:
            formatted_steps.append({
                "step": s.get("step", 0),
                "description": s.get("description", ""),
                "agent": s.get("agent", "shopping"),
                "params": s.get("params", {}),
                "status": "pending",
                "result": "",
            })

        log.info("任务计划生成完成 | summary={} | steps={}", plan.get("plan_summary", ""), len(formatted_steps))

        first_step = formatted_steps[0]
        first_params = first_step.get("params", {})
        step_slots = dict(slots)
        for key in ("category", "budget", "scenario", "style", "must_have"):
            if first_params.get(key):
                step_slots[key] = first_params[key]

        node_result = {
            "current_agent": "PlannerNode",
            "task_status": "executing_step",
            "plan_steps": formatted_steps,
            "plan_current_step": 1,
            "plan_results": [],
            "response": plan.get("plan_summary", ""),
            "slots": step_slots,
        }
        return {**node_result, **record_node_metrics(state, "PlannerNode", t0)}

    # ── 模式 B / C: 从子任务返回 ──
    # 收集上一个子任务的结果
    if plan_current > 0 and plan_current <= len(plan_steps):
        prev_step = plan_steps[plan_current - 1]
        prev_result = {
            "step": prev_step.get("step", plan_current),
            "description": prev_step.get("description", ""),
            "agent": prev_step.get("agent", ""),
            "result": state.get("response", ""),
            "candidates": state.get("candidates", []),
        }
        plan_results.append(prev_result)
        log.info("子任务 {} 结果已收集 | agent={}", plan_current, prev_step.get("agent", ""))

        if _should_clarify_step(prev_result):
            clarify_reply = (
                f"关于「{prev_step.get('description', '该步骤')}」我暂时没有找到合适的结果。"
                "请补充更具体的需求（如预算、品类或偏好），我再继续为您规划。"
            )
            node_result = {
                "current_agent": "PlannerNode",
                "response": clarify_reply,
                "messages": [AIMessage(content=clarify_reply)],
                "task_status": "clarifying",
                "plan_steps": [],
                "plan_current_step": 0,
                "plan_results": plan_results,
            }
            return {**node_result, **record_node_metrics(state, "PlannerNode", t0)}

    # 检查是否所有步骤已完成
    if plan_current >= len(plan_steps):
        log.info("所有子任务已完成，整合结果 | total_steps={}", len(plan_steps))

        plan_summary = state.get("response", "") or "综合购物方案"
        # 从 plan_steps 的首次规划中找到 plan_summary（存在 response 中）
        integrated = await _integrate_results(query, plan_summary, plan_results, log)

        all_candidates = []
        for r in plan_results:
            all_candidates.extend(r.get("candidates", []))

        node_result = {
            "current_agent": "PlannerNode",
            "response": integrated,
            "messages": [AIMessage(content=integrated)],
            "candidates": all_candidates[:10],
            "task_status": "completed",
            "plan_steps": [],
            "plan_current_step": 0,
            "plan_results": [],
        }
        return {**node_result, **record_node_metrics(state, "PlannerNode", t0)}

    # ── 准备执行下一个子任务 ──
    current_step = plan_steps[plan_current]
    agent = current_step.get("agent", "shopping")
    params = current_step.get("params", {})

    log.info(
        "准备执行子任务 | step={}/{} | agent={} | desc={}",
        plan_current + 1, len(plan_steps), agent, current_step.get("description", ""),
    )

    step_slots = dict(slots)
    for key in ("category", "budget", "scenario", "style", "must_have"):
        if params.get(key):
            step_slots[key] = params[key]

    node_result = {
        "current_agent": "PlannerNode",
        "task_status": "executing_step",
        "plan_current_step": plan_current + 1,
        "plan_results": plan_results,
        "slots": step_slots,
    }
    return {**node_result, **record_node_metrics(state, "PlannerNode", t0)}


def planner_route(state: AgentState) -> str:
    """
    Planner 条件边路由函数。

    路由规则：
    - task_status == "executing_step" -> 分发到对应的 Agent
    - task_status == "completed" -> response_formatter
    - task_status == "planning" 且有 plan_steps -> 再次进入 planner 准备分发
    """
    task_status = state.get("task_status", "")
    plan_steps = state.get("plan_steps", [])
    plan_current = state.get("plan_current_step", 0)

    if task_status in ("completed", "clarifying") or not plan_steps:
        return "response_formatter"

    if task_status == "executing_step" and plan_current > 0 and plan_current <= len(plan_steps):
        step = plan_steps[plan_current - 1]
        agent = step.get("agent", "shopping")

        agent_map = {
            "shopping": "shopping",
            "outfit": "outfit",
            "rag": "rag",
        }
        return agent_map.get(agent, "shopping")

    return "response_formatter"

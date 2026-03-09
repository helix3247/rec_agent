"""
app/agents/response_formatter.py
ResponseFormatter —— 统一处理各 Agent 输出的语言润色、格式化及相关问题推荐。
"""

import re

from langchain_core.messages import SystemMessage

from app.state import AgentState
from app.core.agent_routing import invoke_llm_with_routing
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics, merge_token_usage
from app.prompts.response_formatter import FORMATTER_SYSTEM_PROMPT, SUGGESTED_QUESTIONS_PROMPT


async def response_formatter_node(state: AgentState) -> dict:
    """响应格式化节点：润色回答 + 生成推荐问题。"""
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    intent = state.get("user_intent", "")
    response = state.get("response", "")
    candidates = state.get("candidates", [])
    messages = state.get("messages", [])
    log = get_logger(agent_name="ResponseFormatter", trace_id=trace_id)

    log.info("响应格式化开始 | intent={} | response_len={}", intent, len(response))

    task_status = state.get("task_status", "")
    formatted_response = response
    total_token_usage: dict[str, int] = {}

    if response and task_status != "clarifying":
        formatted_response, polish_usage = await _polish_response(response, intent, candidates, log)
        total_token_usage = merge_token_usage(total_token_usage, polish_usage)

    suggested_questions, sq_usage = await _generate_suggested_questions(
        intent=intent,
        messages=messages,
        response=formatted_response,
        candidates=candidates,
        log=log,
    )
    total_token_usage = merge_token_usage(total_token_usage, sq_usage)

    log.info("响应格式化完成 | suggested_questions_count={}", len(suggested_questions))
    node_result = {
        "current_agent": "ResponseFormatter",
        "response": formatted_response,
        "suggested_questions": suggested_questions,
        "task_status": task_status or "completed",
    }
    metrics = record_node_metrics(
        state, "ResponseFormatter", t0, token_usage=total_token_usage,
    )
    return {**node_result, **metrics}


async def _polish_response(
    raw_response: str,
    intent: str,
    candidates: list[dict],
    log,
) -> tuple[str, dict[str, int]]:
    """调用 LLM 对回答进行润色。返回 (润色后文本, token_usage)。"""
    system_prompt = FORMATTER_SYSTEM_PROMPT.format(
        intent=intent,
        raw_response=raw_response,
        candidates_count=len(candidates),
    )

    try:
        polished, usage = await invoke_llm_with_routing(
            [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"请润色以下回答：\n\n{raw_response}"),
            ],
            agent_name="ResponseFormatter", log=log, temperature=0.3,
        )
        if polished and len(polished.strip()) > 10:
            return polished.strip(), usage
    except Exception as e:
        log.warning("响应润色 LLM 调用失败，使用原始回答 | error={}", str(e))

    return raw_response, {}


async def _generate_suggested_questions(
    intent: str,
    messages: list,
    response: str,
    candidates: list[dict],
    log,
) -> tuple[list[str], dict[str, int]]:
    """调用 LLM 生成推荐的后续问题。返回 (问题列表, token_usage)。"""
    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    candidates_summary = "无"
    if candidates:
        items = [f"{c.get('title', '')}（¥{c.get('price', 0)}）" for c in candidates[:5]]
        candidates_summary = "、".join(items)

    response_summary = response[:200] if response else "无"

    system_prompt = SUGGESTED_QUESTIONS_PROMPT.format(
        intent=intent,
        query=query,
        response_summary=response_summary,
        candidates_summary=candidates_summary,
    )

    try:
        content, usage = await invoke_llm_with_routing(
            [SystemMessage(content=system_prompt)],
            agent_name="ResponseFormatter", log=log, temperature=0.7,
        )
        questions = _parse_questions(content)
        if questions:
            return questions[:5], usage
    except Exception as e:
        log.warning("推荐问题生成失败，使用规则兜底 | error={}", str(e))

    return _rule_based_questions(intent, candidates), {}


def _parse_questions(text: str) -> list[str]:
    """从 LLM 输出中解析问题列表。"""
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 去除编号前缀
        for prefix in ["- ", "• ", "· "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if line and len(line) > 3:
            line = re.sub(r"^\d+[.、)\]]\s*", "", line)
            if line:
                questions.append(line)
    return questions


def _rule_based_questions(intent: str, candidates: list[dict]) -> list[str]:
    """规则兜底的推荐问题。"""
    questions = {
        "search": [
            "有没有其他价位的选择？",
            "这些商品的用户评价怎么样？",
            "能推荐一些性价比更高的吗？",
        ],
        "outfit": [
            "有没有其他风格的搭配？",
            "这些单品有没有替代选择？",
            "如果预算更低一些，怎么搭配？",
        ],
        "qa": [
            "这款商品和同类产品相比怎么样？",
            "有没有用户反馈的常见问题？",
            "这款商品适合什么场景使用？",
        ],
        "compare": [
            "能更详细地对比一下它们的性能吗？",
            "哪个更适合新手使用？",
            "它们的售后服务怎么样？",
        ],
        "tool": [
            "我还有其他订单需要查询吗？",
            "如何申请退货退款？",
        ],
    }

    result = questions.get(intent, [
        "有什么可以帮您的吗？",
        "需要我推荐一些商品吗？",
        "想了解什么品类的商品？",
    ])

    # 如果有候选商品，补充个性化问题
    if candidates and len(candidates) > 0:
        first = candidates[0]
        title = first.get("title", "这款商品")
        result.append(f"{title}的详细参数怎么样？")

    return result[:5]

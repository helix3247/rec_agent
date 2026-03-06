"""
app/agents/dialog.py
DialogFlowAgent —— 多轮对话管理。
- 基于 Redis 维护会话级短期记忆（按 thread_id 存取对话历史）。
- 澄清式对话：当关键槽位缺失时生成追问。
- 闲聊兜底：处理 chat 意图的一般性对话。
"""

import json

import redis
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    messages_to_dict,
    messages_from_dict,
)

from app.state import AgentState
from app.core.config import settings
from app.core.llm import get_llm
from app.core.logger import get_logger
from app.prompts.dialog import CLARIFY_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT

_HISTORY_TTL = 1800  # 会话历史 Redis TTL: 30 分钟
_MAX_HISTORY = 20    # 最多保留的消息条数


def _get_redis_client() -> redis.Redis:
    """获取 Redis 客户端实例。"""
    return redis.Redis(
        host=settings.redis.redis_host,
        port=settings.redis.redis_port,
        db=settings.redis.redis_db,
        password=settings.redis.redis_password or None,
        decode_responses=True,
    )


def _history_key(thread_id: str) -> str:
    return f"dialog:history:{thread_id}"


def _slots_key(thread_id: str) -> str:
    return f"dialog:slots:{thread_id}"


def load_history(thread_id: str) -> list[BaseMessage]:
    """从 Redis 加载指定 thread_id 的对话历史。"""
    if not thread_id:
        return []
    try:
        r = _get_redis_client()
        raw = r.get(_history_key(thread_id))
        if not raw:
            return []
        msg_dicts = json.loads(raw)
        return messages_from_dict(msg_dicts)
    except Exception:
        return []


def save_history(thread_id: str, messages: list[BaseMessage]) -> None:
    """将对话历史保存到 Redis。"""
    if not thread_id:
        return
    try:
        r = _get_redis_client()
        trimmed = messages[-_MAX_HISTORY:]
        msg_dicts = messages_to_dict(trimmed)
        r.setex(_history_key(thread_id), _HISTORY_TTL, json.dumps(msg_dicts, ensure_ascii=False))
    except Exception:
        pass


def load_slots(thread_id: str) -> dict:
    """从 Redis 加载已积累的槽位信息。"""
    if not thread_id:
        return {}
    try:
        r = _get_redis_client()
        raw = r.get(_slots_key(thread_id))
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def save_slots(thread_id: str, slots: dict) -> None:
    """将槽位信息保存到 Redis。"""
    if not thread_id:
        return
    try:
        r = _get_redis_client()
        r.setex(_slots_key(thread_id), _HISTORY_TTL, json.dumps(slots, ensure_ascii=False))
    except Exception:
        pass


def _find_missing_slots(intent: str, slots: dict) -> list[str]:
    """根据意图判断哪些关键槽位缺失。"""
    required_map = {
        "search": ["category", "budget"],
        "outfit": ["scenario"],
    }
    required = required_map.get(intent, [])
    return [s for s in required if not slots.get(s)]


def dialog_node(state: AgentState) -> dict:
    """
    DialogFlowAgent 主节点。
    - 如果来自 Dispatcher 的澄清请求（intent 为 search/outfit 但缺槽位），生成追问。
    - 如果 intent 为 chat，进行闲聊回复。
    - 同时负责 Redis 对话历史的存取。
    """
    trace_id = state.get("trace_id", "-")
    thread_id = state.get("thread_id", "")
    intent = state.get("user_intent", "chat")
    slots = state.get("slots", {})
    messages = list(state.get("messages", []))
    log = get_logger(agent_name="DialogFlow", trace_id=trace_id)

    log.info("DialogFlow 开始 | intent={} | thread_id={}", intent, thread_id)

    # 从 Redis 加载历史对话，与当前 messages 合并
    history = load_history(thread_id)
    if history:
        log.info("加载历史对话 | count={}", len(history))

    # 合并 Redis 中积累的槽位
    stored_slots = load_slots(thread_id)
    merged_slots = {**stored_slots, **{k: v for k, v in slots.items() if v}}

    # 判断是澄清式追问还是闲聊
    missing = _find_missing_slots(intent, merged_slots)

    if missing and intent in ("search", "outfit"):
        log.info("槽位缺失，生成追问 | missing={}", missing)
        reply = _generate_clarification(intent, merged_slots, missing, history + messages, log)
        task_status = "clarifying"
    else:
        log.info("进入闲聊/通用对话模式")
        reply = _generate_chat_reply(history + messages, log)
        task_status = "completed"

    # 将本轮对话保存到 Redis
    updated_messages = messages + [AIMessage(content=reply)]
    save_history(thread_id, history + updated_messages)
    save_slots(thread_id, merged_slots)

    log.info("DialogFlow 完成 | status={}", task_status)
    return {
        "current_agent": "DialogFlow",
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "task_status": task_status,
        "slots": merged_slots,
    }


def _generate_clarification(
    intent: str,
    slots: dict,
    missing: list[str],
    messages: list[BaseMessage],
    log,
) -> str:
    """调用 LLM 生成澄清式追问。"""
    slot_labels = {
        "budget": "预算",
        "category": "品类",
        "scenario": "使用场景",
        "style": "风格偏好",
        "must_have": "必须特征",
    }
    known_str = ", ".join(f"{slot_labels.get(k, k)}={v}" for k, v in slots.items() if v) or "暂无"
    missing_str = ", ".join(slot_labels.get(s, s) for s in missing)

    system_prompt = CLARIFY_SYSTEM_PROMPT.format(
        intent=intent,
        slots=known_str,
        missing_slots=missing_str,
    )

    try:
        llm = get_llm("primary")
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
        return response.content
    except Exception as e:
        log.warning("澄清式对话 LLM 调用失败，使用 fallback | error={}", str(e))
        try:
            llm = get_llm("fallback")
            response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
            return response.content
        except Exception:
            return "请问您能补充一下更多信息吗？比如预算范围、使用场景等，这样我能更好地为您推荐。"


def _generate_chat_reply(messages: list[BaseMessage], log) -> str:
    """调用 LLM 生成闲聊回复。"""
    try:
        llm = get_llm("primary")
        response = llm.invoke([SystemMessage(content=CHAT_SYSTEM_PROMPT)] + messages)
        return response.content
    except Exception as e:
        log.warning("闲聊 LLM 调用失败，使用 fallback | error={}", str(e))
        try:
            llm = get_llm("fallback")
            response = llm.invoke([SystemMessage(content=CHAT_SYSTEM_PROMPT)] + messages)
            return response.content
        except Exception:
            return "你好！有什么可以帮您的吗？如果您有购物需求，可以直接告诉我。"

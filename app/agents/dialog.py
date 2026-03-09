"""
app/agents/dialog.py
DialogFlowAgent —— 多轮对话管理。
- 基于 Redis 维护会话级短期记忆（按 thread_id 存取对话历史）。
- 支持长期记忆迁移：会话结束时将对话摘要写入 Milvus，新会话加载历史偏好。
- 澄清式对话：当关键槽位缺失时生成追问。
- 闲聊兜底：处理 chat 意图的一般性对话。

与 checkpoint.py 的职责边界：
    本模块负责**业务级**对话记忆管理：
        - Redis 短期历史：跨请求的对话上下文拼接（load_history / save_history）
        - Redis 槽位积累：跨请求的需求槽位持久化（load_slots / save_slots）
        - Milvus 长期记忆：会话结束时迁移对话摘要，新会话加载历史偏好

    checkpoint.py 负责**框架级** LangGraph 状态持久化：
        - 自动保存 Graph 执行过程中每个节点的全量状态快照
        - 支持进程重启后按 thread_id 恢复中断的 Graph 执行

    两者并行运行：即使 checkpoint 不可用，业务级记忆管理仍正常工作。
"""

import json
import threading

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
from app.core.agent_routing import invoke_llm_with_routing
from app.core.logger import get_logger
from app.core.metrics import start_node_timer, record_node_metrics
from app.prompts.dialog import CLARIFY_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from app.tools.memory import (
    migrate_to_long_term,
    recall_long_term_memory,
    format_memory_context,
)

_HISTORY_TTL = 1800  # 会话历史 Redis TTL: 30 分钟
_MAX_HISTORY = 20    # 最多保留的消息条数
_MIGRATION_TTL_THRESHOLD = 300  # TTL 低于此值时触发迁移 (5 分钟)
_logger = get_logger(agent_name="DialogFlow")


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


def check_and_migrate_memory(thread_id: str, user_id: str) -> None:
    """
    检查 Redis 中会话 TTL，若即将过期则触发记忆迁移到 Milvus。
    在后台线程中执行，不阻塞主流程。
    """
    if not thread_id or not user_id:
        return

    try:
        r = _get_redis_client()
        ttl = r.ttl(_history_key(thread_id))
        if 0 < ttl < _MIGRATION_TTL_THRESHOLD:
            messages = load_history(thread_id)
            if messages:
                _logger.info(
                    "会话 TTL 即将过期，触发记忆迁移 | thread_id={} | ttl={}s",
                    thread_id, ttl,
                )
                t = threading.Thread(
                    target=migrate_to_long_term,
                    args=(user_id, thread_id, messages),
                    daemon=True,
                )
                t.start()
    except Exception:
        pass


def end_session_and_migrate(thread_id: str, user_id: str) -> bool:
    """
    显式结束会话，将对话迁移到长期记忆后清理 Redis。

    Returns:
        是否迁移成功。
    """
    if not thread_id or not user_id:
        return False

    log = get_logger(agent_name="DialogFlow")
    messages = load_history(thread_id)
    if not messages:
        return False

    success = migrate_to_long_term(user_id, thread_id, messages)

    if success:
        try:
            r = _get_redis_client()
            r.delete(_history_key(thread_id))
            r.delete(_slots_key(thread_id))
            log.info("会话结束并迁移完成 | thread_id={}", thread_id)
        except Exception:
            pass

    return success


def load_long_term_context(user_id: str, query: str = "") -> str:
    """
    加载用户的长期记忆上下文，用于注入新会话的 Prompt。

    Args:
        user_id: 用户 ID。
        query: 当前查询（用于语义相关性检索）。

    Returns:
        格式化的长期记忆上下文文本，可直接拼入 System Prompt。
    """
    if not user_id:
        return ""

    memories = recall_long_term_memory(user_id, query=query, top_k=3)
    return format_memory_context(memories)


def _find_missing_slots(intent: str, slots: dict) -> list[str]:
    """根据意图判断哪些关键槽位缺失。"""
    required_map = {
        "search": ["category", "budget"],
        "outfit": ["scenario"],
    }
    required = required_map.get(intent, [])
    return [s for s in required if not slots.get(s)]


async def dialog_node(state: AgentState) -> dict:
    """
    DialogFlowAgent 主节点。
    - 如果来自 Dispatcher 的澄清请求（intent 为 search/outfit 但缺槽位），生成追问。
    - 如果 intent 为 chat，进行闲聊回复。
    - 负责将本轮对话保存到 Redis（历史已由 API 层加载并注入 state.messages）。
    """
    t0 = start_node_timer()
    trace_id = state.get("trace_id", "-")
    thread_id = state.get("thread_id", "")
    intent = state.get("user_intent", "chat")
    slots = state.get("slots", {})
    messages = list(state.get("messages", []))
    log = get_logger(agent_name="DialogFlow", trace_id=trace_id)

    log.info("DialogFlow 开始 | intent={} | thread_id={} | messages_count={}", intent, thread_id, len(messages))

    # 合并 Redis 中积累的槽位与当前 state 中的槽位
    stored_slots = load_slots(thread_id)
    merged_slots = {**stored_slots, **{k: v for k, v in slots.items() if v}}

    missing = _find_missing_slots(intent, merged_slots)
    token_usage: dict[str, int] = {}

    if missing and intent in ("search", "outfit"):
        log.info("槽位缺失，生成追问 | missing={}", missing)
        reply, token_usage = await _generate_clarification(intent, merged_slots, missing, messages, log)
        task_status = "clarifying"
    else:
        log.info("进入闲聊/通用对话模式")
        reply, token_usage = await _generate_chat_reply(messages, log)
        task_status = "completed"

    # 将包含本轮回复的完整对话保存到 Redis
    save_history(thread_id, messages + [AIMessage(content=reply)])
    save_slots(thread_id, merged_slots)

    # 异步检查是否需要迁移记忆到 Milvus（TTL 即将过期时触发）
    user_id = state.get("user_id", "")
    check_and_migrate_memory(thread_id, user_id)

    log.info("DialogFlow 完成 | status={}", task_status)
    node_result = {
        "current_agent": "DialogFlow",
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "task_status": task_status,
        "slots": merged_slots,
    }
    metrics = record_node_metrics(
        state, "DialogFlow", t0, token_usage=token_usage,
    )
    return {**node_result, **metrics}


async def _generate_clarification(
    intent: str,
    slots: dict,
    missing: list[str],
    messages: list[BaseMessage],
    log,
) -> tuple[str, dict[str, int]]:
    """调用 LLM 生成澄清式追问。返回 (回复文本, token_usage)。"""
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
        return await invoke_llm_with_routing(
            [SystemMessage(content=system_prompt)] + messages,
            agent_name="DialogFlow", log=log,
        )
    except Exception:
        return "请问您能补充一下更多信息吗？比如预算范围、使用场景等，这样我能更好地为您推荐。", {}


async def _generate_chat_reply(messages: list[BaseMessage], log) -> tuple[str, dict[str, int]]:
    """调用 LLM 生成闲聊回复。返回 (回复文本, token_usage)。"""
    try:
        return await invoke_llm_with_routing(
            [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + messages,
            agent_name="DialogFlow", log=log,
        )
    except Exception:
        return "你好！有什么可以帮您的吗？如果您有购物需求，可以直接告诉我。", {}

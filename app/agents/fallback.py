"""
app/agents/fallback.py
FallbackAgent —— 智能模型路由与降级机制。

核心能力:
    - 基于任务复杂度进行模型路由（轻量任务用轻量模型，复杂任务用强模型）
    - 主模型调用失败时自动降级至备用模型
    - 监控模型接口状态（响应时间、错误率），动态调整路由策略
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from app.core.logger import get_logger

_logger = get_logger(agent_name="FallbackAgent")


class TaskComplexity(Enum):
    """任务复杂度级别"""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


@dataclass
class ModelHealthMetrics:
    """模型接口健康指标"""
    total_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    last_failure_time: float = 0
    consecutive_failures: int = 0

    @property
    def error_rate(self) -> float:
        if not self.recent_errors:
            return 0.0
        return sum(self.recent_errors) / len(self.recent_errors)

    @property
    def avg_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_failures >= 3:
            cooldown_elapsed = time.time() - self.last_failure_time
            if cooldown_elapsed < 60:
                return False
        return self.error_rate < 0.5

    def record_success(self, latency_ms: float):
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.recent_latencies.append(latency_ms)
        self.recent_errors.append(0)
        self.consecutive_failures = 0

    def record_failure(self):
        self.total_calls += 1
        self.failed_calls += 1
        self.recent_errors.append(1)
        self.consecutive_failures += 1
        self.last_failure_time = time.time()


_COMPLEXITY_MAP = {
    "chat": TaskComplexity.LIGHT,
    "unknown": TaskComplexity.LIGHT,
    "search": TaskComplexity.MEDIUM,
    "qa": TaskComplexity.MEDIUM,
    "tool": TaskComplexity.MEDIUM,
    "outfit": TaskComplexity.MEDIUM,
    "compare": TaskComplexity.HEAVY,
    "plan": TaskComplexity.HEAVY,
}

_AGENT_COMPLEXITY = {
    "DialogFlow": TaskComplexity.LIGHT,
    "IntentParser": TaskComplexity.LIGHT,
    "ShoppingAgent": TaskComplexity.MEDIUM,
    "RAGAgent": TaskComplexity.MEDIUM,
    "ToolCallAgent": TaskComplexity.MEDIUM,
    "OutfitAgent": TaskComplexity.MEDIUM,
    "Reflector": TaskComplexity.HEAVY,
    "PlannerAgent": TaskComplexity.HEAVY,
    "ResponseFormatter": TaskComplexity.LIGHT,
}


class SmartModelRouter:
    """
    智能模型路由器。

    根据任务复杂度和模型健康状态动态选择最优模型。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: dict[str, ModelHealthMetrics] = {
            "primary": ModelHealthMetrics(),
            "fallback": ModelHealthMetrics(),
        }

    def get_metrics(self, model_type: str) -> ModelHealthMetrics:
        with self._lock:
            if model_type not in self._metrics:
                self._metrics[model_type] = ModelHealthMetrics()
            return self._metrics[model_type]

    def record_call(self, model_type: str, success: bool, latency_ms: float):
        """记录一次模型调用的结果。"""
        with self._lock:
            metrics = self._metrics.setdefault(model_type, ModelHealthMetrics())
            if success:
                metrics.record_success(latency_ms)
            else:
                metrics.record_failure()

    def classify_complexity(
        self,
        intent: str = "",
        agent_name: str = "",
        message_count: int = 0,
    ) -> TaskComplexity:
        """
        根据意图、Agent 名称和对话长度判断任务复杂度。
        """
        if agent_name and agent_name in _AGENT_COMPLEXITY:
            base = _AGENT_COMPLEXITY[agent_name]
        elif intent:
            base = _COMPLEXITY_MAP.get(intent, TaskComplexity.MEDIUM)
        else:
            base = TaskComplexity.MEDIUM

        if message_count > 10 and base == TaskComplexity.LIGHT:
            base = TaskComplexity.MEDIUM

        return base

    def select_model(
        self,
        complexity: TaskComplexity,
    ) -> str:
        """
        根据任务复杂度和模型健康状态选择模型。

        路由策略:
            - LIGHT 任务: 优先使用 primary（成本低），不健康则切到 fallback
            - MEDIUM 任务: 优先使用 primary，不健康则切到 fallback
            - HEAVY 任务: 优先使用 fallback（更强），不健康则切到 primary

        当模型连续失败时会进入冷却期（60s），冷却期内不使用该模型。
        """
        primary_healthy = self.get_metrics("primary").is_healthy
        fallback_healthy = self.get_metrics("fallback").is_healthy

        if complexity == TaskComplexity.HEAVY:
            if fallback_healthy:
                return "fallback"
            if primary_healthy:
                _logger.warning("强模型不健康，降级使用 primary 处理复杂任务")
                return "primary"
        else:
            if primary_healthy:
                return "primary"
            if fallback_healthy:
                _logger.warning("primary 不健康，降级使用 fallback")
                return "fallback"

        _logger.error("所有模型均不健康，强制使用 primary")
        return "primary"

    def get_llm_with_routing(
        self,
        intent: str = "",
        agent_name: str = "",
        message_count: int = 0,
        **kwargs,
    ) -> tuple[ChatOpenAI, str]:
        """
        智能路由获取 LLM 客户端。

        Returns:
            (llm_client, model_type) 元组。
        """
        from app.core.llm import get_llm

        complexity = self.classify_complexity(intent, agent_name, message_count)
        model_type = self.select_model(complexity)

        _logger.debug(
            "模型路由 | agent={} | intent={} | complexity={} | selected={}",
            agent_name, intent, complexity.value, model_type,
        )

        return get_llm(model_type, **kwargs), model_type

    async def invoke_with_smart_routing(
        self,
        messages: list[BaseMessage],
        intent: str = "",
        agent_name: str = "",
        message_count: int = 0,
        **kwargs,
    ) -> str:
        """
        带智能路由和降级的 LLM 异步调用。

        路由顺序: 按复杂度选择主模型 -> 失败则自动降级到另一个模型。
        """
        from app.core.llm import get_llm

        complexity = self.classify_complexity(intent, agent_name, message_count)
        preferred = self.select_model(complexity)
        fallback = "fallback" if preferred == "primary" else "primary"

        for model_type in [preferred, fallback]:
            t0 = time.time()
            try:
                llm = get_llm(model_type, **kwargs)
                response = await llm.ainvoke(messages)
                latency_ms = (time.time() - t0) * 1000
                self.record_call(model_type, True, latency_ms)
                _logger.info(
                    "LLM 调用成功 | model={} | complexity={} | latency={}ms",
                    model_type, complexity.value, round(latency_ms),
                )
                return response.content
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                self.record_call(model_type, False, latency_ms)
                _logger.warning(
                    "LLM 调用失败 | model={} | error={} | latency={}ms",
                    model_type, str(e), round(latency_ms),
                )
                if model_type == fallback:
                    raise

        raise RuntimeError("所有模型均调用失败")

    def invoke_with_smart_routing_sync(
        self,
        messages: list[BaseMessage],
        intent: str = "",
        agent_name: str = "",
        message_count: int = 0,
        **kwargs,
    ) -> str:
        """同步版本的智能路由 LLM 调用。"""
        from app.core.llm import get_llm

        complexity = self.classify_complexity(intent, agent_name, message_count)
        preferred = self.select_model(complexity)
        fallback = "fallback" if preferred == "primary" else "primary"

        for model_type in [preferred, fallback]:
            t0 = time.time()
            try:
                llm = get_llm(model_type, **kwargs)
                response = llm.invoke(messages)
                latency_ms = (time.time() - t0) * 1000
                self.record_call(model_type, True, latency_ms)
                _logger.info(
                    "LLM 调用成功 | model={} | complexity={} | latency={}ms",
                    model_type, complexity.value, round(latency_ms),
                )
                return response.content
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                self.record_call(model_type, False, latency_ms)
                _logger.warning(
                    "LLM 调用失败 | model={} | error={} | latency={}ms",
                    model_type, str(e), round(latency_ms),
                )
                if model_type == fallback:
                    raise

        raise RuntimeError("所有模型均调用失败")

    def get_health_report(self) -> dict:
        """获取所有模型的健康状态报告。"""
        report = {}
        with self._lock:
            for model_type, metrics in self._metrics.items():
                report[model_type] = {
                    "healthy": metrics.is_healthy,
                    "total_calls": metrics.total_calls,
                    "error_rate": round(metrics.error_rate, 4),
                    "avg_latency_ms": round(metrics.avg_latency_ms, 1),
                    "consecutive_failures": metrics.consecutive_failures,
                }
        return report


# 全局单例路由器
model_router = SmartModelRouter()

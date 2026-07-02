"""BaseAgent — 所有 Agent 的抽象基类与注册表。

对应文档：AGENTS.md §5, TECHNICAL_SPEC.md §3.2
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .structs import (
    AgentState,
    ErrorRecord,
    ProfileResult,
    StepConfig,
    StepStats,
)


class BaseAgent(ABC):
    """所有 Agent 的基类。

    对应 AGENTS.md §5 和 TECHNICAL_SPEC.md §3.2。
    定义了 Agent 的标准生命周期：INIT → READY → RUNNING → PAUSED → SHUTDOWN。

    Args:
        config: Agent 配置字典。
        name: Agent 唯一标识名称。
    """

    # 合法状态转换表 (current_state -> {allowed_target_states})
    # 对应 AGENTS.md §4 生命周期图
    _TRANSITIONS: Dict[AgentState, set] = {
        AgentState.INIT: {AgentState.READY, AgentState.SHUTDOWN, AgentState.ERROR},
        AgentState.READY: {AgentState.RUNNING, AgentState.SHUTDOWN, AgentState.ERROR},
        AgentState.RUNNING: {AgentState.PAUSED, AgentState.SHUTDOWN, AgentState.ERROR},
        AgentState.PAUSED: {AgentState.RUNNING, AgentState.SHUTDOWN, AgentState.ERROR},
        AgentState.ERROR: {AgentState.INIT, AgentState.SHUTDOWN},
        AgentState.SHUTDOWN: set(),  # 终止状态，无出边
    }

    def __init__(self, config: Dict[str, Any], name: str) -> None:
        self.config: Dict[str, Any] = config
        self.name: str = name
        self.state: AgentState = AgentState.INIT
        self.step_stats: Dict[int, StepStats] = {}  # per-step 统计
        self.error_log: List[ErrorRecord] = []  # 错误日志
        self.performance_summary: Dict[str, float] = {}  # 性能汇总

    # ------------------------------------------------------------------
    # 抽象接口（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def on_init(self, model_config: Dict[str, Any]) -> None:
        """初始化 Agent，加载配置。

        对应 TECHNICAL_SPEC.md §3.2。

        Args:
            model_config: 模型级配置字典（如层数、hidden_dim 等）。
        """
        ...

    @abstractmethod
    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个 step。

        对应 TECHNICAL_SPEC.md §3.2。

        Args:
            step_config: 当前 step 的优化配置。
            inputs: 输入数据字典（内容因 Agent 角色而异）。

        Returns:
            输出数据字典。
        """
        ...

    @abstractmethod
    def on_shutdown(self) -> Optional[ProfileResult]:
        """关闭 Agent，返回汇总统计。

        对应 TECHNICAL_SPEC.md §3.2。

        Returns:
            可选的 ProfileResult 汇总对象。
        """
        ...

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    def get_status(self) -> AgentState:
        """返回当前状态。

        对应 AGENTS.md §5 AgentState.status。
        """
        return self.state

    def set_state(self, state: AgentState) -> None:
        """状态转换。

        实现 AGENTS.md §4 生命周期。
        直接设置状态，不检查转换合法性（用于内部恢复或强制状态）。

        Args:
            state: 目标状态。
        """
        self.state = state

    def transition_to(self, target: AgentState) -> bool:
        """合法状态转换检查。

        实现 INIT→READY→RUNNING→PAUSED→SHUTDOWN 的合法路径。
        若当前状态到目标状态的转换不合法，则返回 False 且状态不变。

        Args:
            target: 目标状态。

        Returns:
            True 表示转换成功，False 表示转换被拒绝。
        """
        allowed = self._TRANSITIONS.get(self.state, set())
        if target in allowed:
            self.state = target
            return True
        return False

    # ------------------------------------------------------------------
    # 统计与日志
    # ------------------------------------------------------------------

    def log_error(self, error: Exception, step_id: int = -1) -> None:
        """记录错误。

        将异常转换为 ErrorRecord 并追加到 error_log。

        Args:
            error: 捕获的异常对象。
            step_id: 发生错误的 step 编号，默认 -1 表示非 step 相关。
        """
        tb = traceback.format_exc()
        record = ErrorRecord.from_exception(error, step_id=step_id, traceback_str=tb)
        self.error_log.append(record)
        # 进入 ERROR 状态
        self.set_state(AgentState.ERROR)

    def record_step_stat(self, step_id: int, stats: Dict[str, Any]) -> None:
        """记录 per-step 统计。

        Args:
            step_id: Step 编号。
            stats: 统计字典，将用于构建 StepStats。
        """
        self.step_stats[step_id] = StepStats(
            step_id=step_id,
            latency_ms=stats.get("latency_ms", 0.0),
            memory_mb=stats.get("memory_mb", 0.0),
            flops=stats.get("flops", 0.0),
            accuracy_delta=stats.get("accuracy_delta", 0.0),
            kv_compression_ratio=stats.get("kv_compression_ratio", 0.0),
            ffn_sparse_ratio=stats.get("ffn_sparse_ratio", 0.0),
            custom_metrics=stats.get("custom_metrics", {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能汇总。

        从 step_stats 中自动计算聚合指标（如平均 latency、
        总 memory、平均压缩率等）。
        """
        if not self.step_stats:
            return self.performance_summary

        total_latency = sum(s.latency_ms for s in self.step_stats.values())
        avg_latency = total_latency / len(self.step_stats)
        avg_memory = sum(s.memory_mb for s in self.step_stats.values()) / len(self.step_stats)
        avg_kv_ratio = sum(s.kv_compression_ratio for s in self.step_stats.values()) / len(
            self.step_stats
        )
        avg_ffn_sparse = sum(s.ffn_sparse_ratio for s in self.step_stats.values()) / len(
            self.step_stats
        )

        self.performance_summary = {
            "total_steps": float(len(self.step_stats)),
            "total_latency_ms": total_latency,
            "avg_latency_ms": avg_latency,
            "avg_memory_mb": avg_memory,
            "avg_kv_compression_ratio": avg_kv_ratio,
            "avg_ffn_sparse_ratio": avg_ffn_sparse,
            "error_count": float(len(self.error_log)),
        }
        return self.performance_summary

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, state={self.state})"


class AgentRegistry:
    """Agent 注册表，用于 Orchestrator 管理所有 Agent 实例。

    提供按名称注册、获取、枚举和移除 Agent 的能力。
    线程安全由调用方（Orchestrator）通过外部锁保证。
    """

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """注册 Agent 实例。

        Args:
            agent: 要注册的 BaseAgent 子类实例。

        Raises:
            ValueError: 如果已存在同名 Agent。
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered.")
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """按名称获取 Agent 实例。

        Args:
            name: Agent 名称。

        Returns:
            Agent 实例，若不存在则返回 None。
        """
        return self._agents.get(name)

    def get_all(self) -> Dict[str, BaseAgent]:
        """获取所有已注册 Agent 的字典视图。"""
        return dict(self._agents)

    def remove(self, name: str) -> bool:
        """移除指定 Agent。

        Args:
            name: Agent 名称。

        Returns:
            True 表示成功移除，False 表示不存在。
        """
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def clear(self) -> None:
        """清空所有注册 Agent。"""
        self._agents.clear()

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __iter__(self):
        return iter(self._agents.values())

"""Profiling Bus — Agent 间通信总线。

对应文档：AGENTS.md §3.1 通信协议
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

import threading
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from .structs import Message


# ------------------------------------------------------------------
# 消息类型常量
# ------------------------------------------------------------------
class MESSAGE_TYPES:
    """Profiling Bus 消息类型常量。

    对应 AGENTS.md §3.1 通信协议中的 message_type 字段。
    """

    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    BROADCAST = "BROADCAST"
    ERROR = "ERROR"
    PROFILE_DATA = "PROFILE_DATA"
    OPTIM_CONFIG = "OPTIM_CONFIG"
    STEP_RESULT = "STEP_RESULT"
    DISPATCH_CONFIG = "DISPATCH_CONFIG"


class ProfilingBus:
    """Profiling Bus — Agent 间通信总线。

    对应 AGENTS.md §3.1 通信协议。
    提供点对点发送、广播、日志记录和统计能力。
    使用 threading.Lock 保证线程安全。

    Args:
        name: 总线实例名称，用于多总线场景区分。
    """

    def __init__(self, name: str = "default") -> None:
        self.name: str = name
        self._subscribers: Dict[str, Callable[[Message], None]] = {}
        self._message_log: List[Message] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 订阅管理
    # ------------------------------------------------------------------

    def subscribe(self, agent_name: str, callback: Callable[[Message], None]) -> None:
        """Agent 订阅消息。

        注册后，当该 Agent 作为 target 收到消息或广播时，
        callback 将被调用。

        Args:
            agent_name: Agent 唯一名称。
            callback: 消息回调函数，接收 Message 对象。

        Raises:
            ValueError: 如果 callback 不可调用。
        """
        if not callable(callback):
            raise ValueError("callback must be callable")
        with self._lock:
            self._subscribers[agent_name] = callback

    def unsubscribe(self, agent_name: str) -> None:
        """取消订阅。

        Args:
            agent_name: Agent 唯一名称。
        """
        with self._lock:
            self._subscribers.pop(agent_name, None)

    # ------------------------------------------------------------------
    # 消息发送
    # ------------------------------------------------------------------

    def send(self, message: Message) -> None:
        """发送点对点消息。

        将消息投递给 target 指定的 Agent（若已订阅）。
        同时记录消息到日志。

        Args:
            message: 要发送的 Message 对象。
        """
        callback: Optional[Callable[[Message], None]] = None
        with self._lock:
            self._message_log.append(message)
            target = message.target
            if target in self._subscribers:
                callback = self._subscribers[target]
        # 在锁外执行回调，避免长时间持有锁
        if callback is not None:
            try:
                callback(message)
            except Exception as exc:
                # 回调异常不应破坏总线，记录为内部错误
                self._log_callback_error(message, exc)

    def broadcast(self, message: Message) -> None:
        """广播消息到所有订阅者。

        将消息投递给所有已订阅的 Agent（target 应为 'all'）。
        同时记录消息到日志。

        Args:
            message: 要广播的 Message 对象。
        """
        callbacks: Dict[str, Callable[[Message], None]] = {}
        with self._lock:
            self._message_log.append(message)
            callbacks = dict(self._subscribers)

        # 在锁外执行所有回调
        errors: List[Exception] = []
        for agent_name, callback in callbacks.items():
            try:
                callback(message)
            except Exception as exc:
                errors.append(exc)
                self._log_callback_error(message, exc, agent_name)

        # 如果有错误，记录一条汇总错误消息到日志
        if errors:
            error_msg = Message.create(
                source=self.name,
                target="orchestrator",
                message_type=MESSAGE_TYPES.ERROR,
                payload={
                    "broadcast_errors": len(errors),
                    "original_message_id": message.message_id,
                },
                step_id=message.step_id,
            )
            with self._lock:
                self._message_log.append(error_msg)

    # ------------------------------------------------------------------
    # 日志与统计
    # ------------------------------------------------------------------

    def get_message_log(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> List[Message]:
        """获取消息日志，支持过滤。

        Args:
            source: 仅返回来自该 source 的消息。
            target: 仅返回发往该 target 的消息。
            message_type: 仅返回该类型的消息。

        Returns:
            符合条件的 Message 列表。
        """
        with self._lock:
            logs = list(self._message_log)

        filtered = []
        for msg in logs:
            if source is not None and msg.source != source:
                continue
            if target is not None and msg.target != target:
                continue
            if message_type is not None and msg.message_type != message_type:
                continue
            filtered.append(msg)
        return filtered

    def clear_log(self) -> None:
        """清空日志。"""
        with self._lock:
            self._message_log.clear()

    def get_message_stats(self) -> Dict[str, int]:
        """统计消息类型分布。

        Returns:
            字典：message_type -> 计数。
        """
        stats: Dict[str, int] = defaultdict(int)
        with self._lock:
            logs = list(self._message_log)
        for msg in logs:
            stats[msg.message_type] += 1
        return dict(stats)

    def get_subscriber_count(self) -> int:
        """获取当前订阅者数量。"""
        with self._lock:
            return len(self._subscribers)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _log_callback_error(
        self,
        original_message: Message,
        exc: Exception,
        agent_name: Optional[str] = None,
    ) -> None:
        """记录回调执行错误。"""
        error_msg = Message.create(
            source=self.name,
            target=agent_name or "orchestrator",
            message_type=MESSAGE_TYPES.ERROR,
            payload={
                "error": str(exc),
                "original_message_id": original_message.message_id,
                "agent_name": agent_name,
            },
            step_id=original_message.step_id,
        )
        with self._lock:
            self._message_log.append(error_msg)

    def __repr__(self) -> str:
        with self._lock:
            subs = list(self._subscribers.keys())
            log_len = len(self._message_log)
        return f"ProfilingBus(name={self.name}, subscribers={subs}, log_len={log_len})"

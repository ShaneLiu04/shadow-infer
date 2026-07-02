"""
ShadowInfer Engineering: Hot Config Reloader
===========================================

Hot-reload configuration without restarting the inference pipeline.
Critical for production AI Infra deployments where downtime is unacceptable.

Features:
- File watcher with debouncing (prevent reload storms)
- Config validation before application
- Graceful degradation on invalid config
- Atomic config swap (no partial state)
- Rollback capability on validation failure

Target: Big Tech AI Infra recruiting (production-grade reliability)
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from shadowinfer.core.config import Config

logger = logging.getLogger(__name__)


class ReloadStatus(Enum):
    """配置重载状态"""

    IDLE = "idle"  # 未启用热重载
    WATCHING = "watching"  # 监听中
    RELOADING = "reloading"  # 正在重载
    FAILED = "failed"  # 上次重载失败
    DEGRADED = "degraded"  # 降级模式（使用旧配置）


class ConfigValidationError(Exception):
    """配置验证失败异常"""


@dataclass
class ReloadEvent:
    """配置重载事件记录"""

    timestamp: float
    old_config_hash: str
    new_config_hash: str
    status: ReloadStatus
    error_message: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)


class HotConfigReloader:
    """
    热配置重载器。

    支持：
    - 文件变更监听（轮询或事件驱动）
    - 配置验证（schema + 业务规则）
    - 原子切换（无中间状态）
    - 失败回滚（保留旧配置）
    - 事件回调（通知所有 agent）

    Usage:
        reloader = HotConfigReloader(config_path="configs/optimize_full.yaml")
        reloader.start_watching(interval_sec=1.0)

        # 注册配置变更回调
        reloader.on_config_change(agent.on_config_update)

        # 手动触发重载
        reloader.force_reload()

        # 停止监听
        reloader.stop_watching()
    """

    def __init__(self, config_path: str, validator: Optional[Callable[[Dict], List[str]]] = None):
        self.config_path = config_path
        self.validator = validator or self._default_validator

        self._current_config: Optional[Config] = None
        self._current_config_hash: str = ""
        self._current_raw: Dict[str, Any] = {}

        self._status = ReloadStatus.IDLE
        self._events: List[ReloadEvent] = []
        self._callbacks: List[Callable[[Config, Config], None]] = []

        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop_event = threading.Event()
        self._watch_interval_sec = 1.0
        self._debounce_sec = 0.5
        self._last_change_time = 0.0

        self._lock = threading.RLock()

        # 初始化加载
        self._load_initial()

    def _load_initial(self) -> None:
        """初始化加载配置"""
        try:
            self._current_config = Config.from_yaml(self.config_path)
            self._current_raw = (
                self._current_config.to_dict() if hasattr(self._current_config, "to_dict") else {}
            )
            self._current_config_hash = self._compute_hash(self._current_raw)
            self._status = ReloadStatus.IDLE
            logger.info(
                f"Initial config loaded: {self.config_path} (hash={self._current_config_hash[:8]})"
            )
        except Exception as e:
            self._status = ReloadStatus.FAILED
            raise ConfigValidationError(f"Failed to load initial config: {e}")

    @staticmethod
    def _compute_hash(config_dict: Dict) -> str:
        """计算配置字典的哈希值"""

        def _default_serializer(obj):
            if isinstance(obj, Config):
                return obj.to_dict()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        config_str = json.dumps(config_dict, sort_keys=True, default=_default_serializer)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _default_validator(self, config_dict: Dict) -> List[str]:
        """
        默认配置验证器。

        验证规则：
        1. 必须包含 optimization 字段
        2. shadowkv.compression_ratio 必须在 (0, 1] 范围内
        3. qdrift.sensitivity_temperature 必须 > 0
        4. ffn.delta_threshold 必须在 [0, 1] 范围内
        5. constraints.max_accuracy_drop 必须 >= 0
        """
        errors = []

        if not isinstance(config_dict, dict):
            errors.append("Config must be a dictionary")
            return errors

        # 规则 1: 必须包含 optimization 字段
        if "optimization" not in config_dict:
            errors.append("Missing required field: 'optimization'")

        # 规则 2: ShadowKV 压缩比范围
        shadowkv = config_dict.get("shadowkv", {})
        compression_ratio = shadowkv.get("compression_ratio")
        if compression_ratio is not None:
            if not (0 < compression_ratio <= 1.0):
                errors.append(
                    f"shadowkv.compression_ratio must be in (0, 1], got {compression_ratio}"
                )

        # 规则 3: Q-drift 温度必须 > 0
        qdrift = config_dict.get("qdrift", {})
        temp = qdrift.get("sensitivity_temperature")
        if temp is not None:
            if temp <= 0:
                errors.append(f"qdrift.sensitivity_temperature must be > 0, got {temp}")

        # 规则 4: FFN delta_threshold 范围
        ffn = config_dict.get("ffn", {})
        delta_threshold = ffn.get("delta_threshold")
        if delta_threshold is not None:
            if not (0 <= delta_threshold <= 1):
                errors.append(f"ffn.delta_threshold must be in [0, 1], got {delta_threshold}")

        # 规则 5: 最大精度损失必须 >= 0
        constraints = config_dict.get("constraints", {})
        max_drop = constraints.get("max_accuracy_drop")
        if max_drop is not None:
            if max_drop < 0:
                errors.append(f"constraints.max_accuracy_drop must be >= 0, got {max_drop}")

        # 规则 6: 精度级别必须是合法值
        precision_levels = shadowkv.get("precision_levels", [])
        valid_precisions = {"fp32", "fp16", "int8", "int4", "bf16"}
        for p in precision_levels:
            if p not in valid_precisions:
                errors.append(f"Invalid precision level: '{p}', valid: {valid_precisions}")

        # 规则 7: 噪声调度必须是合法值
        noise_schedule = qdrift.get("noise_schedule")
        valid_schedules = {"linear", "cosine", "sigmoid"}
        if noise_schedule is not None and noise_schedule not in valid_schedules:
            errors.append(f"Invalid noise_schedule: '{noise_schedule}', valid: {valid_schedules}")

        return errors

    def validate_config(self, config_dict: Dict) -> List[str]:
        """
        验证配置字典。

        Returns:
            空列表表示验证通过，非空列表包含错误信息
        """
        return self.validator(config_dict)

    def _check_file_changed(self) -> Optional[Dict[str, Any]]:
        """检查文件是否变更，如变更则返回新配置"""
        try:
            if not os.path.exists(self.config_path):
                return None

            # 检查修改时间（快速路径）
            mtime = os.path.getmtime(self.config_path)
            if mtime <= self._last_change_time + self._debounce_sec:
                return None

            # 加载新配置
            new_config = Config.from_yaml(self.config_path)
            new_raw = new_config.to_dict() if hasattr(new_config, "to_dict") else {}
            new_hash = self._compute_hash(new_raw)

            # 哈希对比（避免无意义重载）
            if new_hash == self._current_config_hash:
                return None

            self._last_change_time = time.time()
            return new_raw

        except Exception as e:
            logger.warning(f"Error checking config file: {e}")
            return None

    def _reload_config(self, new_raw: Dict[str, Any]) -> bool:
        """
        执行配置重载。

        流程：
        1. 验证新配置
        2. 如果验证失败：记录失败，保留旧配置，触发降级
        3. 如果验证通过：原子切换，通知回调

        Returns:
            True if reload succeeded, False otherwise
        """
        with self._lock:
            self._status = ReloadStatus.RELOADING

        # 1. 验证新配置
        validation_errors = self.validate_config(new_raw)
        if validation_errors:
            event = ReloadEvent(
                timestamp=time.time(),
                old_config_hash=self._current_config_hash,
                new_config_hash=self._compute_hash(new_raw),
                status=ReloadStatus.FAILED,
                error_message="Config validation failed",
                validation_errors=validation_errors,
            )
            with self._lock:
                self._events.append(event)
                self._status = ReloadStatus.DEGRADED

            logger.error(f"Config reload failed: {validation_errors}")
            return False

        # 2. 原子切换
        old_config = self._current_config
        old_hash = self._current_config_hash

        try:
            new_config = (
                Config.from_dict(new_raw) if hasattr(Config, "from_dict") else Config(new_raw)
            )
            new_hash = self._compute_hash(new_raw)

            with self._lock:
                self._current_config = new_config
                self._current_raw = new_raw
                self._current_config_hash = new_hash
                self._status = ReloadStatus.WATCHING

            # 3. 记录成功事件
            event = ReloadEvent(
                timestamp=time.time(),
                old_config_hash=old_hash,
                new_config_hash=new_hash,
                status=ReloadStatus.WATCHING,
            )
            with self._lock:
                self._events.append(event)

            # 4. 通知回调
            self._notify_callbacks(old_config, new_config)

            logger.info(f"Config reloaded successfully: {old_hash[:8]} -> {new_hash[:8]}")
            return True

        except Exception as e:
            event = ReloadEvent(
                timestamp=time.time(),
                old_config_hash=old_hash,
                new_config_hash=self._compute_hash(new_raw),
                status=ReloadStatus.FAILED,
                error_message=str(e),
            )
            with self._lock:
                self._events.append(event)
                self._status = ReloadStatus.DEGRADED

            logger.error(f"Config reload failed during swap: {e}")
            return False

    def _notify_callbacks(self, old_config: Config, new_config: Config) -> None:
        """通知所有注册的回调函数"""
        for callback in self._callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.warning(f"Config change callback failed: {e}")

    def on_config_change(self, callback: Callable[[Config, Config], None]) -> None:
        """注册配置变更回调"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Config, Config], None]) -> None:
        """移除配置变更回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start_watching(self, interval_sec: float = 1.0, debounce_sec: float = 0.5) -> None:
        """
        启动文件监听。

        Args:
            interval_sec: 轮询间隔（秒）
            debounce_sec: 防抖时间（秒），防止文件写入中的中间状态触发重载
        """
        if self._watch_thread is not None and self._watch_thread.is_alive():
            logger.warning("Watcher already running")
            return

        self._watch_interval_sec = interval_sec
        self._debounce_sec = debounce_sec
        self._watch_stop_event.clear()
        self._status = ReloadStatus.WATCHING

        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()

        logger.info(f"Config watcher started: {self.config_path} (interval={interval_sec}s)")

    def _watch_loop(self) -> None:
        """监听循环"""
        while not self._watch_stop_event.is_set():
            try:
                new_raw = self._check_file_changed()
                if new_raw is not None:
                    self._reload_config(new_raw)
            except Exception as e:
                logger.error(f"Watcher error: {e}")

            self._watch_stop_event.wait(self._watch_interval_sec)

    def stop_watching(self) -> None:
        """停止文件监听"""
        self._watch_stop_event.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=2.0)
            self._watch_thread = None
        self._status = ReloadStatus.IDLE
        logger.info("Config watcher stopped")

    def force_reload(self) -> bool:
        """强制重载配置（即使文件未变更）"""
        try:
            new_config = Config.from_yaml(self.config_path)
            new_raw = new_config.to_dict() if hasattr(new_config, "to_dict") else {}
            return self._reload_config(new_raw)
        except Exception as e:
            logger.error(f"Force reload failed: {e}")
            return False

    def get_current_config(self) -> Config:
        """获取当前配置"""
        with self._lock:
            return self._current_config

    def get_status(self) -> ReloadStatus:
        """获取当前状态"""
        return self._status

    def get_events(self, limit: int = 10) -> List[ReloadEvent]:
        """获取最近的重载事件"""
        with self._lock:
            return self._events[-limit:]

    def get_config_hash(self) -> str:
        """获取当前配置哈希"""
        with self._lock:
            return self._current_config_hash


class AgentConfigAdapter:
    """
    Agent 配置适配器。

    将全局配置转换为各 Agent 的专用配置，
    并在配置变更时自动更新 Agent 参数。
    """

    @staticmethod
    def extract_shadowkv_config(global_config: Config) -> Dict[str, Any]:
        """提取 ShadowKV Agent 配置"""
        return {
            "enabled": global_config.get("shadowkv.enabled", True),
            "compression_ratio": global_config.get("shadowkv.compression_ratio", 0.5),
            "precision_levels": global_config.get(
                "shadowkv.precision_levels", ["fp32", "fp16", "int8", "int4"]
            ),
            "importance_thresholds": global_config.get("shadowkv.importance_thresholds", {}),
            "reuse": global_config.get("shadowkv.reuse", {}),
        }

    @staticmethod
    def extract_qdrift_config(global_config: Config) -> Dict[str, Any]:
        """提取 QDrift Agent 配置"""
        return {
            "enabled": global_config.get("qdrift.enabled", True),
            "noise_schedule": global_config.get("qdrift.noise_schedule", "cosine"),
            "sensitivity_temperature": global_config.get("qdrift.sensitivity_temperature", 1.0),
            "drift_method": global_config.get("qdrift.drift_method", "relative_l2"),
            "dispatch_matrix": global_config.get("qdrift.dispatch_matrix", {}),
        }

    @staticmethod
    def extract_ffn_config(global_config: Config) -> Dict[str, Any]:
        """提取 FFN Optimizer Agent 配置"""
        return {
            "enabled": global_config.get("ffn.enabled", True),
            "mixed_precision": global_config.get("ffn.mixed_precision", True),
            "channel_importance_threshold": global_config.get(
                "ffn.channel_importance_threshold", 0.7
            ),
            "sparse_update": global_config.get("ffn.sparse_update", True),
            "delta_threshold": global_config.get("ffn.delta_threshold", 0.05),
        }

    @staticmethod
    def extract_constraints(global_config: Config) -> Dict[str, Any]:
        """提取约束配置"""
        return {
            "max_accuracy_drop": global_config.get("constraints.max_accuracy_drop", 0.01),
            "max_latency_ms": global_config.get("constraints.max_latency_ms", 100),
            "max_memory_mb": global_config.get("constraints.max_memory_mb", 8192),
        }

"""ShadowInfer 结构化日志系统。

基于 ``structlog`` 实现，支持：

* 统一的 JSON 行日志输出（控制台 + 文件）
* 按大小/时间自动轮转（RotatingFileHandler / TimedRotatingFileHandler）
* 运行时动态调整日志级别
* 全局 logger 注册表，一键修改所有 ShadowInfer logger 级别
* 向后兼容原 ``StructuredLogger`` 的 ``log_event`` / ``log_metric`` /
  ``log_alert`` / ``get_logs`` / ``export_json`` / ``export_csv`` API

对应文档：ENGINEERING.md §可观测性、ROADMAP.md §Structured Logging
"""

from __future__ import annotations

__version__ = "3.1"

import csv
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import structlog

# Global registry of StructuredLogger instances created in this process.
# Used by configure_shadowinfer_logging() to update levels/rotation centrally.
_LOGGERS: Dict[str, "StructuredLogger"] = {}

# Global defaults applied to new StructuredLogger instances when the caller
# does not explicitly provide a level or rotation policy.
_GLOBAL_LEVEL: Optional[int] = None
_GLOBAL_ROTATION: Optional[Dict[str, Any]] = None


def _log_level_to_int(level: Union[int, str]) -> int:
    """Convert a logging level (int or string) to its integer value."""
    if isinstance(level, int):
        return level
    # Allow case-insensitive level names.
    normalized = str(level).upper()
    value = logging.getLevelName(normalized)
    if isinstance(value, int):
        return value
    # Fallback to INFO for unknown strings.
    return logging.INFO


def _json_default(obj: Any) -> str:
    """Fallback serializer for non-JSON-native objects."""
    return str(obj)


class StructuredLogger:
    """结构化日志 facade。

    内部使用 ``structlog`` 绑定上下文并输出 JSON，同时保留原始
    ``logging.StreamHandler`` / ``logging.FileHandler`` 以兼容下游工具。

    Args:
        name: logger 名称，会作为 ``logger`` 字段写入每条日志。
        log_dir: 日志文件目录。
        level: 初始日志级别，可以是字符串（"DEBUG"/"INFO"/...）或整数。
        rotation: 轮转策略，可选 ``{"max_bytes": int, "backup_count": int}``
            或 ``{"when": str, "backup_count": int}``。为空则使用每日文件。
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "logs/",
        level: Optional[Union[int, str]] = None,
        rotation: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.log_dir = log_dir
        self.level = _log_level_to_int(
            level
            if level is not None
            else (_GLOBAL_LEVEL if _GLOBAL_LEVEL is not None else logging.INFO)
        )
        self.rotation = (
            rotation
            if rotation is not None
            else (_GLOBAL_ROTATION if _GLOBAL_ROTATION is not None else {})
        )
        self._records: List[Dict[str, Any]] = []

        os.makedirs(log_dir, exist_ok=True)

        # Configure stdlib logger.
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()
        self.logger.propagate = False

        # Console handler: JSON lines.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

        # File handler with optional rotation.
        self._file_handler = self._create_file_handler()
        self.logger.addHandler(self._file_handler)

        # Configure structlog to use the same JSON renderer.
        # We keep configuration local per-logger so multiple StructuredLogger
        # instances can coexist without clobbering global state.
        self._struct_logger = structlog.wrap_logger(
            self.logger,
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(serializer=json.dumps, default=_json_default),
            ],
        )

        # Register in global registry.
        _LOGGERS[name] = self

    # ------------------------------------------------------------------
    # Rotation & level control
    # ------------------------------------------------------------------

    def _create_file_handler(self) -> logging.Handler:
        """Create the file handler according to the configured rotation policy."""
        log_file = os.path.join(
            self.log_dir, f"{self.name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
        )

        if "max_bytes" in self.rotation:
            handler: logging.Handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=int(self.rotation["max_bytes"]),
                backupCount=int(self.rotation.get("backup_count", 5)),
                encoding="utf-8",
            )
        elif "when" in self.rotation:
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when=str(self.rotation["when"]),
                backupCount=int(self.rotation.get("backup_count", 7)),
                encoding="utf-8",
                utc=True,
            )
        else:
            handler = logging.FileHandler(log_file, encoding="utf-8")

        handler.setLevel(self.level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    def set_level(self, level: Union[int, str]) -> "StructuredLogger":
        """运行时动态设置日志级别。

        Args:
            level: 新日志级别，字符串或整数。

        Returns:
            self，支持链式调用。
        """
        self.level = _log_level_to_int(level)
        self.logger.setLevel(self.level)
        for handler in self.logger.handlers:
            handler.setLevel(self.level)
        return self

    def get_level(self) -> int:
        """返回当前日志级别的整数值。"""
        return self.level

    def set_rotation(self, rotation: Optional[Dict[str, Any]]) -> "StructuredLogger":
        """重新配置日志文件轮转策略。

        Args:
            rotation: 轮转配置字典。``{"max_bytes": int, "backup_count": int}``
                开启按大小轮转；``{"when": str, "backup_count": int}`` 开启
                按时间轮转；``None`` 取消轮转。

        Returns:
            self，支持链式调用。
        """
        self.rotation = dict(rotation) if rotation else {}
        # Replace existing file handler.
        new_handler = self._create_file_handler()
        for handler in list(self.logger.handlers):
            if isinstance(
                handler,
                (
                    logging.FileHandler,
                    logging.handlers.RotatingFileHandler,
                    logging.handlers.TimedRotatingFileHandler,
                ),
            ) and not isinstance(handler, logging.StreamHandler):
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
        self.logger.addHandler(new_handler)
        self._file_handler = new_handler
        return self

    def flush(self) -> None:
        """强制刷新所有 handler 缓冲区。"""
        for handler in self.logger.handlers:
            handler.flush()

    def close(self) -> None:
        """关闭所有 handler 并注销全局注册表中的记录。

        主要用于测试环境清理，避免日志文件句柄占用导致临时目录无法删除。
        """
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
        _LOGGERS.pop(self.name, None)

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_metric(
        self,
        metric_name: str,
        value: float,
        step_id: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录一个指标。

        Args:
            metric_name: 指标名称。
            value: 指标值。
            step_id: 可选 step id。
            tags: 可选标签字典。
        """
        record = {
            "event": "metric",
            "metric_name": metric_name,
            "value": value,
            "step_id": step_id,
            "tags": tags or {},
        }
        self._emit(record, level=logging.INFO)

    def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        step_id: Optional[int] = None,
    ) -> None:
        """记录一个事件。

        Args:
            event_type: 事件类型，如 ``step_start`` / ``agent_init``。
            message: 事件描述。
            data: 可选附加数据。
            step_id: 可选 step id。
        """
        record = {
            "event": event_type,
            "message": message,
            "step_id": step_id,
            "data": data or {},
        }
        self._emit(record, level=logging.INFO)

    def log_alert(
        self,
        level: str,
        message: str,
        recommendation: Optional[str] = None,
        step_id: Optional[int] = None,
    ) -> None:
        """记录告警。

        Args:
            level: 告警级别，如 ``critical`` / ``error`` / ``warning`` / ``info``。
            message: 告警内容。
            recommendation: 可选修复建议。
            step_id: 可选 step id。
        """
        level_no = _log_level_to_int(str(level).upper())
        record = {
            "event": "alert",
            "alert_level": level,
            "message": message,
            "recommendation": recommendation,
            "step_id": step_id,
        }
        self._emit(record, level=level_no)

    def _emit(self, record: Dict[str, Any], level: int = logging.INFO) -> None:
        """Internal: emit a JSON log record through structlog and cache it."""
        if level < self.level:
            return
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["logger"] = self.name
        record["level_no"] = level
        self._records.append(record)
        # Use structlog to produce the JSON line.
        event = record.get("event", "log")
        kwargs = {k: v for k, v in record.items() if k != "event"}
        self._struct_logger.log(level, event, **kwargs)

    # ------------------------------------------------------------------
    # Retrieval & export
    # ------------------------------------------------------------------

    def get_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """检索缓存中的日志记录。

        Args:
            start_time: 可选 ISO 格式起始时间。
            end_time: 可选 ISO 格式结束时间。
            level: 可选过滤级别（匹配 ``record["level"]`` 或 ``alert_level``）。

        Returns:
            符合条件的日志记录列表。
        """
        filtered = self._records

        if start_time:
            filtered = [r for r in filtered if r.get("timestamp", "") >= start_time]
        if end_time:
            filtered = [r for r in filtered if r.get("timestamp", "") <= end_time]
        if level:
            normalized = level.upper()
            filtered = [
                r
                for r in filtered
                if r.get("level") == normalized or r.get("alert_level", "").upper() == normalized
            ]

        return filtered

    def export_json(self, filepath: str) -> None:
        """导出所有缓存的日志到 JSON 文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2, default=_json_default)

    def export_csv(self, filepath: str, metric_name: Optional[str] = None) -> None:
        """导出指标记录到 CSV 文件。

        Args:
            filepath: 输出文件路径。
            metric_name: 可选指标名称过滤，只导出该指标。
        """
        metric_records = [r for r in self._records if r.get("event") == "metric"]
        if metric_name:
            metric_records = [r for r in metric_records if r.get("metric_name") == metric_name]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "metric_name", "value", "step_id", "tags"])
            for r in metric_records:
                writer.writerow(
                    [
                        r["timestamp"],
                        r.get("metric_name", ""),
                        r.get("value", ""),
                        r.get("step_id", ""),
                        json.dumps(r.get("tags", {}), ensure_ascii=False, default=_json_default),
                    ]
                )


# --------------------------------------------------------------------------
# Global configuration helpers
# --------------------------------------------------------------------------


def configure_shadowinfer_logging(
    level: Optional[Union[int, str]] = None,
    rotation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """全局配置 ShadowInfer 内部所有 ``StructuredLogger`` 实例。

    通常在 Orchestrator 初始化或热配置重载时调用。

    Args:
        level: 新的全局日志级别。若未指定则保持各 logger 当前级别。
        rotation: 新的全局轮转策略。若未指定则保持各 logger 当前策略。

    Returns:
        更新统计字典，包含 ``updated_loggers`` 列表与应用的配置。
    """
    global _GLOBAL_LEVEL, _GLOBAL_ROTATION
    if level is not None:
        _GLOBAL_LEVEL = _log_level_to_int(level)
    if rotation is not None:
        _GLOBAL_ROTATION = dict(rotation)

    updated: List[str] = []
    for logger in _LOGGERS.values():
        if level is not None:
            logger.set_level(level)
        if rotation is not None:
            logger.set_rotation(rotation)
        updated.append(logger.name)

    return {
        "updated_loggers": updated,
        "level": _GLOBAL_LEVEL,
        "rotation": _GLOBAL_ROTATION,
    }


def get_structured_loggers() -> Dict[str, StructuredLogger]:
    """Return a snapshot of registered StructuredLogger instances."""
    return dict(_LOGGERS)


def set_default_shadowinfer_processors() -> None:
    """Configure structlog global defaults for external / non-StructuredLogger code.

    This is optional; ``StructuredLogger`` instances use their own processors.
    Calling this makes plain ``structlog.get_logger()`` calls produce JSON lines
    consistent with ShadowInfer logs.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=json.dumps, default=_json_default),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

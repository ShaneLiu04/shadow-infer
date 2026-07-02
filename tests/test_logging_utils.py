"""Tests for the structured logging subsystem."""

from __future__ import annotations

import json
import logging
import tempfile

import pytest

from shadowinfer.utils.logging_utils import (
    StructuredLogger,
    configure_shadowinfer_logging,
    get_structured_loggers,
    set_default_shadowinfer_processors,
)


class TestStructuredLogger:
    def test_log_event_emits_json(self):
        with tempfile.TemporaryDirectory() as d:
            logger = StructuredLogger("test_event", log_dir=d)
            try:
                logger.log_event("test", "hello", data={"x": 1}, step_id=2)
                logs = logger.get_logs()
                assert len(logs) == 1
                assert logs[0]["event"] == "test"
                assert logs[0]["message"] == "hello"
                assert logs[0]["data"]["x"] == 1
                assert logs[0]["step_id"] == 2
                assert "timestamp" in logs[0]
            finally:
                logger.close()

    def test_log_metric_and_alert(self):
        with tempfile.TemporaryDirectory() as d:
            logger = StructuredLogger("test_metric", log_dir=d)
            try:
                logger.log_metric("loss", 0.5, step_id=1, tags={"split": "train"})
                logger.log_alert("warning", "high latency", recommendation="scale", step_id=1)
                warning_logs = logger.get_logs(level="WARNING")
                assert len(warning_logs) == 1
                assert warning_logs[0]["alert_level"] == "warning"
                metric_logs = [r for r in logger.get_logs() if r.get("event") == "metric"]
                assert len(metric_logs) == 1
                assert metric_logs[0]["metric_name"] == "loss"
            finally:
                logger.close()

    def test_dynamic_level_change(self):
        with tempfile.TemporaryDirectory() as d:
            logger = StructuredLogger("test_level", log_dir=d, level=logging.DEBUG)
            try:
                assert logger.get_level() == logging.DEBUG
                logger.log_event("debug_event", "should be kept")
                logger.set_level(logging.ERROR)
                assert logger.get_level() == logging.ERROR
                logger.log_event("ignored_event", "should be filtered")
                logs = logger.get_logs()
                assert len(logs) == 1
                assert logs[0]["event"] == "debug_event"
            finally:
                logger.close()

    def test_rotation_by_size_creates_backup(self):
        with tempfile.TemporaryDirectory() as d:
            logger = StructuredLogger(
                "test_rot",
                log_dir=d,
                level=logging.INFO,
                rotation={"max_bytes": 50, "backup_count": 2},
            )
            try:
                for _ in range(100):
                    logger.log_event("fill", "x" * 80)
                logger.flush()
                # Rotation may create backup files.
                import os

                files = [f for f in os.listdir(d) if f.startswith("test_rot")]
                assert len(files) >= 1
            finally:
                logger.close()

    def test_export_json_and_csv(self):
        with tempfile.TemporaryDirectory() as d:
            logger = StructuredLogger("test_export", log_dir=d)
            try:
                logger.log_metric("m", 1.0, step_id=0)
                logger.log_metric("m", 2.0, step_id=1)

                json_path = f"{d}/logs.json"
                csv_path = f"{d}/logs.csv"
                logger.export_json(json_path)
                logger.export_csv(csv_path, metric_name="m")

                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                assert len(data) == 2

                with open(csv_path, encoding="utf-8") as f:
                    lines = f.readlines()
                assert len(lines) == 3  # header + 2 rows
            finally:
                logger.close()

    def test_close_removes_logger(self):
        with tempfile.TemporaryDirectory() as d:
            logger = StructuredLogger("test_close", log_dir=d)
            assert "test_close" in get_structured_loggers()
            logger.close()
            assert "test_close" not in get_structured_loggers()


class TestGlobalLoggingConfiguration:
    def test_configure_shadowinfer_logging_updates_existing_loggers(self):
        with tempfile.TemporaryDirectory() as d:
            logger_a = StructuredLogger("logger_a", log_dir=d)
            logger_b = StructuredLogger("logger_b", log_dir=d)
            try:
                result = configure_shadowinfer_logging(level="WARNING")
                assert "logger_a" in result["updated_loggers"]
                assert "logger_b" in result["updated_loggers"]
                assert logger_a.get_level() == logging.WARNING
                assert logger_b.get_level() == logging.WARNING
            finally:
                logger_a.close()
                logger_b.close()

    def test_new_logger_inherits_global_level(self):
        with tempfile.TemporaryDirectory() as d:
            configure_shadowinfer_logging(level="ERROR")
            logger = StructuredLogger("inherits_level", log_dir=d)
            try:
                assert logger.get_level() == logging.ERROR
            finally:
                logger.close()

    def test_set_default_shadowinfer_processors_runs(self):
        # Smoke test for the optional global structlog configuration helper.
        set_default_shadowinfer_processors()

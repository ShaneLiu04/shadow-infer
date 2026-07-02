"""Tests for the declarative Policy DSL.

Covers rule matching, priority ordering, default values, validation, and file
loading.
"""

import json
from pathlib import Path

import pytest

from shadowinfer.core.policy import PolicyContext, PolicyEngine, load_policy


def _sample_policy() -> dict:
    return {
        "version": "1.0",
        "defaults": {
            "thresholds": {"high": 0.7, "low": 0.3},
            "mode": "balanced",
        },
        "rules": [
            {
                "name": "accuracy-critical",
                "priority": 100,
                "conditions": [{"field": "accuracy_drop", "op": ">=", "value": 0.01}],
                "actions": [
                    {"target": "shadowkv.mode", "value": "conservative"},
                    {"target": "ffn.mode", "value": "full"},
                ],
            },
            {
                "name": "memory-pressure",
                "priority": 50,
                "conditions": [{"field": "memory_ratio", "op": ">=", "value": 0.9}],
                "actions": [
                    {"target": "shadowkv.mode", "value": "aggressive"},
                ],
            },
            {
                "name": "latency-pressure",
                "priority": 60,
                "conditions": [{"field": "latency_ratio", "op": ">=", "value": 1.5}],
                "actions": [
                    {"target": "ffn.mode", "value": "sparse"},
                ],
            },
        ],
    }


def test_load_from_dict():
    """PolicyEngine.load must accept a dictionary."""
    engine = PolicyEngine.load(_sample_policy())
    assert len(engine.rules) == 3
    assert engine.get("thresholds.high") == 0.7


def test_load_from_yaml(tmp_path):
    """PolicyEngine.load must accept a YAML file."""
    path = tmp_path / "policy.yaml"
    path.write_text(
        "version: '1.0'\ndefaults:\n  threshold: 0.5\nrules: []\n",
        encoding="utf-8",
    )
    engine = PolicyEngine.load(str(path))
    assert engine.get("threshold") == 0.5


def test_load_from_json(tmp_path):
    """PolicyEngine.load must accept a JSON file."""
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_sample_policy()), encoding="utf-8")
    engine = PolicyEngine.load(str(path))
    assert len(engine.rules) == 3


def test_load_missing_file_raises():
    """Loading a missing file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PolicyEngine.load("/nonexistent/policy.yaml")


def test_load_unsupported_format_raises(tmp_path):
    """Loading an unsupported extension must raise ValueError."""
    path = tmp_path / "policy.txt"
    path.write_text("version: '1.0'", encoding="utf-8")
    with pytest.raises(ValueError):
        PolicyEngine.load(str(path))


def test_evaluate_accuracy_critical():
    """A critical accuracy drop must trigger conservative modes."""
    engine = PolicyEngine.load(_sample_policy())
    ctx = PolicyContext({"accuracy_drop": 0.015})
    actions = engine.evaluate(ctx)
    assert actions["shadowkv.mode"] == "conservative"
    assert actions["ffn.mode"] == "full"


def test_evaluate_priority_wins():
    """Higher-priority rules override lower-priority rules for the same target."""
    engine = PolicyEngine.load(_sample_policy())
    ctx = PolicyContext({"accuracy_drop": 0.015, "memory_ratio": 0.95})
    actions = engine.evaluate(ctx)
    # accuracy-critical has priority 100, memory-pressure 50 -> conservative wins.
    assert actions["shadowkv.mode"] == "conservative"


def test_evaluate_latency_pressure():
    """Latency pressure must switch FFN to sparse."""
    engine = PolicyEngine.load(_sample_policy())
    ctx = PolicyContext({"latency_ratio": 1.8})
    actions = engine.evaluate(ctx)
    assert actions["ffn.mode"] == "sparse"


def test_evaluate_no_match():
    """When no rule matches, evaluate returns an empty action dict."""
    engine = PolicyEngine.load(_sample_policy())
    ctx = PolicyContext({})
    assert engine.evaluate(ctx) == {}


def test_evaluate_with_defaults():
    """evaluate_with_defaults must overlay rule actions onto defaults."""
    engine = PolicyEngine.load(_sample_policy())
    ctx = PolicyContext({"latency_ratio": 1.8})
    result = engine.evaluate_with_defaults(ctx)
    assert result.get("mode") == "balanced"
    assert result.get("thresholds.high") == 0.7
    assert result.get("ffn.mode") == "sparse"


def test_context_dotted_get():
    """PolicyContext must support dotted paths."""
    ctx = PolicyContext({"a": {"b": {"c": 42}}})
    assert ctx.get("a.b.c") == 42
    assert ctx.get("a.b.missing") is None
    assert ctx.get("a.b.missing", "default") == "default"


def test_context_set_dotted():
    """PolicyContext.set must create intermediate dicts."""
    ctx = PolicyContext()
    ctx.set("x.y.z", 1)
    assert ctx.get("x.y.z") == 1


def test_between_operator():
    """The 'between' operator must match ranges."""
    engine = PolicyEngine.load(
        {
            "version": "1.0",
            "rules": [
                {
                    "name": "mid",
                    "priority": 1,
                    "conditions": [{"field": "value", "op": "between", "value": [0.2, 0.8]}],
                    "actions": [{"target": "flag", "value": True}],
                }
            ],
        }
    )
    assert engine.evaluate(PolicyContext({"value": 0.5})) == {"flag": True}
    assert engine.evaluate(PolicyContext({"value": 0.1})) == {}


def test_in_operator():
    """The 'in' operator must match membership."""
    engine = PolicyEngine.load(
        {
            "version": "1.0",
            "rules": [
                {
                    "name": "mode-in",
                    "priority": 1,
                    "conditions": [
                        {"field": "mode", "op": "in", "value": ["aggressive", "sparse"]}
                    ],
                    "actions": [{"target": "fast", "value": True}],
                }
            ],
        }
    )
    assert engine.evaluate(PolicyContext({"mode": "sparse"})) == {"fast": True}
    assert engine.evaluate(PolicyContext({"mode": "full"})) == {}


def test_validation_catches_unsupported_op():
    """validate must report unsupported operators."""
    engine = PolicyEngine.load(
        {
            "version": "1.0",
            "rules": [
                {
                    "name": "bad",
                    "conditions": [{"field": "x", "op": "matches", "value": 1}],
                    "actions": [],
                }
            ],
        }
    )
    errors = engine.validate()
    assert any("unsupported operator" in e for e in errors)


def test_validation_catches_bad_between():
    """validate must report malformed 'between' values."""
    engine = PolicyEngine.load(
        {
            "version": "1.0",
            "rules": [
                {
                    "name": "bad-between",
                    "conditions": [{"field": "x", "op": "between", "value": [0.0]}],
                    "actions": [],
                }
            ],
        }
    )
    errors = engine.validate()
    assert any("between" in e for e in errors)


def test_load_policy_validates(tmp_path):
    """load_policy must raise on invalid policies."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        (
            "version: '1.0'\n"
            "rules:\n"
            "  - name: bad\n"
            "    conditions:\n"
            "      - {field: x, op: bad_op, value: 1}\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_policy(str(path))


def test_default_policy_loads():
    """The bundled default policy must load and validate without errors."""
    default_path = Path(__file__).resolve().parents[1] / "configs" / "policy_default.yaml"
    engine = load_policy(str(default_path))
    assert engine.get("qdrift.sensitivity_high") == 0.7
    ctx = PolicyContext(
        {
            "alert_accuracy_critical": True,
            "memory_ratio": 0.0,
            "latency_ratio": 0.0,
            "accuracy_drop": 0.0,
        }
    )
    actions = engine.evaluate(ctx)
    assert actions["shadowkv.mode"] == "conservative"
    assert actions["ffn.mode"] == "full"

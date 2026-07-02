"""Declarative Policy DSL for ShadowInfer optimization decisions.

The Policy DSL externalizes the hard-coded thresholds and arbitration rules
scattered across Orchestrator and Agents into YAML/JSON files. Rules are
evaluated in priority order; the first rule whose conditions all match wins.

Version: 3.2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PolicyCondition:
    """A single condition: ``field op value`` evaluated against a context."""

    field: str
    op: str
    value: Any

    def evaluate(self, context: "PolicyContext") -> bool:
        """Return True if the condition holds in ``context``."""
        actual = context.get(self.field)
        return _compare(actual, self.op, self.value)


@dataclass
class PolicyAction:
    """An action to apply when a rule matches: set ``target`` to ``value``."""

    target: str
    value: Any


@dataclass
class PolicyRule:
    """A named rule with conditions and actions."""

    name: str
    priority: int = 0
    description: str = ""
    conditions: List[PolicyCondition] = field(default_factory=list)
    actions: List[PolicyAction] = field(default_factory=list)

    def matches(self, context: "PolicyContext") -> bool:
        """Return True if all conditions match."""
        return all(cond.evaluate(context) for cond in self.conditions)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class PolicyContext:
    """Simple dotted-path context for policy evaluation."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = dict(data or {})

    def get(self, path: str, default: Any = None) -> Any:
        """Return the value at dotted path ``path``."""
        parts = path.split(".")
        node: Any = self._data
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, path: str, value: Any) -> None:
        """Set ``path`` to ``value``, creating intermediate dicts as needed."""
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Load, validate, and evaluate a declarative policy.

    A policy file contains:

    - ``version``: schema version string.
    - ``defaults``: nested threshold/default values retrievable via ``get()``.
    - ``rules``: ordered list of rules with conditions and actions.

    Rules are sorted by descending ``priority``; within the same priority the
    file order is preserved.
    """

    SUPPORTED_OPS: Tuple[str, ...] = (
        ">=",
        "<=",
        ">",
        "<",
        "==",
        "!=",
        "in",
        "not_in",
        "between",
    )

    def __init__(self) -> None:
        self.version: str = "1.0"
        self.defaults: PolicyContext = PolicyContext()
        self.rules: List[PolicyRule] = []
        self._source_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Any) -> "PolicyEngine":
        """Load a policy from a YAML/JSON file or from a dictionary."""
        if isinstance(path, (str, Path)):
            return cls._load_file(Path(path))
        if isinstance(path, dict):
            return cls._load_dict(path)
        raise TypeError(f"Policy must be loaded from str, Path or dict, got {type(path)}")

    @classmethod
    def _load_file(cls, path: Path) -> "PolicyEngine":
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        suffix = path.suffix.lower()
        with path.open("r", encoding="utf-8") as f:
            if suffix in (".yaml", ".yml"):
                if not _YAML_AVAILABLE:
                    raise RuntimeError("PyYAML is required to load YAML policy files")
                data = yaml.safe_load(f)
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported policy file format: {suffix}")
        engine = cls._load_dict(data or {})
        engine._source_path = path
        return engine

    @classmethod
    def _load_dict(cls, data: Dict[str, Any]) -> "PolicyEngine":
        engine = cls()
        engine.version = str(data.get("version", "1.0"))
        engine.defaults = PolicyContext(data.get("defaults", {}))
        for raw_rule in data.get("rules", []):
            engine.rules.append(_parse_rule(raw_rule))
        engine._sort_rules()
        return engine

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, context: Optional[PolicyContext] = None) -> Dict[str, Any]:
        """Evaluate rules against ``context`` and return merged actions.

        Actions are merged by target; higher-priority rules overwrite lower-
        priority rules for the same target.
        """
        ctx = context or PolicyContext()
        actions: Dict[str, Any] = {}
        for rule in self.rules:
            if rule.matches(ctx):
                for action in rule.actions:
                    # Higher-priority rules are evaluated first; they win.
                    if action.target not in actions:
                        actions[action.target] = action.value
        return actions

    def evaluate_with_defaults(self, context: Optional[PolicyContext] = None) -> PolicyContext:
        """Return a context that starts from defaults and overlays rule actions."""
        result = PolicyContext(self.defaults.to_dict())
        for target, value in self.evaluate(context).items():
            result.set(target, value)
        return result

    def get(self, path: str, default: Any = None) -> Any:
        """Return a default/threshold value from the policy."""
        return self.defaults.get(path, default)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the policy and return a list of human-readable errors."""
        errors: List[str] = []
        for rule in self.rules:
            if not rule.conditions and not rule.actions:
                errors.append(f"Rule {rule.name!r} has no conditions and no actions")
            for cond in rule.conditions:
                if cond.op not in self.SUPPORTED_OPS:
                    errors.append(f"Rule {rule.name!r}: unsupported operator {cond.op!r}")
                if cond.op == "between" and (
                    not isinstance(cond.value, (list, tuple)) or len(cond.value) != 2
                ):
                    errors.append(f"Rule {rule.name!r}: 'between' requires a two-element list")
        return errors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sort_rules(self) -> None:
        self.rules.sort(key=lambda r: r.priority, reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare(actual: Any, op: str, expected: Any) -> bool:
    """Compare ``actual`` against ``expected`` using ``op``."""
    try:
        if op == ">=":
            return actual is not None and actual >= expected
        if op == "<=":
            return actual is not None and actual <= expected
        if op == ">":
            return actual is not None and actual > expected
        if op == "<":
            return actual is not None and actual < expected
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op == "in":
            return actual in expected
        if op == "not_in":
            return actual not in expected
        if op == "between":
            lo, hi = expected
            return actual is not None and lo <= actual <= hi
    except TypeError:
        return False
    raise ValueError(f"Unsupported operator: {op}")


def _parse_rule(raw: Dict[str, Any]) -> PolicyRule:
    """Parse a rule dictionary into a ``PolicyRule``."""
    conditions = []
    for raw_cond in raw.get("conditions", []):
        conditions.append(
            PolicyCondition(
                field=str(raw_cond["field"]),
                op=str(raw_cond["op"]),
                value=raw_cond["value"],
            )
        )
    actions = []
    for raw_act in raw.get("actions", []):
        actions.append(
            PolicyAction(
                target=str(raw_act["target"]),
                value=raw_act["value"],
            )
        )
    return PolicyRule(
        name=str(raw.get("name", "unnamed")),
        priority=int(raw.get("priority", 0)),
        description=str(raw.get("description", "")),
        conditions=conditions,
        actions=actions,
    )


def load_policy(path: Any) -> PolicyEngine:
    """Convenience helper: load and validate a policy."""
    engine = PolicyEngine.load(path)
    errors = engine.validate()
    if errors:
        raise ValueError("Invalid policy:\n" + "\n".join(errors))
    return engine

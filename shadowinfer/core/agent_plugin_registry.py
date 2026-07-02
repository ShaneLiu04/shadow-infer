"""Agent plugin registry.

Supports built-in agents (profiler, shadowkv, qdrift, ffn_optimizer) and
third-party agents registered via the ``shadowinfer.agents`` entry-point group
in ``pyproject.toml``.

Version: 3.2.2
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

try:
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover - Python <3.10 fallback
    try:
        from importlib_metadata import entry_points  # type: ignore
    except ImportError:
        entry_points = None  # type: ignore

from shadowinfer.core.base_agent import BaseAgent


@dataclass
class AgentPluginSpec:
    """Metadata for a registered agent plugin."""

    name: str
    cls: Callable[..., BaseAgent]
    help: str = ""
    is_plugin: bool = False

    def is_available(self) -> bool:
        """Return True if the agent class can be instantiated.

        If the agent class exposes an ``is_available()`` classmethod, trust it.
        Otherwise assume the agent is importable (true for built-ins).
        """
        availability_check = getattr(self.cls, "is_available", None)
        if availability_check is not None:
            try:
                return bool(availability_check())
            except Exception:
                return False
        return True


class AgentPluginRegistry:
    """Registry for ``BaseAgent`` implementations.

    Agents can be registered manually or discovered automatically from
    ``pyproject.toml`` entry points under the ``shadowinfer.agents`` group.

    Example entry point:

    .. code-block:: toml

        [project.entry-points."shadowinfer.agents"]
        my_agent = "my_package:MyAgent"
    """

    ENTRY_POINT_GROUP = "shadowinfer.agents"

    def __init__(self) -> None:
        self._agents: Dict[str, AgentPluginSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        cls: Type[BaseAgent],
        help: str = "",  # noqa: A002
        is_plugin: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Register an agent class under ``name``."""
        if not issubclass(cls, BaseAgent):
            raise TypeError(f"Agent {name!r} must inherit from BaseAgent")
        if name in self._agents and not overwrite:
            raise ValueError(f"Agent {name!r} is already registered")
        self._agents[name] = AgentPluginSpec(name, cls, help=help, is_plugin=is_plugin)

    def register_builtin_agents(self) -> None:
        """Register all built-in agents."""
        from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent
        from shadowinfer.profiler.profiler_agent import ProfilerAgent
        from shadowinfer.qdrift.qdrift_agent import QDriftAgent
        from shadowinfer.shadowkv.shadowkv_agent import ShadowKVAgent

        self.register(
            "profiler",
            ProfilerAgent,
            help="Profiler agent: latency, memory and accuracy tracking",
        )
        self.register(
            "shadowkv",
            ShadowKVAgent,
            help="ShadowKV agent: KV cache compression and reuse",
        )
        self.register(
            "qdrift",
            QDriftAgent,
            help="Q-drift agent: diffusion step sensitivity and noise scheduling",
        )
        self.register(
            "ffn_optimizer",
            FFNOptimizerAgent,
            help="FFN optimizer agent: sparse/mixed-precision FFN dispatch",
        )

    def discover_plugins(self) -> None:
        """Discover third-party agents from entry points.

        Failures for individual plugins are logged as warnings so that a broken
        optional dependency does not prevent the rest of the framework from
        loading.
        """
        if entry_points is None:
            return

        try:
            eps = entry_points(group=self.ENTRY_POINT_GROUP)
        except Exception as exc:
            warnings.warn(f"Failed to discover agent plugins: {exc}")
            return

        for ep in eps:
            try:
                cls = ep.load()
                if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
                    warnings.warn(
                        f"Agent plugin {ep.name!r} from {ep.value!r} does not "
                        "inherit from BaseAgent; skipping"
                    )
                    continue
                self.register(
                    ep.name,
                    cls,
                    help=f"Plugin agent from {ep.value}",
                    is_plugin=True,
                )
            except Exception as exc:
                warnings.warn(f"Failed to load agent plugin {ep.name!r}: {exc}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Type[BaseAgent]:
        """Return the agent class registered under ``name``."""
        spec = self._agents.get(name)
        if spec is None:
            available = ", ".join(sorted(self.list_names()))
            raise KeyError(f"Unknown agent {name!r}. Available: {available}")
        return spec.cls

    def get_spec(self, name: str) -> AgentPluginSpec:
        """Return the full ``AgentPluginSpec`` for ``name``."""
        spec = self._agents.get(name)
        if spec is None:
            available = ", ".join(sorted(self.list_names()))
            raise KeyError(f"Unknown agent {name!r}. Available: {available}")
        return spec

    def list_names(self) -> List[str]:
        """Return all registered agent names."""
        return list(self._agents.keys())

    def list_available(self) -> List[str]:
        """Return agent names that can be instantiated in this environment."""
        return [name for name, spec in self._agents.items() if spec.is_available()]

    def is_registered(self, name: str) -> bool:
        """Return True if ``name`` has been registered."""
        return name in self._agents

    def is_available(self, name: str) -> bool:
        """Return True if ``name`` is registered and available."""
        spec = self._agents.get(name)
        return spec is not None and spec.is_available()

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Instantiate the agent ``name`` with ``config``.

        Args:
            name: Registered agent name.
            config: Configuration dictionary passed to the agent constructor.

        Returns:
            An instance of the registered agent class.
        """
        spec = self.get_spec(name)
        if not spec.is_available():
            raise RuntimeError(
                f"Agent {name!r} is not available in this environment "
                f"(missing optional dependencies?)"
            )
        cfg = dict(config) if config is not None else {}
        return spec.cls(config=cfg)

    def help_text(self, name: Optional[str] = None) -> str:
        """Return a human-readable help string for one or all agents."""
        if name is not None:
            spec = self.get_spec(name)
            avail = "available" if spec.is_available() else "unavailable"
            return f"  {spec.name:16} [{avail}] {spec.help}"
        lines = ["Available agents:"]
        for spec in self._agents.values():
            avail = "available" if spec.is_available() else "unavailable"
            lines.append(f"  {spec.name:16} [{avail}] {spec.help}")
        return "\n".join(lines)


# Global registry instance. Built-in agents are registered immediately; plugins
# are discovered lazily on first access to avoid import-time side effects.
_global_registry: Optional[AgentPluginRegistry] = None


def get_agent_plugin_registry(discover_plugins: bool = True) -> AgentPluginRegistry:
    """Return the global agent plugin registry.

    Args:
        discover_plugins: If True, discover entry-point plugins on first call.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentPluginRegistry()
        _global_registry.register_builtin_agents()
        if discover_plugins:
            _global_registry.discover_plugins()
    return _global_registry

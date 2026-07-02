"""Backend plugin registry.

Supports built-in backends (mock, pytorch, huggingface) and third-party backends
registered via the ``shadowinfer.backends`` entry-point group in ``pyproject.toml``.

Version: 3.2.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

try:
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover - Python <3.10 fallback
    try:
        from importlib_metadata import entry_points  # type: ignore
    except ImportError:
        entry_points = None  # type: ignore

from shadowinfer.core.model_backend import ModelBackend


@dataclass
class BackendSpec:
    """Metadata for a registered backend."""

    name: str
    cls: Type[ModelBackend]
    help: str = ""
    is_plugin: bool = False

    def is_available(self) -> bool:
        """Return True if the backend can be instantiated in this environment."""
        # If the backend class exposes an is_available() classmethod, trust it.
        # Otherwise assume the backend is importable (true for built-ins).
        availability_check = getattr(self.cls, "is_available", None)
        if availability_check is not None:
            try:
                return bool(availability_check())
            except Exception:
                return False
        return True


class BackendRegistry:
    """Registry for ``ModelBackend`` implementations.

    Backends can be registered manually or discovered automatically from
    ``pyproject.toml`` entry points under the ``shadowinfer.backends`` group.

    Example entry point:

    .. code-block:: toml

        [project.entry-points."shadowinfer.backends"]
        vllm = "shadowinfer_vllm:VLLMModelBackend"
    """

    ENTRY_POINT_GROUP = "shadowinfer.backends"

    def __init__(self) -> None:
        self._backends: Dict[str, BackendSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        cls: Type[ModelBackend],
        help: str = "",  # noqa: A002
        is_plugin: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Register a backend class under ``name``."""
        if not issubclass(cls, ModelBackend):
            raise TypeError(f"Backend {name!r} must inherit from ModelBackend")
        if name in self._backends and not overwrite:
            raise ValueError(f"Backend {name!r} is already registered")
        self._backends[name] = BackendSpec(name, cls, help=help, is_plugin=is_plugin)

    def register_builtin_backends(self) -> None:
        """Register all built-in backends."""
        from shadowinfer.core.backends.huggingface_backend import HuggingFaceModelBackend
        from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
        from shadowinfer.core.backends.sglang_backend import SGLangModelBackend
        from shadowinfer.core.backends.vllm_backend import VLLMModelBackend
        from shadowinfer.core.model_backend import MockModelBackend

        self.register(
            "mock",
            MockModelBackend,
            help="Mock backend for framework testing (no real model)",
        )
        self.register(
            "pytorch",
            PyTorchModelBackend,
            help="PyTorch backend wrapping SimpleDiffusionLLM",
        )
        self.register(
            "huggingface",
            HuggingFaceModelBackend,
            help="HuggingFace AutoModelForCausalLM backend",
        )
        self.register(
            "vllm",
            VLLMModelBackend,
            help="vLLM backend (requires optional 'vllm' package)",
        )
        self.register(
            "sglang",
            SGLangModelBackend,
            help="SGLang backend (requires optional 'sglang' package)",
        )

    def discover_plugins(self) -> None:
        """Discover third-party backends from entry points.

        Failures for individual plugins are logged as warnings so that a broken
        optional dependency does not prevent the rest of the framework from
        loading.
        """
        if entry_points is None:
            return

        try:
            eps = entry_points(group=self.ENTRY_POINT_GROUP)
        except Exception as exc:
            warnings.warn(f"Failed to discover backend plugins: {exc}")
            return

        for ep in eps:
            try:
                cls = ep.load()
                if not isinstance(cls, type) or not issubclass(cls, ModelBackend):
                    warnings.warn(
                        f"Backend plugin {ep.name!r} from {ep.value!r} does not "
                        "inherit from ModelBackend; skipping"
                    )
                    continue
                self.register(ep.name, cls, help=f"Plugin backend from {ep.value}", is_plugin=True)
            except Exception as exc:
                warnings.warn(f"Failed to load backend plugin {ep.name!r}: {exc}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Type[ModelBackend]:
        """Return the backend class registered under ``name``."""
        spec = self._backends.get(name)
        if spec is None:
            available = ", ".join(sorted(self.list_names()))
            raise KeyError(f"Unknown backend {name!r}. Available: {available}")
        return spec.cls

    def get_spec(self, name: str) -> BackendSpec:
        """Return the full ``BackendSpec`` for ``name``."""
        spec = self._backends.get(name)
        if spec is None:
            available = ", ".join(sorted(self.list_names()))
            raise KeyError(f"Unknown backend {name!r}. Available: {available}")
        return spec

    def list_names(self) -> List[str]:
        """Return all registered backend names."""
        return list(self._backends.keys())

    def list_available(self) -> List[str]:
        """Return backend names that can be instantiated in this environment."""
        return [name for name, spec in self._backends.items() if spec.is_available()]

    def is_registered(self, name: str) -> bool:
        """Return True if ``name`` has been registered."""
        return name in self._backends

    def is_available(self, name: str) -> bool:
        """Return True if ``name`` is registered and available."""
        spec = self._backends.get(name)
        return spec is not None and spec.is_available()

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(self, name: str, **kwargs: Any) -> ModelBackend:
        """Instantiate the backend ``name`` with ``kwargs``."""
        spec = self.get_spec(name)
        if not spec.is_available():
            raise RuntimeError(
                f"Backend {name!r} is not available in this environment "
                f"(missing optional dependencies?)"
            )
        return spec.cls(**kwargs)

    def help_text(self, name: Optional[str] = None) -> str:
        """Return a human-readable help string for one or all backends."""
        if name is not None:
            spec = self.get_spec(name)
            avail = "available" if spec.is_available() else "unavailable"
            return f"  {spec.name:16} [{avail}] {spec.help}"
        lines = ["Available backends:"]
        for spec in self._backends.values():
            avail = "available" if spec.is_available() else "unavailable"
            lines.append(f"  {spec.name:16} [{avail}] {spec.help}")
        return "\n".join(lines)


# Global registry instance. Built-in backends are registered immediately; plugins
# are discovered lazily on first access to avoid import-time side effects.
_global_registry: Optional[BackendRegistry] = None


def get_backend_registry(discover_plugins: bool = True) -> BackendRegistry:
    """Return the global backend registry.

    Args:
        discover_plugins: If True, discover entry-point plugins on first call.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = BackendRegistry()
        _global_registry.register_builtin_backends()
        if discover_plugins:
            _global_registry.discover_plugins()
    return _global_registry

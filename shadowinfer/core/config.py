"""ShadowInfer Configuration Management.

Implements YAML configuration loader and nested access manager.
Corresponds to TECHNICAL_SPEC.md §4 配置规范。

Version: 3.0
Corresponds to: TECHNICAL_SPEC.md v2.0
"""

from __future__ import annotations

__version__ = "3.0"
__doc_version__ = "TECHNICAL_SPEC.md v2.0"

import os
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> dict:
    """加载 YAML 配置文件并返回原始字典。

    对应 TECHNICAL_SPEC.md §4 配置规范 — 配置加载基础函数。

    Args:
        path: YAML 配置文件的绝对路径或相对路径。

    Returns:
        解析后的配置字典。若文件不存在或解析失败，抛出 FileNotFoundError / yaml.YAMLError。

    Example:
        >>> cfg = load_config("configs/optimize_full.yaml")
        >>> cfg["shadowkv"]["enabled"]
        True
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    """配置管理类，支持嵌套属性访问与字典式访问。

    对应 TECHNICAL_SPEC.md §4 配置规范 — 统一配置访问接口。

    特性：
        - 支持点号属性访问：``config.shadowkv.enabled``
        - 支持字典式访问：``config["shadowkv"]["enabled"]`` 或 ``config.get("enabled", False)``
        - 自动将嵌套 dict 转换为 ``Config`` 子实例，从而支持无限级嵌套访问。
        - 预先加载 ``configs/`` 下的默认配置模板（如 ``model_fast_dllm_7b.yaml``、
          ``optimize_full.yaml``、``profiler_full.yaml``）。

    Attributes:
        _data: 内部存储的原始字典。
    """

    _defaults_loaded: bool = False
    _default_configs: Dict[str, "Config"] = {}

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """初始化配置实例。

        Args:
            data: 原始配置字典。若为 None，则创建空配置。
        """
        if data is None:
            data = {}
        self._data: Dict[str, Any] = data

        # 将嵌套 dict 自动转换为 Config 实例，以便支持点号访问
        for key, value in self._data.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)

    # ------------------------------------------------------------------
    # 字典式访问
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """支持 ``config[key]`` 字典式访问。

        Args:
            key: 配置键名。

        Returns:
            对应值（可能是 Config 子实例、列表或标量）。

        Raises:
            KeyError: 键不存在时抛出。
        """
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """支持 ``config[key] = value`` 字典式赋值。

        若 value 为 dict，自动包装为 Config 子实例。
        """
        if isinstance(value, dict):
            value = Config(value)
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """支持 ``"key" in config`` 成员检测。"""
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """支持带默认值的字典式访问，包括点号分隔的嵌套访问。

        对应 TECHNICAL_SPEC.md §4 配置规范 — 字典式访问接口。

        支持点号分隔键：``config.get("shadowkv.enabled", False)`` 会遍历
        嵌套 Config 并返回 ``enabled`` 的值。若任意中间键不存在，返回 default。

        Args:
            key: 配置键名。可包含点号分隔符以访问嵌套字段。
            default: 键不存在时返回的默认值。

        Returns:
            对应值或 default。
        """
        if "." not in key:
            return self._data.get(key, default)

        keys = key.split(".")
        current = self._data
        for k in keys:
            if isinstance(current, Config):
                current = current._data
            if isinstance(current, dict):
                if k not in current:
                    return default
                current = current[k]
            else:
                return default
        return current

    def keys(self):
        """返回所有顶层键名。对应 TECHNICAL_SPEC.md §4 配置规范。"""
        return self._data.keys()

    def values(self):
        """返回所有顶层值。对应 TECHNICAL_SPEC.md §4 配置规范。"""
        return self._data.values()

    def items(self):
        """返回所有顶层键值对。对应 TECHNICAL_SPEC.md §4 配置规范。"""
        return self._data.items()

    # ------------------------------------------------------------------
    # 属性式访问（点号）
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """支持 ``config.shadowkv.enabled`` 嵌套属性访问。

        注意：仅对非下划线开头的属性名生效，避免与 Python 内部机制冲突。

        Args:
            name: 属性名（即配置键名）。

        Returns:
            对应值。若不存在，抛出 AttributeError。
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Config key '{name}' not found") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """支持属性式赋值；自动将 dict 包装为 Config 子实例。"""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if isinstance(value, dict):
                value = Config(value)
            self._data[name] = value

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """将 Config 递归展开为原始字典。

        对应 TECHNICAL_SPEC.md §4 配置规范 — 配置序列化接口。

        Returns:
            完全展开后的普通 Python 字典，可用于序列化或外部库消费。
        """
        result: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item for item in value
                ]
            else:
                result[key] = value
        return result

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载并返回 Config 实例。

        对应 TECHNICAL_SPEC.md §4 配置规范 — YAML 配置加载入口。

        Args:
            path: YAML 文件路径。

        Returns:
            包装后的 Config 实例。
        """
        raw = load_config(path)
        return cls(raw)

    @classmethod
    def load_defaults(cls, configs_dir: Optional[str] = None) -> Dict[str, "Config"]:
        """预先加载 ``configs/`` 目录下的默认配置模板。

        对应 TECHNICAL_SPEC.md §4.1 ~ §4.3 中定义的默认配置。

        默认加载以下文件：
            - ``configs/model_fast_dllm_7b.yaml``
            - ``configs/optimize_full.yaml``
            - ``configs/profiler_full.yaml``

        Args:
            configs_dir: 配置目录路径。若未提供，则尝试从当前模块路径推断
                ``configs/`` 文件夹。

        Returns:
            字典，key 为文件名（不含扩展名），value 为 Config 实例。
        """
        if configs_dir is None:
            # 尝试从当前模块路径向上查找 configs/
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # 从 shadowinfer/core/ 回退两级到项目根
            project_root = os.path.dirname(os.path.dirname(module_dir))
            configs_dir = os.path.join(project_root, "configs")

        default_files = [
            "model_fast_dllm_7b.yaml",
            "optimize_full.yaml",
            "profiler_full.yaml",
        ]

        loaded: Dict[str, "Config"] = {}
        for filename in default_files:
            filepath = os.path.join(configs_dir, filename)
            if os.path.exists(filepath):
                key = os.path.splitext(filename)[0]
                loaded[key] = cls.from_yaml(filepath)

        cls._defaults_loaded = True
        cls._default_configs = loaded
        return loaded

    @classmethod
    def get_default(cls, name: str) -> "Config":
        """获取已加载的默认配置。

        对应 TECHNICAL_SPEC.md §4 配置规范 — 默认配置访问接口。

        Args:
            name: 默认配置标识（如 ``model_fast_dllm_7b``、``optimize_full``、
                ``profiler_full``）。

        Returns:
            对应 Config 实例。

        Raises:
            KeyError: 若默认配置尚未加载或名称不存在。
        """
        if not cls._defaults_loaded:
            cls.load_defaults()
        return cls._default_configs[name]

    def __repr__(self) -> str:
        """返回 Config 实例的 repr 字符串。对应 TECHNICAL_SPEC.md §4 配置规范。"""
        return f"{type(self).__name__}({self._data!r})"

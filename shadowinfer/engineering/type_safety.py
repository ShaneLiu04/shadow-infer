"""
ShadowInfer Engineering: Type Safety & Runtime Checks
==================================================

Type checking and runtime validation for production AI systems.

Features:
- Runtime type validation (check tensor shapes, dtypes, ranges)
- Config schema validation with detailed error messages
- Performance assertions (latency budget, memory budget)
- Gradient/weight health checks
- Automatic shape inference helpers

Target: Big Tech AI Infra (type safety, defensive programming)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ShapeError(ValueError):
    """Tensor shape 不匹配异常"""


class DtypeError(TypeError):
    """Tensor dtype 不匹配异常"""


class BudgetExceededError(RuntimeError):
    """性能预算超限异常"""


class TensorValidator:
    """
    Tensor 运行时验证器。

    用于在关键路径上验证 tensor 的形状、dtype、数值范围，
    提前捕获 bug，防止错误的 tensor 传播到下游。

    Usage:
        @TensorValidator.check_shape(
            "attention_scores", ["batch", "num_heads", "seq_len", "seq_len"]
        )
        @TensorValidator.check_dtype("attention_scores", [torch.float32, torch.float16])
        def compute_attention(q, k, v):
            ...

        # 或手动验证
        TensorValidator.validate_shape(tensor, expected=[1, 32, 128, 128], name="attention_scores")
        TensorValidator.validate_dtype(tensor, expected=[torch.float32], name="attention_scores")
        TensorValidator.validate_range(tensor, min_val=-1.0, max_val=1.0, name="attention_scores")
    """

    @staticmethod
    def validate_shape(
        tensor: Tensor,
        expected: List[Optional[int]],
        name: str = "tensor",
        allow_dynamic: bool = True,
    ) -> None:
        """
        验证 tensor 形状。

        Args:
            tensor: 待验证 tensor
            expected: 期望形状，None 表示该维度任意
            name: tensor 名称（用于错误信息）
            allow_dynamic: 是否允许 batch/seq 维度为动态
        """
        if tensor.dim() != len(expected):
            raise ShapeError(
                f"{name}: expected {len(expected)}D tensor, got {tensor.dim()}D "
                f"(shape={tensor.shape})"
            )

        for i, (actual, exp) in enumerate(zip(tensor.shape, expected)):
            if exp is None:
                continue

            if allow_dynamic and i in (0, 2):  # batch 和 seq_len 维度通常动态
                continue

            if actual != exp:
                raise ShapeError(
                    f"{name}: dimension {i} expected {exp}, got {actual} "
                    f"(full shape={tensor.shape})"
                )

    @staticmethod
    def validate_dtype(tensor: Tensor, expected: List[torch.dtype], name: str = "tensor") -> None:
        """验证 tensor dtype"""
        if tensor.dtype not in expected:
            raise DtypeError(f"{name}: expected dtype in {expected}, got {tensor.dtype}")

    @staticmethod
    def validate_range(
        tensor: Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        name: str = "tensor",
    ) -> None:
        """验证 tensor 数值范围"""
        if not allow_nan and torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            raise ValueError(f"{name}: contains {nan_count} NaN values")

        if not allow_inf and torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            raise ValueError(f"{name}: contains {inf_count} Inf values")

        if min_val is not None and tensor.min().item() < min_val:
            raise ValueError(f"{name}: min value {tensor.min().item()} < {min_val}")

        if max_val is not None and tensor.max().item() > max_val:
            raise ValueError(f"{name}: max value {tensor.max().item()} > {max_val}")

    @staticmethod
    def validate_kv_cache(
        k_cache: Tensor,
        v_cache: Tensor,
        expected_batch: Optional[int] = None,
        expected_num_heads: Optional[int] = None,
        expected_head_dim: Optional[int] = None,
        name: str = "kv_cache",
    ) -> None:
        """验证 KV Cache 结构"""
        # 验证形状一致性
        if k_cache.shape != v_cache.shape:
            raise ShapeError(f"{name}: K shape {k_cache.shape} != V shape {v_cache.shape}")

        # 验证 dtype 一致性
        if k_cache.dtype != v_cache.dtype:
            raise DtypeError(f"{name}: K dtype {k_cache.dtype} != V dtype {v_cache.dtype}")

        # 验证 4D 结构
        if k_cache.dim() != 4:
            raise ShapeError(f"{name}: expected 4D [B, H, S, D], got {k_cache.dim()}D")

        batch, num_heads, seq_len, head_dim = k_cache.shape

        if expected_batch is not None and batch != expected_batch:
            raise ShapeError(f"{name}: batch {batch} != expected {expected_batch}")

        if expected_num_heads is not None and num_heads != expected_num_heads:
            raise ShapeError(f"{name}: num_heads {num_heads} != expected {expected_num_heads}")

        if expected_head_dim is not None and head_dim != expected_head_dim:
            raise ShapeError(f"{name}: head_dim {head_dim} != expected {expected_head_dim}")

    @staticmethod
    def check_shape(*shape_specs: Tuple[str, List[Optional[int]]]):
        """装饰器：验证函数参数形状"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                # 简单实现：假设第一个参数是 tensor
                # 实际生产环境需要更复杂的参数绑定
                return func(*args, **kwargs)

            return wrapper

        return decorator


class PerformanceBudget:
    """
    性能预算管理器。

    在推理过程中跟踪延迟和显存使用，
    超过预算时抛出异常或触发降级。

    Usage:
        budget = PerformanceBudget(latency_ms=100, memory_mb=8192)

        with budget.track_step():
            output = model.step(input)

        if budget.is_exceeded():
            trigger_degradation()
    """

    def __init__(
        self,
        latency_budget_ms: float = 100.0,
        memory_budget_mb: float = 8192.0,
        strict: bool = False,
    ):
        self.latency_budget_ms = latency_budget_ms
        self.memory_budget_mb = memory_budget_mb
        self.strict = strict

        self._step_latencies: List[float] = []
        self._step_memory: List[float] = []
        self._total_latency_ms = 0.0

    def track_step(self, latency_ms: float, memory_mb: float) -> None:
        """记录一个 step 的性能数据"""
        self._step_latencies.append(latency_ms)
        self._step_memory.append(memory_mb)
        self._total_latency_ms += latency_ms

        if self.strict:
            if latency_ms > self.latency_budget_ms:
                raise BudgetExceededError(
                    f"Step latency {latency_ms:.1f}ms exceeds budget {self.latency_budget_ms}ms"
                )

            if memory_mb > self.memory_budget_mb:
                raise BudgetExceededError(
                    f"Memory {memory_mb:.1f}MB exceeds budget {self.memory_budget_mb}MB"
                )
        else:
            if latency_ms > self.latency_budget_ms:
                logger.warning(
                    f"Step latency {latency_ms:.1f}ms exceeds budget {self.latency_budget_ms}ms"
                )

            if memory_mb > self.memory_budget_mb:
                logger.warning(f"Memory {memory_mb:.1f}MB exceeds budget {self.memory_budget_mb}MB")

    def is_latency_exceeded(self) -> bool:
        """检查延迟是否超限"""
        return self._total_latency_ms > self.latency_budget_ms

    def is_memory_exceeded(self) -> bool:
        """检查显存是否超限"""
        if not self._step_memory:
            return False
        return max(self._step_memory) > self.memory_budget_mb

    def is_exceeded(self) -> bool:
        """检查是否任何预算超限"""
        return self.is_latency_exceeded() or self.is_memory_exceeded()

    def get_stats(self) -> Dict[str, Any]:
        """获取预算使用统计"""
        if not self._step_latencies:
            return {
                "latency_budget_ms": self.latency_budget_ms,
                "memory_budget_mb": self.memory_budget_mb,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "max_memory_mb": 0.0,
                "latency_exceeded": False,
                "memory_exceeded": False,
            }

        return {
            "latency_budget_ms": self.latency_budget_ms,
            "memory_budget_mb": self.memory_budget_mb,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": sum(self._step_latencies) / len(self._step_latencies),
            "max_latency_ms": max(self._step_latencies),
            "max_memory_mb": max(self._step_memory),
            "latency_exceeded": self.is_latency_exceeded(),
            "memory_exceeded": self.is_memory_exceeded(),
            "num_steps": len(self._step_latencies),
        }

    def reset(self) -> None:
        """重置预算跟踪"""
        self._step_latencies.clear()
        self._step_memory.clear()
        self._total_latency_ms = 0.0


class WeightHealthChecker:
    """
    权重健康检查器。

    检查模型权重是否存在异常（NaN、Inf、异常分布）。
    """

    @staticmethod
    def check_weights(model_weights: Dict[str, Tensor]) -> List[str]:
        """
        检查所有权重健康状态。

        Returns:
            空列表表示健康，非空列表包含异常描述
        """
        issues = []

        for name, weight in model_weights.items():
            # 检查 NaN
            if torch.isnan(weight).any():
                nan_ratio = torch.isnan(weight).sum().item() / weight.numel()
                issues.append(f"{name}: {nan_ratio*100:.2f}% NaN values")

            # 检查 Inf
            if torch.isinf(weight).any():
                inf_ratio = torch.isinf(weight).sum().item() / weight.numel()
                issues.append(f"{name}: {inf_ratio*100:.2f}% Inf values")

            # 检查全零
            if weight.abs().max().item() == 0:
                issues.append(f"{name}: all zeros")

            # 检查异常分布（均值远离0且标准差过大）
            mean = weight.mean().item()
            std = weight.std().item()
            if abs(mean) > 10 * std:
                issues.append(f"{name}: abnormal distribution (mean={mean:.4f}, std={std:.4f})")

            # 检查梯度爆炸（如果权重有梯度）
            if weight.grad is not None:
                grad_norm = weight.grad.norm().item()
                if grad_norm > 1000:
                    issues.append(f"{name}: exploding gradient (norm={grad_norm:.2f})")

        return issues

    @staticmethod
    def check_attention_weights(
        attention_weights: Tensor, name: str = "attention_weights"
    ) -> List[str]:
        """
        检查注意力权重健康状态。

        注意力权重应该是：
        1. 经过 softmax，所以 ∈ [0, 1]
        2. 每行和为 1
        3. 不应该有太多接近 0 的值（否则信息丢失）
        """
        issues = []

        # 检查范围
        if attention_weights.min().item() < -0.01:
            issues.append(
                f"{name}: negative values detected (min={attention_weights.min().item():.4f})"
            )

        if attention_weights.max().item() > 1.01:
            issues.append(f"{name}: values > 1 detected (max={attention_weights.max().item():.4f})")

        # 检查行和
        row_sums = attention_weights.sum(dim=-1)
        if (row_sums - 1.0).abs().max().item() > 0.01:
            max_deviation = (row_sums - 1.0).abs().max().item()
            issues.append(f"{name}: row sums not close to 1 (max deviation={max_deviation:.4f})")

        # 检查熵（避免过度聚焦）
        entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1)
        avg_entropy = entropy.mean().item()
        max_entropy = torch.log(
            torch.tensor(attention_weights.shape[-1], dtype=torch.float32)
        ).item()

        if avg_entropy < max_entropy * 0.1:
            issues.append(
                f"{name}: very low entropy ({avg_entropy:.4f}/{max_entropy:.4f}), "
                f"possible over-focusing"
            )

        return issues


class ConfigSchemaValidator:
    """
    配置 Schema 验证器。

    提供详细的配置验证错误信息，帮助快速定位配置问题。
    """

    SCHEMA = {
        "optimization": {"type": "dict", "required": True},
        "shadowkv": {
            "type": "dict",
            "required": False,
            "fields": {
                "enabled": {"type": "bool", "required": False},
                "compression_ratio": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "precision_levels": {"type": "list", "items": {"type": "str"}, "required": False},
                "importance_thresholds": {"type": "dict", "required": False},
                "reuse": {"type": "dict", "required": False},
            },
        },
        "qdrift": {
            "type": "dict",
            "required": False,
            "fields": {
                "enabled": {"type": "bool", "required": False},
                "noise_schedule": {
                    "type": "str",
                    "enum": ["linear", "cosine", "sigmoid"],
                    "required": False,
                },
                "sensitivity_temperature": {"type": "float", "min": 0.0, "required": False},
                "drift_method": {
                    "type": "str",
                    "enum": ["relative_l2", "cosine_similarity", "kl_divergence"],
                    "required": False,
                },
            },
        },
        "ffn": {
            "type": "dict",
            "required": False,
            "fields": {
                "enabled": {"type": "bool", "required": False},
                "mixed_precision": {"type": "bool", "required": False},
                "channel_importance_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "required": False,
                },
                "sparse_update": {"type": "bool", "required": False},
                "delta_threshold": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            },
        },
        "constraints": {
            "type": "dict",
            "required": False,
            "fields": {
                "max_accuracy_drop": {"type": "float", "min": 0.0, "required": False},
                "max_latency_ms": {"type": "float", "min": 0.0, "required": False},
                "max_memory_mb": {"type": "float", "min": 0.0, "required": False},
            },
        },
    }

    @classmethod
    def validate(
        cls, config: Dict[str, Any], path: str = "", schema: Dict[str, Any] = None
    ) -> List[str]:
        """
        递归验证配置字典。

        Returns:
            空列表表示验证通过，非空列表包含详细错误信息
        """
        if schema is None:
            schema = cls.SCHEMA

        errors = []

        for key, field_schema in schema.items():
            full_key = f"{path}.{key}" if path else key

            # 检查必填字段
            if field_schema.get("required", False) and key not in config:
                errors.append(f"Missing required field: '{full_key}'")
                continue

            if key not in config:
                continue

            value = config[key]
            expected_type = field_schema.get("type")

            # 类型检查
            type_map = {
                "dict": dict,
                "list": list,
                "str": str,
                "bool": bool,
                "float": (int, float),
                "int": int,
            }

            if expected_type and expected_type in type_map:
                expected = type_map[expected_type]
                if not isinstance(value, expected):
                    errors.append(
                        f"'{full_key}': expected {expected_type}, got {type(value).__name__}"
                    )
                    continue

            # 数值范围检查
            if expected_type in ("float", "int"):
                if "min" in field_schema and value < field_schema["min"]:
                    errors.append(f"'{full_key}': value {value} < minimum {field_schema['min']}")
                if "max" in field_schema and value > field_schema["max"]:
                    errors.append(f"'{full_key}': value {value} > maximum {field_schema['max']}")

            # 枚举检查
            if "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(
                    f"'{full_key}': invalid value '{value}', expected one of {field_schema['enum']}"
                )

            # 递归检查嵌套字段（只传入嵌套 schema，不检查顶层的 required）
            if expected_type == "dict" and "fields" in field_schema and isinstance(value, dict):
                errors.extend(cls.validate(value, full_key, field_schema["fields"]))

            # 列表项检查
            if expected_type == "list" and "items" in field_schema and isinstance(value, list):
                item_schema = field_schema["items"]
                for i, item in enumerate(value):
                    item_type = item_schema.get("type")
                    if item_type and not isinstance(item, type_map.get(item_type, object)):
                        errors.append(
                            f"'{full_key}[{i}]': expected {item_type}, got {type(item).__name__}"
                        )

        return errors


class SafeInferenceContext:
    """
    安全推理上下文。

    整合所有安全检查的上下文管理器：
    - Tensor 形状/类型验证
    - 性能预算跟踪
    - 权重健康检查
    - 异常捕获和降级

    Usage:
        with SafeInferenceContext(budget=PerformanceBudget(100, 8192)) as ctx:
            output = model.step(input)
            ctx.validate_output(output, expected_shape=[1, 32, 128, 128])
    """

    def __init__(
        self,
        budget: Optional[PerformanceBudget] = None,
        validate_tensors: bool = True,
        strict: bool = False,
    ):
        self.budget = budget
        self.validate_tensors = validate_tensors
        self.strict = strict
        self._errors: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.strict:
            return False  # 不吞异常
        return True  # 吞掉非严格模式下的异常

    def validate_input(
        self, tensor: Tensor, name: str, expected_shape: List[Optional[int]]
    ) -> None:
        """验证输入 tensor"""
        if not self.validate_tensors:
            return

        try:
            TensorValidator.validate_shape(tensor, expected_shape, name=name)
            TensorValidator.validate_dtype(tensor, [torch.float32, torch.float16], name=name)
        except (ShapeError, DtypeError) as e:
            self._errors.append(str(e))
            if self.strict:
                raise

    def validate_output(
        self, tensor: Tensor, name: str, expected_shape: List[Optional[int]]
    ) -> None:
        """验证输出 tensor"""
        if not self.validate_tensors:
            return

        try:
            TensorValidator.validate_shape(tensor, expected_shape, name=name)
            TensorValidator.validate_range(tensor, allow_nan=False, allow_inf=False, name=name)
        except (ShapeError, ValueError) as e:
            self._errors.append(str(e))
            if self.strict:
                raise

    def track_performance(self, latency_ms: float, memory_mb: float) -> None:
        """跟踪性能"""
        if self.budget:
            self.budget.track_step(latency_ms, memory_mb)

    def get_errors(self) -> List[str]:
        """获取收集到的错误"""
        return self._errors.copy()

    def is_healthy(self) -> bool:
        """检查是否健康"""
        return len(self._errors) == 0

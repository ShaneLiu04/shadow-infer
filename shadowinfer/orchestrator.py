"""Orchestrator — 统一调度协调器。

对应文档：AGENTS.md §2.1, ARCHITECTURE.md §4.1 / §4.2
版本：v3.2.2
"""

from __future__ import annotations

__version__ = "3.2.2"

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from shadowinfer.core.agent_plugin_registry import (
    AgentPluginRegistry,
    get_agent_plugin_registry,
)
from shadowinfer.core.base_agent import AgentRegistry, BaseAgent
from shadowinfer.core.bus import MESSAGE_TYPES, ProfilingBus
from shadowinfer.core.config import Config
from shadowinfer.core.model_backend import MockModelBackend, ModelBackend
from shadowinfer.core.policy import PolicyContext, PolicyEngine
from shadowinfer.core.scheduler import LearnedScheduler, StepExperience
from shadowinfer.core.structs import (
    AgentState,
    KVCacheEntry,
    Message,
    PipelineContext,
    StepConfig,
    StepState,
)
from shadowinfer.engineering.degradation_circuit import ProductionSafetyNet
from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent
from shadowinfer.profiler.profiler_agent import ProfilerAgent
from shadowinfer.qdrift.early_stopper import EarlyStopConfig, UncertaintyEarlyStopper
from shadowinfer.qdrift.qdrift_agent import QDriftAgent
from shadowinfer.shadowkv.shadowkv_agent import ShadowKVAgent
from shadowinfer.utils.logging_utils import (
    StructuredLogger,
    configure_shadowinfer_logging,
)


class InferenceResult:
    """推理结果包装类（测试兼容）。"""

    def __init__(self) -> None:
        self.step_results: List[Dict[str, Any]] = []
        self.accuracy_drop: float = 0.0
        self.latency_ms: float = 0.0
        self.memory_mb: float = 0.0
        self.constraints_satisfied: bool = True
        self.outputs: List[Any] = []


class Orchestrator:
    """Orchestrator — 统一调度协调器。

    对应文档：AGENTS.md §2.1, ARCHITECTURE.md §4.1 / §4.2, ROADMAP.md
    版本：v3.2.2

    职责：
    1. 初始化所有 Agent 并建立通信通道
    2. 按 denoising step 调度各 Agent 的执行顺序
    3. 收集各 Agent 的优化决策，合成 per-step 配置
    4. 监控全局状态（显存、延迟、精度约束）
    5. Agent 决策冲突时仲裁
    6. 支持流式输出、事件溯源快照与运行期取消
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        config_path: str = None,
        config: str = None,
        model_backend: Optional[ModelBackend] = None,
        scheduler: Optional[LearnedScheduler] = None,
    ) -> None:
        """初始化 Orchestrator。

        Args:
            config_path: 配置文件路径、Config 对象或字典。若为 None，使用默认配置。
            config: 兼容别名，等价于 config_path。
            model_backend: 可选的模型后端实例。
            scheduler: 可选的学习调度器实例。
        """
        path = config_path or config
        if isinstance(path, Config):
            self.config = path
        elif isinstance(path, dict):
            self.config = Config(path)
        else:
            self.config = self._load_config(path)
        self.bus = ProfilingBus(name="shadowinfer_main")
        self.registry = AgentRegistry()
        self.agent_registry: AgentPluginRegistry = get_agent_plugin_registry()
        self.profiler: Optional[ProfilerAgent] = None
        self.shadowkv: Optional[ShadowKVAgent] = None
        self.qdrift: Optional[QDriftAgent] = None
        self.ffn_optimizer: Optional[FFNOptimizerAgent] = None
        self.logger = StructuredLogger("orchestrator", self.config.get("log_dir", "logs/"))
        self.step_results: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {
            "total_steps": 0,
            "current_step": 0,
            "accuracy_drop_cumulative": 0.0,
            "memory_used_mb": 0.0,
            "latency_budget_ms": self.config.get("latency_budget_ms", 100.0),
            "memory_budget_mb": self.config.get("memory_budget_mb", 8192.0),
        }
        self._run_id: str = str(uuid.uuid4())[:8]
        self._model_config: Dict[str, Any] = {}
        self._initialized = False
        self.model_backend: Optional[ModelBackend] = model_backend
        self.scheduler: Optional[LearnedScheduler] = scheduler
        self._scheduler_model_path: Optional[str] = None
        self._prev_scheduler_state: Dict[str, float] = {
            "latency_ms": 0.0,
            "memory_mb": 0.0,
            "accuracy_drop": 0.0,
        }
        self.policy: PolicyEngine = self._load_policy()
        self.safety_net: Optional[ProductionSafetyNet] = None
        self.early_stopper: Optional[UncertaintyEarlyStopper] = None
        self._cancelled = False
        self._pipeline_context: Optional[PipelineContext] = None

        self._configure_logging()

    def _load_config(self, config_path: Optional[str]) -> Config:
        """加载配置。如果 config_path 为 None，使用默认配置。"""
        if config_path is not None and isinstance(config_path, str) and os.path.exists(config_path):
            return Config.from_yaml(config_path)
        return Config({})

    def _load_policy(self) -> PolicyEngine:
        """Load declarative policy from config or the bundled default."""
        explicit_path = self.config.get("policy_path")
        if explicit_path and os.path.exists(str(explicit_path)):
            return PolicyEngine.load(str(explicit_path))

        default_path = Path(__file__).resolve().parents[1] / "configs" / "policy_default.yaml"
        if default_path.exists():
            return PolicyEngine.load(str(default_path))

        return PolicyEngine()

    def _configure_logging(self) -> None:
        """Apply global log level and rotation settings from config."""
        log_cfg = self.config.get("logging", {})
        level = log_cfg.get("level")
        rotation = log_cfg.get("rotation")
        if level is not None or rotation is not None:
            configure_shadowinfer_logging(level=level, rotation=rotation)
            self.logger.log_event(
                "logging_config",
                "Applied global logging configuration.",
                data={"level": level, "rotation": rotation},
            )

    def set_log_level(self, level: Union[int, str]) -> None:
        """Dynamically update the log level for all ShadowInfer loggers."""
        configure_shadowinfer_logging(level=level)
        self.logger.log_event(
            "logging_level_changed",
            f"Log level changed to {level}.",
            data={"level": str(level)},
        )

    def initialize(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        """初始化 Orchestrator（对外统一入口）。

        Args:
            model_config: 模型级配置字典；若为 None，则使用适合测试/快速运行的默认配置。
        """
        if model_config is None:
            model_config = {
                "name": "Fast-dLLM-v2-7B",
                "num_layers": 4,
                "num_heads": 32,
                "head_dim": 128,
                "hidden_dim": 4096,
                "intermediate_dim": 11008,
                "batch_size": 1,
                "seq_len": 128,
                "max_latency_ms": 100.0,
                "max_memory_mb": 8192.0,
            }
        self.initialize_agents(model_config)
        self._initialized = True
        for agent in self.registry:
            agent.set_state(AgentState.READY)

    def initialize_agents(self, model_config: Dict[str, Any]) -> None:
        """初始化所有 Agent（内置 + 插件）。

        顺序：
        1. 通过 AgentPluginRegistry 创建内置 Agent
        2. 从配置读取并创建 extra plugin agents
        3. 注册到 AgentRegistry
        4. 订阅到 ProfilingBus
        5. 调用每个 Agent 的 on_init()

        Args:
            model_config: 模型级配置字典（如层数、hidden_dim 等）。
        """
        self._model_config = model_config
        model_name = model_config.get("name", "unknown")
        run_id = self._run_id

        builtin_cfgs: Dict[str, Dict[str, Any]] = {
            "profiler": {"model_name": model_name, "run_id": run_id},
            "shadowkv": {
                "model_name": model_name,
                "run_id": run_id,
                "num_layers": model_config.get("num_layers", 32),
                "num_heads": model_config.get("num_heads", 32),
                "head_dim": model_config.get("head_dim", 128),
            },
            "qdrift": {"model_name": model_name, "run_id": run_id},
            "ffn_optimizer": {"model_name": model_name, "run_id": run_id},
        }
        for key, defaults in builtin_cfgs.items():
            user_cfg = dict(self.config.get(key, {}))
            user_cfg.update(defaults)
            builtin_cfgs[key] = user_cfg

        self.profiler = self.agent_registry.create("profiler", builtin_cfgs["profiler"])
        self.shadowkv = self.agent_registry.create("shadowkv", builtin_cfgs["shadowkv"])
        self.qdrift = self.agent_registry.create("qdrift", builtin_cfgs["qdrift"])
        self.ffn_optimizer = self.agent_registry.create(
            "ffn_optimizer", builtin_cfgs["ffn_optimizer"]
        )

        early_stop_cfg = EarlyStopConfig(**dict(self.config.get("early_stop", {"enabled": False})))
        if early_stop_cfg.enabled:
            self.early_stopper = early_stop_cfg.build()
        else:
            self.early_stopper = None

        for agent in (self.profiler, self.shadowkv, self.qdrift, self.ffn_optimizer):
            self.registry.register(agent)

        # Load optional third-party agents declared in config["extra_agents"].
        extra_agents = self.config.get("extra_agents", []) or []
        for entry in extra_agents:
            if isinstance(entry, Config):
                entry = entry.to_dict()
            if not isinstance(entry, dict):
                self.logger.log_event(
                    "agent_plugin_skip",
                    "Skipping malformed extra agent entry.",
                    data={"entry": str(entry)},
                )
                continue
            agent_name = entry.get("name")
            plugin_name = entry.get("agent")
            if not agent_name or not plugin_name:
                self.logger.log_event(
                    "agent_plugin_skip",
                    "Skipping extra agent entry without 'name' or 'agent'.",
                    data={"entry": entry},
                )
                continue
            if not self.agent_registry.is_registered(plugin_name):
                raise KeyError(
                    f"Unknown agent plugin {plugin_name!r}. "
                    f"Available: {', '.join(sorted(self.agent_registry.list_names()))}"
                )
            cfg = dict(entry.get("config", {}))
            cfg.setdefault("model_name", model_name)
            cfg.setdefault("run_id", run_id)
            agent = self.agent_registry.create(plugin_name, cfg)
            agent.name = agent_name
            self.registry.register(agent)

        for agent in self.registry:
            self.bus.subscribe(agent.name, self._make_bus_callback(agent))

        for agent in self.registry:
            agent.on_init(model_config)

        self.logger.log_event(
            "orchestrator_init",
            f"All agents initialized for model={model_name}, run_id={run_id}",
            data={"agents": list(self.registry.get_all().keys())},
        )

    def _make_bus_callback(self, agent: BaseAgent) -> Callable[[Message], None]:
        """为 Agent 创建 ProfilingBus 消息回调。"""

        def callback(msg: Message) -> None:
            self.logger.log_event(
                "bus_message",
                f"Message to {agent.name}: {msg.message_type}",
                data={"message_id": msg.message_id, "step_id": msg.step_id},
            )

        return callback

    def _run_plugin_agents(
        self, step_config: StepConfig, state: StepState, inputs: Dict[str, Any]
    ) -> None:
        """执行所有非内置的插件 Agent。

        插件 Agent 可以观察当前 step 的内置 Agent 输出并产生附加结果，
        结果会写入 ``state.outputs`` 并随 STEP_RESULT 消息广播。
        """
        builtin_names = {
            self.profiler.name,
            self.shadowkv.name,
            self.qdrift.name,
            self.ffn_optimizer.name,
        }
        for agent in self.registry:
            if agent.name in builtin_names:
                continue
            plugin_inputs = {
                "step_id": state.step_id,
                "total_steps": state.total_steps,
                "step_config": step_config,
                "qdrift": state.qdrift_result,
                "shadowkv": state.kv_result,
                "ffn": state.ffn_result,
                "profiler": state.profiler_result,
                "inputs": inputs,
            }
            try:
                result = agent.on_step(step_config, plugin_inputs)
                state.outputs[agent.name] = result
            except Exception as exc:
                self.logger.log_alert(
                    "warning",
                    f"Plugin agent {agent.name} failed at step {state.step_id}: {exc}",
                    recommendation="Check plugin implementation.",
                    step_id=state.step_id,
                )

    # ------------------------------------------------------------------
    # 模型后端与安全网
    # ------------------------------------------------------------------

    def _get_model_backend(self) -> ModelBackend:
        """获取当前模型后端，若未配置则自动创建 MockModelBackend。"""
        if self.model_backend is None:
            self.model_backend = MockModelBackend(self._model_config)
        return self.model_backend

    def set_model_backend(self, backend: ModelBackend) -> None:
        """设置模型后端。"""
        self.model_backend = backend

    def enable_safety_net(self, safety_net: Optional[ProductionSafetyNet] = None) -> None:
        """启用生产安全网。"""
        self.safety_net = safety_net or ProductionSafetyNet()
        self.safety_net.enable()

    def enable_learned_scheduler(
        self,
        scheduler: Optional[LearnedScheduler] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """启用学习调度器。"""
        if scheduler is not None:
            self.scheduler = scheduler
        elif model_path is not None:
            self.scheduler = LearnedScheduler(model_path=model_path)
        else:
            self.scheduler = LearnedScheduler()
        self._scheduler_model_path = model_path

    def _degrade_to_conservative(self, step_config: StepConfig) -> StepConfig:
        """将 step 配置降级到保守设置。"""
        return StepConfig(
            step_id=step_config.step_id,
            total_steps=step_config.total_steps,
            noise_level=step_config.noise_level,
            shadowkv_mode="conservative",
            ffn_mode="full",
            reuse_layers=step_config.reuse_layers,
            compression_target=step_config.compression_target,
            weight_precision_map=step_config.weight_precision_map,
            compute_path=step_config.compute_path,
            sensitivity_score=step_config.sensitivity_score,
            drift_score=step_config.drift_score,
        )

    def _is_real_backend(self) -> bool:
        """判断当前模型后端是否为真实推理后端（非 MockModelBackend）。"""
        backend = self._get_model_backend()
        return backend is not None and not isinstance(backend, MockModelBackend)

    def _get_backend_device(self) -> torch.device:
        """获取模型后端当前使用的 torch 设备。"""
        backend = self._get_model_backend()
        if hasattr(backend, "_device"):
            return backend._device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _extract_ffn_weights(self, backend: ModelBackend) -> Dict[str, torch.Tensor]:
        """从真实模型后端中提取 FFN 权重（取第 0 层作为代表）。"""
        model = getattr(backend, "_model", None)
        if model is not None and hasattr(model, "blocks") and len(model.blocks) > 0:
            block = model.blocks[0]
            return {
                "up": block.up_proj.weight.detach(),
                "down": block.down_proj.weight.detach(),
            }

        device = self._get_backend_device()
        cfg = backend.get_model_config()
        hidden_dim = cfg.get("hidden_dim", 128)
        intermediate_dim = cfg.get("intermediate_dim", hidden_dim * 4)
        return {
            "up": torch.randn(intermediate_dim, hidden_dim, device=device),
            "down": torch.randn(hidden_dim, intermediate_dim, device=device),
        }

    def _build_baseline_kv_metrics(self, kv_cache: Dict[int, KVCacheEntry]) -> Dict[str, Any]:
        """根据真实后端返回的 KV cache 构建基线 profiler 输入。"""
        precision_map: Dict[int, Dict[int, str]] = {}
        reuse_decision: Dict[int, Dict[str, Any]] = {}
        per_layer_mem: Dict[int, float] = {}
        total_mem = 0.0

        for lid, entry in kv_cache.items():
            k = entry.k_tensor
            v = entry.v_tensor
            mem_mb = (k.numel() * k.element_size() + v.numel() * v.element_size()) / (
                1024.0 * 1024.0
            )
            per_layer_mem[lid] = mem_mb
            total_mem += mem_mb
            num_heads = k.shape[1]
            precision_map[lid] = {h: entry.precision for h in range(num_heads)}
            reuse_decision[lid] = {"reused": entry.is_reused}

        return {
            "precision_map": precision_map,
            "reuse_decision": reuse_decision,
            "memory_mb": total_mem,
            "per_layer_memory": per_layer_mem,
            "baseline_total_memory": total_mem,
        }

    def _build_baseline_ffn_metrics(
        self,
        num_layers: int,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> Dict[str, Any]:
        """构建基线 FFN 指标（全精度、无稀疏）。"""
        compute_stats: Dict[int, Dict[str, float]] = {}
        sparse_per_layer: Dict[int, float] = {}
        for lid in range(num_layers):
            compute_stats[lid] = {
                "flops": batch_size * seq_len * hidden_dim * intermediate_dim * 2,
                "bandwidth_gb": 0.0,
                "compute_time_ms": 0.0,
            }
            sparse_per_layer[lid] = 0.0

        return {
            "compute_path": "full",
            "quantization": {},
            "sparse_update": {"overall": 0.0, "per_layer": sparse_per_layer},
            "compute_stats": compute_stats,
        }

    def _prepare_backend_step_inputs(
        self,
        x: torch.Tensor,
        prev_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """为真实后端准备一个 step 的输入数据字典。"""
        kv_previous = None
        if prev_inputs is not None:
            kv_previous = prev_inputs.get("kv_current")
        return {
            "query_current": x,
            "activation_current": x,
            "ffn_input_current": x,
            "ffn_output_current": x,
            "attention_scores": torch.empty(()),
            "kv_current": {"k": torch.empty(()), "v": torch.empty(())},
            "weights": self._extract_ffn_weights(self._get_model_backend()),
            "query_previous": prev_inputs.get("query_current") if prev_inputs else None,
            "activation_previous": prev_inputs.get("activation_current") if prev_inputs else None,
            "ffn_input_previous": prev_inputs.get("ffn_input_current") if prev_inputs else None,
            "ffn_output_previous": prev_inputs.get("ffn_output_current") if prev_inputs else None,
            "kv_previous": kv_previous,
            "_backend_input": x,
        }

    def _prepare_backend_inputs_from_result(
        self,
        result: Dict[str, Any],
        inputs: Dict[str, Any],
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        """将 ``backend.forward_step()`` 的结果转换为 Orchestrator 可用的输入。"""
        kv_cache = result.get("kv_cache", {})
        kv_entry = kv_cache.get(0)
        if kv_entry is not None and hasattr(kv_entry, "k_tensor") and hasattr(kv_entry, "v_tensor"):
            kv_current = {
                "k": kv_entry.k_tensor.detach(),
                "v": kv_entry.v_tensor.detach(),
            }
        else:
            kv_current = inputs.get("kv_current", {"k": torch.empty(()), "v": torch.empty(())})

        attention_scores = result.get(
            "attention_scores", inputs.get("attention_scores", torch.empty(()))
        )
        if isinstance(attention_scores, torch.Tensor):
            attention_scores = attention_scores.detach()

        return {
            "query_current": x,
            "activation_current": x,
            "ffn_input_current": x,
            "ffn_output_current": x,
            "attention_scores": attention_scores,
            "kv_current": kv_current,
            "weights": inputs.get("weights"),
            "_backend_full_kv_cache": kv_cache,
        }

    # ------------------------------------------------------------------
    # 基线运行
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        prompt: str,
        num_steps: int = 50,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """运行基线推理（无优化）。

        对应 ARCHITECTURE.md §4.2 的 Profiling Phase。

        Args:
            prompt: 输入提示文本（基线中仅用于记录）。
            num_steps: denoising 步数。
            on_step: 每完成一个 step 后调用的回调，接收 step 结果字典。

        Returns:
            基线结果字典。
        """
        if self.profiler is None:
            if not self._initialized:
                self.initialize()
            if self.profiler is None:
                raise RuntimeError("Agents not initialized. Call initialize() first.")

        self.logger.log_event(
            "run_baseline_start",
            f"Starting baseline profiling for {num_steps} steps.",
            data={"prompt": prompt, "num_steps": num_steps},
        )

        warmup_steps = self.config.get("warmup_steps", 5)
        device = self._get_backend_device()

        mc = self._model_config
        num_layers = mc.get("num_layers", 32)
        num_heads = mc.get("num_heads", 32)
        head_dim = mc.get("head_dim", 128)
        hidden_dim = mc.get("hidden_dim", 4096)
        intermediate_dim = mc.get("intermediate_dim", 11008)
        batch_size = mc.get("batch_size", 1)
        seq_len = mc.get("seq_len", 128)

        use_real = self._is_real_backend()
        backend = self._get_model_backend() if use_real else None

        if use_real and backend is not None:
            warmup_cfg = StepConfig(
                step_id=0,
                total_steps=1,
                noise_level=0.0,
                shadowkv_mode="conservative",
                ffn_mode="full",
            )
            for _ in range(warmup_steps):
                x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
                _ = backend.forward_step(x, warmup_cfg)
        else:
            for _ in range(warmup_steps):
                _ = self._simulate_step_tensors(
                    device,
                    batch_size,
                    seq_len,
                    num_layers,
                    num_heads,
                    head_dim,
                    hidden_dim,
                    intermediate_dim,
                )

        baseline_per_step: List[Dict[str, Any]] = []
        total_latency_ms = 0.0

        for step_id in range(num_steps):
            step_start = time.perf_counter()

            if use_real and backend is not None:
                x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
                step_config = StepConfig(
                    step_id=step_id,
                    total_steps=num_steps,
                    noise_level=0.0,
                    shadowkv_mode="conservative",
                    ffn_mode="full",
                    reuse_layers=[],
                )
                result = backend.forward_step(x, step_config)
                kv_metrics = self._build_baseline_kv_metrics(result.get("kv_cache", {}))
                ffn_metrics = self._build_baseline_ffn_metrics(
                    num_layers, batch_size, seq_len, hidden_dim, intermediate_dim
                )
                attention_scores = result.get("attention_scores", torch.empty(()))
            else:
                tensors = self._simulate_step_tensors(
                    device,
                    batch_size,
                    seq_len,
                    num_layers,
                    num_heads,
                    head_dim,
                    hidden_dim,
                    intermediate_dim,
                )
                attention_scores = tensors["attention_scores"]
                kv_metrics = {
                    "precision_map": {},
                    "reuse_decision": {},
                    "memory_mb": 0.0,
                    "per_layer_memory": {},
                    "baseline_total_memory": 0.0,
                }
                per_layer_mem = {}
                for lid in range(num_layers):
                    k_bytes = batch_size * num_heads * seq_len * head_dim * 4
                    v_bytes = k_bytes
                    mem_mb = (k_bytes + v_bytes) / (1024.0 * 1024.0)
                    per_layer_mem[lid] = mem_mb
                    kv_metrics["precision_map"][lid] = {h: "fp32" for h in range(num_heads)}
                    kv_metrics["reuse_decision"][lid] = {"reused": False}
                kv_metrics["per_layer_memory"] = per_layer_mem
                kv_metrics["memory_mb"] = sum(per_layer_mem.values())
                kv_metrics["baseline_total_memory"] = kv_metrics["memory_mb"]

                ffn_metrics = self._build_baseline_ffn_metrics(
                    num_layers, batch_size, seq_len, hidden_dim, intermediate_dim
                )

            qdrift_metrics = {
                "sensitivity_score": 0.0,
                "drift_score": 0.0,
                "dispatch": {"shadowkv_mode": "conservative", "ffn_mode": "full"},
            }

            perf_data = {
                "latency_ms": 0.0,
                "memory_mb": kv_metrics["memory_mb"],
                "gpu_utilization": 0.0,
                "tokens_per_sec": 0.0,
                "tokens_per_step": 1.0,
                "per_layer_ms": {},
            }

            accuracy_data = {
                "baseline_perplexity": 0.0,
                "optimized_perplexity": 0.0,
                "baseline_bleu": 0.0,
                "optimized_bleu": 0.0,
            }

            step_config = StepConfig(
                step_id=step_id,
                total_steps=num_steps,
                noise_level=0.0,
                shadowkv_mode="conservative",
                ffn_mode="full",
            )

            profiler_inputs = {
                "kv_metrics": kv_metrics,
                "qdrift_metrics": qdrift_metrics,
                "ffn_metrics": ffn_metrics,
                "performance": perf_data,
                "accuracy": accuracy_data,
            }

            result = self.profiler.on_step(step_config, profiler_inputs)
            step_latency = (time.perf_counter() - step_start) * 1000.0
            total_latency_ms += step_latency
            step_dict = {"step_id": step_id, "latency_ms": step_latency, "result": result}
            baseline_per_step.append(step_dict)
            if on_step is not None:
                on_step(step_dict)
            _ = attention_scores

        baseline_data = {
            "model": mc.get("name", "unknown"),
            "run_id": self._run_id,
            "num_steps": num_steps,
            "warmup_steps": warmup_steps,
            "latency": {
                "e2e_ms": total_latency_ms,
                "per_step_ms": {i: d["latency_ms"] for i, d in enumerate(baseline_per_step)},
            },
            "kv_cache": {
                "memory_mb": dict(kv_metrics["per_layer_memory"]),
            },
            "accuracy": {
                "perplexity_delta": 0.0,
                "bleu_drop": 0.0,
            },
            "alerts": [],
        }

        self.profiler.baseline_data = baseline_data
        self.profiler._baseline_collected = True

        output_dir = self.config.get("output_dir", "outputs/")
        os.makedirs(output_dir, exist_ok=True)
        baseline_path = os.path.join(output_dir, "profile_baseline.json")
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2, default=str)

        self.logger.log_event(
            "run_baseline_complete",
            f"Baseline profiling completed. Latency={total_latency_ms:.2f}ms",
            data={"output_path": baseline_path},
        )
        return baseline_data

    # ------------------------------------------------------------------
    # 优化运行
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """如果 Agent 尚未初始化，则自动初始化。"""
        if any(a is None for a in [self.profiler, self.shadowkv, self.qdrift, self.ffn_optimizer]):
            if not self._initialized:
                self.initialize()
            if any(
                a is None for a in [self.profiler, self.shadowkv, self.qdrift, self.ffn_optimizer]
            ):
                raise RuntimeError("Agents not initialized. Call initialize() first.")

    def _create_pipeline_context(
        self,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        close_loop: bool = False,
    ) -> PipelineContext:
        """根据配置创建本次运行的 PipelineContext。"""
        snapshot_dir = self.config.get("snapshot_dir")
        enable_snapshots = bool(snapshot_dir) or self.config.get("enable_snapshots", False)
        if enable_snapshots and snapshot_dir is None:
            snapshot_dir = os.path.join(self.config.get("output_dir", "outputs/"), "snapshots")

        def _typed_callback(state: StepState) -> None:
            if on_step is not None:
                on_step(state.to_dict())

        ctx = PipelineContext(
            run_id=self._run_id,
            start_time=time.perf_counter(),
            latency_budget_ms=self.global_state["latency_budget_ms"],
            memory_budget_mb=self.global_state["memory_budget_mb"],
            snapshot_dir=snapshot_dir,
            enable_snapshots=enable_snapshots,
            on_step=_typed_callback if on_step is not None else None,
            close_loop=close_loop,
        )
        if self._cancelled:
            ctx.cancel()
        return ctx

    def run_optimized(
        self,
        prompt: str,
        num_steps: int = 50,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        close_loop: bool = False,
    ) -> Dict[str, Any]:
        """运行优化后的推理。

        对应 ARCHITECTURE.md §4.2 的 Optimization Phase。

        Args:
            prompt: 输入提示文本。
            num_steps: denoising 步数。
            on_step: 每完成一个 step 后调用的回调。
            close_loop: 是否使用真实后端的输出作为下一步输入（实验性）。

        Returns:
            优化结果字典。
        """
        self._ensure_initialized()

        self.logger.log_event(
            "run_optimized_start",
            f"Starting optimized inference for {num_steps} steps.",
            data={"prompt": prompt, "num_steps": num_steps},
        )

        self.global_state["total_steps"] = num_steps
        self.step_results.clear()

        context = self._create_pipeline_context(on_step=on_step, close_loop=close_loop)
        self._pipeline_context = context

        total_latency_ms = 0.0
        for state in self._run_optimized_states(num_steps, context):
            total_latency_ms += state.latency_ms

        optimized_data = (
            self.profiler.optimized_data if self.profiler and self.profiler.optimized_data else {}
        )
        if not optimized_data:
            optimized_data = self._aggregate_optimized_data(num_steps, total_latency_ms)

        # Annotate early-stop metadata if enabled.
        actual_steps = len(self.step_results)
        early_stopped = False
        stopped_step: Optional[int] = None
        for sr in self.step_results:
            es = sr.get("outputs", {}).get("early_stop", {})
            if es.get("should_stop"):
                early_stopped = True
                stopped_step = sr.get("step_id")
                break
        optimized_data["early_stopped"] = early_stopped
        optimized_data["stopped_step"] = stopped_step
        optimized_data["actual_steps"] = actual_steps
        optimized_data["requested_steps"] = num_steps

        output_dir = self.config.get("output_dir", "outputs/")
        os.makedirs(output_dir, exist_ok=True)
        optimized_path = os.path.join(output_dir, "profile_optimized.json")
        with open(optimized_path, "w", encoding="utf-8") as f:
            json.dump(optimized_data, f, ensure_ascii=False, indent=2, default=str)

        baseline_data = self.profiler.baseline_data if self.profiler else {}
        if baseline_data:
            comparison_path = os.path.join(output_dir, "profile_comparison.html")
            from shadowinfer.profiler.reporter import HTMLReporter

            reporter = HTMLReporter()
            reporter.generate(baseline_data, optimized_data, comparison_path)
            self.logger.log_event(
                "run_optimized_complete",
                "Optimized inference and comparison report generated.",
                data={
                    "optimized_path": optimized_path,
                    "comparison_path": comparison_path,
                    "total_latency_ms": total_latency_ms,
                },
            )
        else:
            self.logger.log_event(
                "run_optimized_complete",
                "Optimized inference completed without baseline comparison.",
                data={"optimized_path": optimized_path, "total_latency_ms": total_latency_ms},
            )

        self._pipeline_context = None
        return optimized_data

    def run_stream(
        self,
        prompt: str,
        num_steps: int = 50,
        close_loop: bool = False,
    ):
        """流式运行优化推理，逐 step 生成 StepState 字典。

        Args:
            prompt: 输入提示文本。
            num_steps: denoising 步数。
            close_loop: 是否使用真实后端的输出作为下一步输入（实验性）。

        Yields:
            每个 step 的状态字典（与 run_step 返回格式一致）。
        """
        self._ensure_initialized()
        self.logger.log_event(
            "run_stream_start",
            f"Starting streaming optimized inference for {num_steps} steps.",
            data={"prompt": prompt, "num_steps": num_steps},
        )
        self.global_state["total_steps"] = num_steps
        self.step_results.clear()
        context = self._create_pipeline_context(close_loop=close_loop)
        self._pipeline_context = context
        try:
            for state in self._run_optimized_states(num_steps, context):
                yield state.to_dict()
        finally:
            self._pipeline_context = None

    def _run_optimized_states(
        self,
        num_steps: int,
        context: PipelineContext,
    ):
        """优化推理的 step 状态生成器（内部）。"""
        device = self._get_backend_device()
        mc = self._model_config
        num_layers = mc.get("num_layers", 32)
        num_heads = mc.get("num_heads", 32)
        head_dim = mc.get("head_dim", 128)
        hidden_dim = mc.get("hidden_dim", 4096)
        intermediate_dim = mc.get("intermediate_dim", 11008)
        batch_size = mc.get("batch_size", 1)
        seq_len = mc.get("seq_len", 128)

        use_real = self._is_real_backend()
        prev_tensors: Optional[Dict[str, Any]] = None
        prev_backend_inputs: Optional[Dict[str, Any]] = None

        for step_id in range(num_steps):
            if context.is_cancelled():
                self.logger.log_event(
                    "pipeline_cancelled",
                    f"Pipeline cancelled before step {step_id}.",
                    step_id=step_id,
                )
                break

            step_start = time.perf_counter()
            inputs = self._build_step_inputs(
                step_id=step_id,
                total_steps=num_steps,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                prev_tensors=prev_tensors,
                prev_backend_inputs=prev_backend_inputs,
                use_real=use_real,
                close_loop=context.close_loop,
            )

            state = StepState(
                step_id=step_id,
                total_steps=num_steps,
                inputs=inputs,
            )
            self._execute_step(state, context)
            state.latency_ms = (time.perf_counter() - step_start) * 1000.0

            # Uncertainty-aware early stopping
            if self.early_stopper is not None:
                obs_tensor = inputs.get("activation_current")
                if isinstance(obs_tensor, torch.Tensor):
                    es_state = self.early_stopper.observe(step_id, obs_tensor)
                    state.outputs["early_stop"] = {
                        "similarity": es_state.similarity,
                        "stable_steps": es_state.stable_steps,
                        "should_stop": es_state.should_stop,
                    }
                    if es_state.should_stop:
                        self.logger.log_event(
                            "early_stop",
                            f"Stopping early at step {step_id} due to stable output.",
                            data={"similarity": es_state.similarity},
                            step_id=step_id,
                        )
                        self.step_results.append(state.to_dict())
                        self._maybe_snapshot_step(state, context)
                        if context.on_step is not None:
                            context.on_step(state)
                        yield state
                        break

            self.step_results.append(state.to_dict())
            self._maybe_snapshot_step(state, context)
            if context.on_step is not None:
                context.on_step(state)

            yield state

            if use_real:
                prev_backend_inputs = inputs
            else:
                prev_tensors = inputs

            self.global_state["current_step"] = step_id + 1

    def _build_step_inputs(
        self,
        step_id: int,
        total_steps: int,
        device: torch.device,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        intermediate_dim: int,
        prev_tensors: Optional[Dict[str, Any]] = None,
        prev_backend_inputs: Optional[Dict[str, Any]] = None,
        use_real: bool = False,
        close_loop: bool = False,
    ) -> Dict[str, Any]:
        """构造单个 step 的输入字典（模拟或真实后端）。"""
        if use_real:
            x: torch.Tensor
            if close_loop and prev_backend_inputs is not None:
                x = prev_backend_inputs.get(
                    "_backend_next_input",
                    torch.randn(batch_size, seq_len, hidden_dim, device=device),
                )
            else:
                x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            return self._prepare_backend_step_inputs(x, prev_backend_inputs)

        tensors = self._simulate_step_tensors(
            device,
            batch_size,
            seq_len,
            num_layers,
            num_heads,
            head_dim,
            hidden_dim,
            intermediate_dim,
        )
        if prev_tensors is not None:
            tensors["query_previous"] = prev_tensors["query_current"]
            tensors["activation_previous"] = prev_tensors["activation_current"]
            tensors["ffn_input_previous"] = prev_tensors["ffn_input_current"]
            tensors["ffn_output_previous"] = prev_tensors["ffn_output_current"]
            tensors["kv_previous"] = prev_tensors["kv_current"]

        return {
            "query_current": tensors["query_current"],
            "activation_current": tensors["activation_current"],
            "ffn_input_current": tensors["ffn_input_current"],
            "ffn_output_current": tensors["ffn_output_current"],
            "attention_scores": tensors["attention_scores"],
            "kv_current": tensors["kv_current"],
            "weights": tensors["weights"],
            "query_previous": tensors.get("query_previous"),
            "activation_previous": tensors.get("activation_previous"),
            "ffn_input_previous": tensors.get("ffn_input_previous"),
            "ffn_output_previous": tensors.get("ffn_output_previous"),
            "kv_previous": tensors.get("kv_previous"),
        }

    def _run_single_step(
        self,
        step_id: int,
        total_steps: int,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行单个 denoising step（兼容旧接口）。

        Args:
            step_id: 当前 step 编号。
            total_steps: 总 step 数。
            inputs: 输入数据字典。

        Returns:
            当前 step 的结果字典。
        """
        state = StepState(
            step_id=step_id,
            total_steps=total_steps,
            inputs=inputs,
        )
        context = self._create_pipeline_context()
        self._execute_step(state, context)
        return state.to_dict()

    def _execute_step(
        self,
        state: StepState,
        context: PipelineContext,
    ) -> None:
        """执行 StepState 所描述的单个 step。

        对应 ARCHITECTURE.md §4.1 单 Step 执行流程。
        """
        assert self.qdrift is not None
        assert self.shadowkv is not None
        assert self.ffn_optimizer is not None
        assert self.profiler is not None

        step_id = state.step_id
        total_steps = state.total_steps
        inputs = state.inputs

        # Step 1: QDrift Agent 评估
        qdrift_result = self._run_qdrift(step_id, total_steps, inputs)
        state.qdrift_result = qdrift_result
        sensitivity_score = qdrift_result.get("sensitivity_score", 0.0)
        drift_score = qdrift_result.get("drift_score", 0.0)

        # Step 2: 构建 StepConfig
        step_config = self._build_step_config(step_id, total_steps, qdrift_result)
        state.step_config = step_config

        # Safety net pre-flight check
        if self.safety_net is not None:
            try:
                can_proceed, reason = self.safety_net.pre_flight_check(
                    step_id=step_id, step_config=step_config
                )
                if not can_proceed:
                    self.logger.log_alert(
                        "warning",
                        f"Safety net pre-flight check failed at step {step_id}: {reason}",
                        recommendation="Degrading to conservative settings.",
                        step_id=step_id,
                    )
                    step_config = self._degrade_to_conservative(step_config)
                    state.step_config = step_config
            except Exception as exc:
                self.logger.log_alert(
                    "warning",
                    f"Safety net pre-flight check error at step {step_id}: {exc}",
                    recommendation="Degrading to conservative settings.",
                    step_id=step_id,
                )
                step_config = self._degrade_to_conservative(step_config)
                state.step_config = step_config

        # 若使用真实后端，执行真实模型前向并将结果写回 inputs 供后续 Agent 使用
        if self._is_real_backend() and "_backend_input" in inputs:
            backend = self._get_model_backend()
            x = inputs["_backend_input"]
            try:
                backend_result = backend.forward_step(x, step_config)
                updated_inputs = self._prepare_backend_inputs_from_result(backend_result, inputs, x)
                inputs.update(updated_inputs)
                if context.close_loop:
                    next_x = self._derive_next_backend_input(backend_result, x)
                    if next_x is not None:
                        inputs["_backend_next_input"] = next_x
            except Exception as exc:
                self.logger.log_alert(
                    "warning",
                    f"Real backend forward_step failed at step {step_id}: {exc}",
                    recommendation="Falling back to simulation tensors for this step.",
                    step_id=step_id,
                )

        # Step 3: ShadowKV Agent 执行（逐层）
        # 真实后端的 KV cache 在 step 之间会追加 token，形状可能不一致，
        # 因此禁用跨 step 复用比较，避免维度不匹配。
        if self._is_real_backend():
            inputs["kv_previous"] = None
        state.kv_result = self._run_shadowkv(step_config, inputs, sensitivity_score, drift_score)

        # Step 4: FFN Optimizer Agent 执行（逐层）
        state.ffn_result = self._run_ffn(step_config, inputs, sensitivity_score)

        # Step 5: Profiler Agent 记录
        profiler_result = self._run_profiler(
            step_config,
            state.kv_result,
            state.qdrift_result,
            state.ffn_result,
        )
        state.profiler_result = profiler_result
        alerts = list(profiler_result.get("alerts", []))

        # Step 5.5: 插件 Agent 执行
        self._run_plugin_agents(step_config, state, inputs)

        # Record step experience for the learned scheduler
        if self.scheduler is not None:
            self._record_scheduler_experience(
                step_config, qdrift_result, profiler_result, step_id, total_steps
            )

        # Step 6: 全局状态更新与仲裁
        metrics = profiler_result.get("performance_metrics", {})
        self.global_state["memory_used_mb"] = metrics.get("memory", {}).get("total_mb", 0.0)
        acc_delta = (
            profiler_result.get("accuracy_metrics", {})
            .get("perplexity", {})
            .get("delta_percent", 0.0)
        )
        self.global_state["accuracy_drop_cumulative"] = max(
            self.global_state["accuracy_drop_cumulative"], acc_delta
        )

        constraint_alerts = self._evaluate_global_constraints(metrics)
        alerts.extend(constraint_alerts)

        conflicts = [a for a in alerts if a.get("level") in ("CRITICAL", "WARNING")]

        if any(a.get("level") == "CRITICAL" for a in alerts):
            step_config = self._arbitrate_step_config(step_config, alerts)
            state.step_config = step_config

        # Safety net post-flight check
        if self.safety_net is not None:
            try:
                latency_ms = (
                    profiler_result.get("performance_metrics", {})
                    .get("latency", {})
                    .get("e2e_ms", 0.0)
                )
                memory_mb = self.global_state["memory_used_mb"]
                accuracy_drop = self.global_state["accuracy_drop_cumulative"]
                has_critical = any(a.get("level") == "CRITICAL" for a in alerts)
                self.safety_net.post_flight_check(
                    latency_ms=latency_ms,
                    memory_mb=memory_mb,
                    accuracy_drop=accuracy_drop,
                    success=not has_critical,
                )
            except Exception as exc:
                self.logger.log_alert(
                    "warning",
                    f"Safety net post-flight check error at step {step_id}: {exc}",
                    recommendation="Degrading to conservative settings.",
                    step_id=step_id,
                )
                step_config = self._degrade_to_conservative(step_config)
                state.step_config = step_config

        state.alerts = alerts
        state.conflicts = conflicts
        state.resolution = {
            "accuracy_priority": any(
                a.get("metric") == "accuracy_drop" and a.get("level") == "CRITICAL" for a in alerts
            ),
            "memory_priority": any(
                a.get("metric") == "memory" and a.get("level") in ("CRITICAL", "WARNING")
                for a in alerts
            ),
            "latency_priority": any(
                a.get("metric") == "latency" and a.get("level") == "WARNING" for a in alerts
            ),
            "actions": [],
        }

        # 广播 step 结果到 bus
        msg = Message.create(
            source="orchestrator",
            target="broadcast",
            message_type=MESSAGE_TYPES.STEP_RESULT,
            payload={
                "step_id": step_id,
                "qdrift": state.qdrift_result,
                "shadowkv": state.kv_result,
                "ffn": state.ffn_result,
                "profiler": state.profiler_result,
                "resolution": state.resolution,
                "outputs": state.outputs,
            },
            step_id=step_id,
        )
        self.bus.broadcast(msg)

    def _run_qdrift(
        self,
        step_id: int,
        total_steps: int,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行 QDrift Agent 评估。"""
        noise_level = step_id / max(total_steps, 1)
        qdrift_inputs = {
            "step_id": step_id,
            "total_steps": total_steps,
            "noise_level": noise_level,
            "query_current": inputs["query_current"],
            "query_previous": inputs.get("query_previous"),
            "activation_current": inputs["activation_current"],
            "activation_previous": inputs.get("activation_previous"),
            "profiler_feedback": None,
        }
        qdrift_dummy_step = StepConfig(
            step_id=step_id,
            total_steps=total_steps,
            noise_level=noise_level,
            shadowkv_mode="balanced",
            ffn_mode="mixed",
        )
        qdrift_result = self.qdrift.on_step(qdrift_dummy_step, qdrift_inputs)

        if self.scheduler is not None:
            predicted_shadowkv_mode, predicted_ffn_mode = self.scheduler.predict(
                step_id=step_id,
                total_steps=total_steps,
                noise_level=noise_level,
                sensitivity_score=qdrift_result.get("sensitivity_score", 0.0),
                drift_score=qdrift_result.get("drift_score", 0.0),
                prev_latency_ms=self._prev_scheduler_state["latency_ms"],
                prev_memory_mb=self._prev_scheduler_state["memory_mb"],
                prev_accuracy_drop=self._prev_scheduler_state["accuracy_drop"],
            )
            qdrift_result["dispatch"] = {
                "shadowkv_mode": predicted_shadowkv_mode,
                "ffn_mode": predicted_ffn_mode,
            }

        return qdrift_result

    def _run_shadowkv(
        self,
        step_config: StepConfig,
        inputs: Dict[str, Any],
        sensitivity_score: float,
        drift_score: float,
    ) -> Dict[str, Any]:
        """执行 ShadowKV Agent（逐层）并聚合结果。"""
        num_layers = self._model_config.get("num_layers", 32)
        dispatch = inputs.get("qdrift_signal", {})

        # Compute memory pressure for the decision plane / eviction.
        total_mb = self.shadowkv.cache_manager.get_memory_usage_mb()
        budget_mb = max(1.0, self.global_state.get("memory_budget_mb", 8192.0))
        memory_pressure = min(1.0, total_mb / budget_mb)

        all_kv_results: List[Dict[str, Any]] = []
        for layer_id in range(num_layers):
            kv_inputs = {
                "attention_scores": inputs["attention_scores"],
                "kv_current": inputs["kv_current"],
                "kv_previous": inputs.get("kv_previous"),
                "layer_id": layer_id,
                "step_id": step_config.step_id,
                "total_steps": step_config.total_steps,
                "memory_pressure": memory_pressure,
                "qdrift_signal": {
                    "sensitivity_score": sensitivity_score,
                    "drift_score": drift_score,
                    "shadowkv_mode": dispatch.get("shadowkv_mode", step_config.shadowkv_mode),
                },
            }
            kv_result = self.shadowkv.on_step(step_config, kv_inputs)
            all_kv_results.append(kv_result)

        aggregated_kv = {
            "precision_map": {},
            "reuse_decision": {},
            "memory_mb": 0.0,
            "per_layer_memory": {},
        }
        for lid, kv_res in enumerate(all_kv_results):
            aggregated_kv["precision_map"][lid] = kv_res.get("precision_map", {})
            aggregated_kv["reuse_decision"][lid] = kv_res.get("reuse_decision", {})
            mem_mb = kv_res.get("compressed_kv", {}).get("memory_mb", 0.0)
            aggregated_kv["per_layer_memory"][lid] = mem_mb
            aggregated_kv["memory_mb"] += mem_mb

        # Optional: prefetch KV cache for the next step based on Q-drift.
        if self.shadowkv.prefetch_enabled and not self._is_real_backend():
            predicted_sensitivity = self.qdrift.predict_next_sensitivity(
                step_config.step_id, step_config.total_steps
            )
            predicted_drift = self.qdrift.predict_next_drift(
                inputs.get("query_current"), inputs.get("activation_current")
            )
            predicted_mode = dispatch.get("shadowkv_mode", step_config.shadowkv_mode)
            for layer_id, kv_res in enumerate(all_kv_results):
                self.shadowkv.prefetch_next_step(
                    {
                        "attention_scores": inputs["attention_scores"],
                        "kv_current": inputs["kv_current"],
                        "kv_previous": inputs.get("kv_previous"),
                        "layer_id": layer_id,
                        "step_id": step_config.step_id,
                        "total_steps": step_config.total_steps,
                    },
                    kv_res,
                    predicted_sensitivity=predicted_sensitivity,
                    predicted_drift=predicted_drift,
                    predicted_mode=predicted_mode,
                )

        return aggregated_kv

    def _run_ffn(
        self,
        step_config: StepConfig,
        inputs: Dict[str, Any],
        sensitivity_score: float,
    ) -> Dict[str, Any]:
        """执行 FFN Optimizer Agent（逐层）并聚合结果。"""
        num_layers = self._model_config.get("num_layers", 32)
        all_ffn_results: List[Dict[str, Any]] = []
        for layer_id in range(num_layers):
            ffn_inputs = {
                "ffn_input_current": inputs["ffn_input_current"],
                "ffn_input_previous": inputs.get("ffn_input_previous"),
                "ffn_output_previous": inputs.get("ffn_output_previous"),
                "weights": inputs["weights"],
                "qdrift_signal": {
                    "sensitivity_score": sensitivity_score,
                    "ffn_mode": step_config.ffn_mode,
                },
                "layer_id": layer_id,
            }
            ffn_result = self.ffn_optimizer.on_step(step_config, ffn_inputs)
            all_ffn_results.append(ffn_result)

        aggregated_ffn = {
            "compute_path": "full",
            "quantization": {},
            "sparse_update": {"overall": 0.0, "per_layer": {}},
            "compute_stats": {},
        }
        for lid, ffn_res in enumerate(all_ffn_results):
            aggregated_ffn["compute_stats"][lid] = ffn_res.get("compute_stats", {})
            sparse_update = ffn_res.get("sparse_update", {})
            if sparse_update:
                aggregated_ffn["sparse_update"]["per_layer"][lid] = sparse_update.get(
                    "changed_tokens_ratio", 0.0
                )
            quantization = ffn_res.get("quantization", {})
            if quantization:
                aggregated_ffn["quantization"].update(quantization)
            aggregated_ffn["compute_path"] = ffn_res.get("compute_path", "full")

        if aggregated_ffn["sparse_update"]["per_layer"]:
            ratios = list(aggregated_ffn["sparse_update"]["per_layer"].values())
            aggregated_ffn["sparse_update"]["overall"] = sum(ratios) / len(ratios)

        return aggregated_ffn

    def _run_profiler(
        self,
        step_config: StepConfig,
        aggregated_kv: Dict[str, Any],
        qdrift_result: Dict[str, Any],
        aggregated_ffn: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行 Profiler Agent 并返回结果。"""
        perf_data = {
            "latency_ms": 0.0,
            "memory_mb": aggregated_kv["memory_mb"],
            "gpu_utilization": 0.0,
            "tokens_per_sec": 0.0,
            "tokens_per_step": 1.0,
            "per_layer_ms": {},
        }
        accuracy_data = {
            "baseline_perplexity": 0.0,
            "optimized_perplexity": 0.0,
            "baseline_bleu": 0.0,
            "optimized_bleu": 0.0,
        }
        profiler_inputs = {
            "kv_metrics": aggregated_kv,
            "qdrift_metrics": qdrift_result,
            "ffn_metrics": aggregated_ffn,
            "performance": perf_data,
            "accuracy": accuracy_data,
        }
        return self.profiler.on_step(step_config, profiler_inputs)

    def _record_scheduler_experience(
        self,
        step_config: StepConfig,
        qdrift_result: Dict[str, Any],
        profiler_result: Dict[str, Any],
        step_id: int,
        total_steps: int,
    ) -> None:
        """将本 step 经验记录到学习调度器。"""
        perf_metrics = profiler_result.get("performance_metrics", {})
        acc_metrics = profiler_result.get("accuracy_metrics", {})
        latency_ms = perf_metrics.get("latency", {}).get("e2e_ms", 0.0)
        memory_mb = perf_metrics.get("memory", {}).get("total_mb", 0.0)
        accuracy_drop = acc_metrics.get("perplexity", {}).get("delta_percent", 0.0)
        experience = StepExperience(
            step_id=step_id,
            total_steps=total_steps,
            noise_level=step_id / max(total_steps, 1),
            sensitivity_score=qdrift_result.get("sensitivity_score", 0.0),
            drift_score=qdrift_result.get("drift_score", 0.0),
            prev_latency_ms=self._prev_scheduler_state["latency_ms"],
            prev_memory_mb=self._prev_scheduler_state["memory_mb"],
            prev_accuracy_drop=self._prev_scheduler_state["accuracy_drop"],
            shadowkv_mode=step_config.shadowkv_mode,
            ffn_mode=step_config.ffn_mode,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            accuracy_drop=accuracy_drop,
        )
        self.scheduler.add_experience(experience)
        self._prev_scheduler_state["latency_ms"] = latency_ms
        self._prev_scheduler_state["memory_mb"] = memory_mb
        self._prev_scheduler_state["accuracy_drop"] = accuracy_drop

    def _build_policy_context(self, alerts: List[Dict]) -> PolicyContext:
        """Build a policy evaluation context from current global state and alerts."""
        memory_budget = max(1.0, self.global_state["memory_budget_mb"])
        latency_budget = max(1.0, self.global_state["latency_budget_ms"])
        memory_used = self.global_state.get("memory_used_mb", 0.0)
        latency_ms = self.global_state.get("latency_ms", 0.0)

        alert_metrics = {alert.get("metric", ""): alert.get("level", "") for alert in alerts}

        return PolicyContext(
            {
                "accuracy_drop": self.global_state["accuracy_drop_cumulative"],
                "memory_ratio": memory_used / memory_budget,
                "latency_ratio": latency_ms / latency_budget,
                "alert_accuracy_critical": alert_metrics.get("accuracy_drop") == "CRITICAL",
                "alert_memory_warning": alert_metrics.get("memory") in ("CRITICAL", "WARNING"),
                "alert_latency_warning": alert_metrics.get("latency") == "WARNING",
                "alerts": alerts,
            }
        )

    def _arbitrate_step_config(self, step_config: StepConfig, alerts: List[Dict]) -> StepConfig:
        """冲突仲裁（对应 AGENTS.md §3.3）。

        Arbitration is driven by the declarative policy loaded in ``self.policy``.
        The bundled default policy replicates the previous hard-coded rules;
        users can override it via ``policy_path`` in their config.
        """
        shadowkv_mode = step_config.shadowkv_mode
        ffn_mode = step_config.ffn_mode
        modified = False

        context = self._build_policy_context(alerts)
        actions = self.policy.evaluate(context)

        sid = step_config.step_id
        if actions.get("shadowkv.mode") is not None:
            new_mode = actions["shadowkv.mode"]
            if new_mode != shadowkv_mode:
                shadowkv_mode = new_mode
                modified = True
                self.logger.log_alert(
                    "info",
                    f"Step {sid}: Policy sets shadowkv_mode to {shadowkv_mode}.",
                    step_id=sid,
                )
        if actions.get("ffn.mode") is not None:
            new_mode = actions["ffn.mode"]
            if new_mode != ffn_mode:
                ffn_mode = new_mode
                modified = True
                self.logger.log_alert(
                    "info",
                    f"Step {sid}: Policy sets ffn_mode to {ffn_mode}.",
                    step_id=sid,
                )

        # Backward-compatible fallback for tests/configs that do not use a policy.
        if not modified:
            for alert in alerts:
                if alert.get("metric") == "accuracy_drop" and alert.get("level") == "CRITICAL":
                    shadowkv_mode = "conservative"
                    ffn_mode = "full"
                    modified = True
                    self.logger.log_alert(
                        "critical",
                        f"Step {sid}: Accuracy drop CRITICAL. Rollback to full precision.",
                        recommendation="Use conservative ShadowKV and full FFN.",
                        step_id=sid,
                    )
                    break

            for alert in alerts:
                if alert.get("metric") == "memory" and alert.get("level") in (
                    "CRITICAL",
                    "WARNING",
                ):
                    if shadowkv_mode != "aggressive":
                        shadowkv_mode = "aggressive"
                        modified = True
                        self.logger.log_alert(
                            "warning",
                            f"Step {sid}: Memory budget exceeded. Tighten ShadowKV compression.",
                            recommendation="Switch to aggressive ShadowKV mode.",
                            step_id=sid,
                        )
                    break

            for alert in alerts:
                if alert.get("metric") == "latency" and alert.get("level") == "WARNING":
                    if ffn_mode != "sparse" and shadowkv_mode != "aggressive":
                        if not any(a.get("metric") == "accuracy_drop" for a in alerts):
                            ffn_mode = "sparse"
                            modified = True
                            self.logger.log_alert(
                                "warning",
                                f"Step {sid}: Latency budget exceeded. Adopt more aggressive FFN.",
                                recommendation="Switch to sparse FFN mode.",
                                step_id=sid,
                            )
                    break

        if modified:
            return StepConfig(
                step_id=step_config.step_id,
                total_steps=step_config.total_steps,
                noise_level=step_config.noise_level,
                shadowkv_mode=shadowkv_mode,
                ffn_mode=ffn_mode,
                reuse_layers=step_config.reuse_layers,
                compression_target=step_config.compression_target,
                weight_precision_map=step_config.weight_precision_map,
                compute_path=step_config.compute_path,
                sensitivity_score=step_config.sensitivity_score,
                drift_score=step_config.drift_score,
            )
        return step_config

    def _evaluate_global_constraints(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查全局约束，返回告警列表（内部实现）。"""
        alerts: List[Dict] = []
        memory_mb = metrics.get("memory", {}).get("total_mb", 0.0)
        latency_ms = metrics.get("latency", {}).get("e2e_ms", 0.0)
        memory_budget = self.global_state["memory_budget_mb"]
        latency_budget = self.global_state["latency_budget_ms"]
        acc_drop = self.global_state["accuracy_drop_cumulative"]

        current_step = self.global_state["current_step"]
        if memory_budget > 0 and memory_mb > memory_budget:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "message": f"Memory budget exceeded: {memory_mb:.1f}MB > {memory_budget:.1f}MB",
                    "recommendation": "Immediately increase compression or reduce batch size.",
                    "metric": "memory",
                    "value": memory_mb,
                    "step_id": current_step,
                }
            )

        if latency_budget > 0 and latency_ms > latency_budget:
            alerts.append(
                {
                    "level": "WARNING",
                    "message": (
                        f"Latency budget exceeded: {latency_ms:.1f}ms > {latency_budget:.1f}ms"
                    ),
                    "recommendation": "Adopt more aggressive optimization strategies.",
                    "metric": "latency",
                    "value": latency_ms,
                    "step_id": current_step,
                }
            )

        if acc_drop > 0.01:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "message": f"Cumulative accuracy drop exceeded 1%: {acc_drop * 100:.2f}%",
                    "recommendation": "Rollback to full precision for remaining steps.",
                    "metric": "accuracy_drop",
                    "value": acc_drop,
                    "step_id": self.global_state["current_step"],
                }
            )

        return alerts

    def _build_step_config(self, step_id: int, total_steps: int, qdrift_result: Dict) -> StepConfig:
        """根据 QDrift 结果构建 StepConfig。"""
        dispatch = qdrift_result.get("dispatch", {})
        sensitivity_score = qdrift_result.get("sensitivity_score", 0.0)
        drift_score = qdrift_result.get("drift_score", 0.0)
        noise_level = step_id / max(total_steps, 1)

        shadowkv_mode = dispatch.get("shadowkv_mode", "balanced")
        ffn_mode = dispatch.get("ffn_mode", "mixed")

        return StepConfig(
            step_id=step_id,
            total_steps=total_steps,
            noise_level=noise_level,
            shadowkv_mode=shadowkv_mode,
            ffn_mode=ffn_mode,
            sensitivity_score=sensitivity_score,
            drift_score=drift_score,
        )

    def _derive_next_backend_input(
        self,
        backend_result: Dict[str, Any],
        current_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """从真实后端结果推导下一个 step 的输入（close_loop 实验性）。"""
        output = backend_result.get("output")
        if not isinstance(output, torch.Tensor):
            return None
        if output.shape == current_x.shape:
            return output.detach()
        # 若输出为 logits [..., vocab_size]，无法直接作为 hidden，回退到 None
        if output.dim() == current_x.dim() and output.shape[:-1] == current_x.shape[:-1]:
            return None
        return None

    # ------------------------------------------------------------------
    # 数据聚合与辅助
    # ------------------------------------------------------------------

    def _aggregate_optimized_data(self, num_steps: int, total_latency_ms: float) -> Dict[str, Any]:
        """聚合优化后的数据字典。"""
        if not self.step_results:
            return {}

        kv_cache_data = {"memory_mb": {}, "precision_distribution": {}, "reuse_rate": {}}
        ffn_data = {"compute_load": {}, "sparse_update_ratio": 0.0}
        qdrift_data = {"step_hit_rate": {}, "activation_delta": {}}
        latency_per_step = {}
        alerts_all: List[Dict] = []

        for sr in self.step_results:
            sid = sr["step_id"]
            latency_per_step[sid] = 0.0
            kv = sr.get("shadowkv", sr.get("kv", {}))
            for lid, mem in kv.get("per_layer_memory", {}).items():
                if lid not in kv_cache_data["memory_mb"]:
                    kv_cache_data["memory_mb"][lid] = []
                kv_cache_data["memory_mb"][lid].append(mem)
            qdrift_data["step_hit_rate"][sid] = 1.0 if sr.get("qdrift", {}).get("dispatch") else 0.0
            alerts_all.extend(sr.get("alerts", []))

        for lid, mems in kv_cache_data["memory_mb"].items():
            kv_cache_data["memory_mb"][lid] = sum(mems) / len(mems)

        return {
            "model": self._model_config.get("name", "unknown"),
            "run_id": self._run_id,
            "num_steps": num_steps,
            "latency": {
                "e2e_ms": total_latency_ms,
                "per_step_ms": latency_per_step,
            },
            "kv_cache": kv_cache_data,
            "ffn": ffn_data,
            "qdrift": qdrift_data,
            "alerts": alerts_all,
        }

    def _get_step_tensors(
        self,
        device: torch.device,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> Dict[str, Any]:
        """获取单个 step 的输入张量。"""
        backend = self._get_model_backend()
        if isinstance(backend, MockModelBackend):
            return self._simulate_step_tensors(
                device,
                batch_size,
                seq_len,
                num_layers,
                num_heads,
                head_dim,
                hidden_dim,
                intermediate_dim,
            )

        try:
            return self._generate_tensors_from_backend(
                backend,
                device,
                batch_size,
                seq_len,
                num_layers,
                num_heads,
                head_dim,
                hidden_dim,
                intermediate_dim,
            )
        except Exception as exc:
            self.logger.log_alert(
                "warning",
                f"Model backend tensor generation failed, falling back to simulation: {exc}",
                recommendation="Check model backend implementation.",
            )
        return self._simulate_step_tensors(
            device,
            batch_size,
            seq_len,
            num_layers,
            num_heads,
            head_dim,
            hidden_dim,
            intermediate_dim,
        )

    def _generate_tensors_from_backend(
        self,
        backend: ModelBackend,
        device: torch.device,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> Dict[str, Any]:
        """使用 ModelBackend 生成单个 step 的输入张量。"""
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        step_cfg = StepConfig(
            step_id=0,
            total_steps=1,
            noise_level=0.0,
            shadowkv_mode="balanced",
            ffn_mode="mixed",
        )
        result = backend.forward_step(x, step_cfg)
        output = result.get("output", torch.randn_like(x))
        attention_scores = result.get(
            "attention_scores",
            torch.randn(batch_size, num_heads, seq_len, seq_len, device=device),
        )
        kv_cache = result.get("kv_cache", {})

        kv_entry = kv_cache.get(0)
        if kv_entry is not None and hasattr(kv_entry, "k_tensor") and hasattr(kv_entry, "v_tensor"):
            k = kv_entry.k_tensor
            v = kv_entry.v_tensor
        else:
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        return {
            "query_current": output,
            "activation_current": output,
            "ffn_input_current": output,
            "ffn_output_current": output,
            "attention_scores": attention_scores,
            "kv_current": {"k": k, "v": v},
            "weights": {
                "up": torch.randn(intermediate_dim, hidden_dim, device=device),
                "down": torch.randn(hidden_dim, intermediate_dim, device=device),
            },
        }

    def _simulate_step_tensors(
        self,
        device: torch.device,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> Dict[str, Any]:
        """生成单个 step 的模拟张量数据。"""
        return {
            "query_current": torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
            "activation_current": torch.randn(batch_size, seq_len, hidden_dim, device=device),
            "ffn_input_current": torch.randn(batch_size, seq_len, hidden_dim, device=device),
            "ffn_output_current": torch.randn(batch_size, seq_len, hidden_dim, device=device),
            "attention_scores": torch.randn(batch_size, num_heads, seq_len, seq_len, device=device),
            "kv_current": {
                "k": torch.randn(batch_size, num_heads, seq_len, head_dim, device=device),
                "v": torch.randn(batch_size, num_heads, seq_len, head_dim, device=device),
            },
            "weights": {
                "up": torch.randn(intermediate_dim, hidden_dim, device=device),
                "down": torch.randn(hidden_dim, intermediate_dim, device=device),
            },
        }

    # ------------------------------------------------------------------
    # 完整流水线
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        prompt: str,
        num_steps: int = 50,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        close_loop: bool = False,
    ) -> Dict[str, Any]:
        """完整流水线：基线 + 优化 + 对比报告。

        对应 ARCHITECTURE.md §4.2 端到端推理流程：
        1. 初始化 Agent
        2. 运行基线（可选）
        3. 运行优化
        4. 验证对比
        5. 输出综合报告

        Args:
            prompt: 输入提示文本。
            num_steps: denoising 步数。
            on_step: 每完成一个优化 step 后调用的回调。
            close_loop: 是否使用真实后端输出作为下一步输入（实验性）。

        Returns:
            综合结果字典。
        """
        self.logger.log_event(
            "pipeline_start",
            "Starting full pipeline: baseline + optimization + comparison.",
            data={"prompt": prompt, "num_steps": num_steps},
        )

        baseline_result = self.run_baseline(prompt, num_steps)
        optimized_result = self.run_optimized(
            prompt, num_steps, on_step=on_step, close_loop=close_loop
        )

        baseline_latency = baseline_result.get("latency", {}).get("e2e_ms", 0.0)
        optimized_latency = optimized_result.get("latency", {}).get("e2e_ms", 0.0)
        speedup = baseline_latency / optimized_latency if optimized_latency > 0 else 0.0

        baseline_kv_mem = sum(baseline_result.get("kv_cache", {}).get("memory_mb", {}).values())
        optimized_kv_mem = sum(optimized_result.get("kv_cache", {}).get("memory_mb", {}).values())
        memory_savings = 1.0 - (optimized_kv_mem / baseline_kv_mem) if baseline_kv_mem > 0 else 0.0

        ppl_delta = optimized_result.get("accuracy", {}).get("perplexity_delta", 0.0)
        bleu_drop = optimized_result.get("accuracy", {}).get("bleu_drop", 0.0)

        summary = {
            "model": self._model_config.get("name", "unknown"),
            "run_id": self._run_id,
            "baseline_latency_ms": baseline_latency,
            "optimized_latency_ms": optimized_latency,
            "speedup": speedup,
            "memory_savings_ratio": memory_savings,
            "accuracy_drop": ppl_delta,
            "bleu_drop": bleu_drop,
            "total_steps": num_steps,
            "baseline_path": os.path.join(
                self.config.get("output_dir", "outputs/"), "profile_baseline.json"
            ),
            "optimized_path": os.path.join(
                self.config.get("output_dir", "outputs/"), "profile_optimized.json"
            ),
            "comparison_path": os.path.join(
                self.config.get("output_dir", "outputs/"), "profile_comparison.html"
            ),
        }

        output_dir = self.config.get("output_dir", "outputs/")
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        self.logger.log_event(
            "pipeline_complete",
            "Full pipeline completed.",
            data=summary,
        )
        return summary

    def get_summary(self) -> Dict[str, Any]:
        """获取运行汇总。"""
        agent_states = {name: agent.state.value for name, agent in self.registry.get_all().items()}
        bus_stats = self.bus.get_message_stats()
        return {
            "run_id": self._run_id,
            "global_state": dict(self.global_state),
            "agent_states": agent_states,
            "bus_stats": bus_stats,
            "total_steps_executed": len(self.step_results),
        }

    def shutdown(self) -> None:
        """关闭所有 Agent。"""
        self.logger.log_event("shutdown", "Orchestrator shutting down all agents.")

        if self.scheduler is not None and self.scheduler.experiences:
            try:
                self.scheduler.train()
                save_path = self._scheduler_model_path
                if save_path is not None:
                    self.scheduler.save(save_path)
                    self.logger.log_event(
                        "scheduler_save",
                        f"Learned scheduler model saved to {save_path}.",
                    )
            except Exception as exc:
                self.logger.log_alert(
                    "warning",
                    f"Learned scheduler training/saving failed: {exc}",
                    recommendation="Check scheduler model and experiences.",
                )

        for agent in list(self.registry):
            try:
                agent.on_shutdown()
                self.logger.log_event(
                    "agent_shutdown",
                    f"Agent {agent.name} shut down successfully.",
                )
            except Exception as exc:
                self.logger.log_alert(
                    "error",
                    f"Agent {agent.name} shutdown failed: {exc}",
                    recommendation="Check agent logs for details.",
                )

        self.registry.clear()
        self.bus.clear_log()
        self.logger.log_event("shutdown", "Orchestrator shutdown complete.")
        self.logger.flush()

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "Orchestrator":
        """上下文管理器入口。"""
        if not self._initialized:
            self.initialize()
        self.logger.log_event("context", "Orchestrator entered context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出，自动关闭。"""
        if exc_type is not None:
            self.logger.log_alert(
                "critical",
                f"Exception during orchestration: {exc_val}",
                recommendation="Review logs and stack trace.",
            )
        self.shutdown()

    # ------------------------------------------------------------------
    # 测试兼容与便捷接口
    # ------------------------------------------------------------------

    @property
    def agents(self) -> Dict[str, BaseAgent]:
        """返回已注册 Agent 的字典映射（测试兼容）。"""
        return dict(self.registry.get_all())

    def run_step(self, step_id: int, total_steps: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """单 step 执行（测试兼容）。"""
        return self._run_single_step(step_id, total_steps, inputs)

    def _make_dummy_inputs(self, step_id: int, total_steps: int) -> Dict[str, Any]:
        """生成模拟输入（测试兼容）。"""
        device = torch.device("cpu")
        mc = self._model_config or {
            "num_layers": 4,
            "num_heads": 32,
            "head_dim": 128,
            "hidden_dim": 4096,
            "intermediate_dim": 11008,
            "batch_size": 1,
            "seq_len": 128,
        }
        num_layers = mc.get("num_layers", 4)
        num_heads = mc.get("num_heads", 32)
        head_dim = mc.get("head_dim", 128)
        hidden_dim = mc.get("hidden_dim", 4096)
        intermediate_dim = mc.get("intermediate_dim", 11008)
        batch_size = mc.get("batch_size", 1)
        seq_len = mc.get("seq_len", 128)
        return self._simulate_step_tensors(
            device,
            batch_size,
            seq_len,
            num_layers,
            num_heads,
            head_dim,
            hidden_dim,
            intermediate_dim,
        )

    def run_inference(
        self,
        model: str,
        prompt: str,
        num_steps: int,
    ) -> InferenceResult:
        """推理方法（测试兼容）。"""
        if not self._initialized:
            self.initialize()
        result = InferenceResult()
        context = self._create_pipeline_context()
        for state in self._run_optimized_states(num_steps, context):
            result.step_results.append(state.to_dict())

        result.accuracy_drop = self._compute_accuracy_drop(result)
        result.latency_ms = self._compute_total_latency(result)
        result.memory_mb = self._compute_peak_memory(result)
        result.constraints_satisfied = self._check_global_constraints(result)
        return result

    def _compute_accuracy_drop(self, result: InferenceResult) -> float:
        """计算平均精度下降。"""
        deltas = []
        for step_result in result.step_results:
            profiler = step_result.get("profiler", {})
            acc = profiler.get("accuracy_metrics", {})
            delta = acc.get("perplexity", {}).get("delta_percent", 0.0)
            deltas.append(delta)
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def _compute_total_latency(self, result: InferenceResult) -> float:
        """计算总延迟。"""
        total = 0.0
        for step_result in result.step_results:
            perf = step_result.get("profiler", {}).get("performance_metrics", {})
            lat = perf.get("latency", {}).get("e2e_ms", 0.0)
            if lat == 0.0:
                lat = 10.0
            total += lat
        return total

    def _compute_peak_memory(self, result: InferenceResult) -> float:
        """计算峰值内存。"""
        peak = 0.0
        for step_result in result.step_results:
            perf = step_result.get("profiler", {}).get("performance_metrics", {})
            mem = perf.get("memory", {}).get("total_mb", 0.0)
            if mem == 0.0:
                mem = step_result.get("shadowkv", {}).get("memory_mb", 0.0)
            peak = max(peak, mem)
        return peak

    def _check_conflicts(self, profiler_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """测试兼容冲突检查。"""
        conflicts = []
        alerts = profiler_output.get("alerts", [])
        for alert in alerts:
            if alert.get("level") in ("CRITICAL", "WARNING"):
                conflicts.append(
                    {
                        "type": alert.get("metric", "unknown"),
                        "level": alert.get("level"),
                        "message": alert.get("message", ""),
                        "value": alert.get("value", 0.0),
                    }
                )
        return conflicts

    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """测试兼容冲突仲裁。"""
        resolution = {
            "accuracy_priority": False,
            "memory_priority": False,
            "latency_priority": False,
            "actions": [],
        }
        for conflict in conflicts:
            metric = conflict.get("type", "")
            level = conflict.get("level", "")
            if metric == "accuracy_drop" and level == "CRITICAL":
                resolution["accuracy_priority"] = True
                resolution["actions"].append("rollback_to_full_precision")
            elif metric == "memory" and level == "WARNING":
                resolution["memory_priority"] = True
                resolution["actions"].append("increase_kv_compression")
            elif metric == "latency" and level == "WARNING":
                resolution["latency_priority"] = True
                resolution["actions"].append("adopt_aggressive_ffn")
        return resolution

    def _check_global_constraints(self, result: Union[InferenceResult, Dict[str, Any]]) -> bool:
        """测试兼容全局约束检查（接受 InferenceResult 或 dict）。"""
        if isinstance(result, InferenceResult):
            acc_ok = result.accuracy_drop <= 0.01
            lat_ok = result.latency_ms <= self.global_state.get("latency_budget_ms", 100.0)
            mem_ok = result.memory_mb <= self.global_state.get("memory_budget_mb", 8192.0)
            return acc_ok and lat_ok and mem_ok
        if isinstance(result, dict):
            return self._evaluate_global_constraints(result) == []
        return True

    # ------------------------------------------------------------------
    # 取消与事件溯源快照
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """请求取消当前正在运行的推理流水线。"""
        self._cancelled = True
        if self._pipeline_context is not None:
            self._pipeline_context.cancel()
        self.logger.log_event("pipeline_cancel_request", "Pipeline cancellation requested.")

    def _maybe_snapshot_step(
        self,
        state: StepState,
        context: PipelineContext,
    ) -> None:
        """如果启用，将 step 状态持久化为 JSON 快照。"""
        if not context.enable_snapshots or not context.snapshot_dir:
            return
        try:
            os.makedirs(context.snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(
                context.snapshot_dir,
                f"step_{state.step_id:04d}_{context.run_id}.json",
            )
            payload = {
                "run_id": context.run_id,
                "step_id": state.step_id,
                "total_steps": state.total_steps,
                "latency_ms": state.latency_ms,
                "step_config": {
                    "shadowkv_mode": state.step_config.shadowkv_mode if state.step_config else None,
                    "ffn_mode": state.step_config.ffn_mode if state.step_config else None,
                    "noise_level": state.step_config.noise_level if state.step_config else None,
                },
                "qdrift": state.qdrift_result,
                "shadowkv": state.kv_result,
                "ffn": state.ffn_result,
                "profiler": state.profiler_result,
                "alerts": state.alerts,
                "resolution": state.resolution,
            }
            safe_payload = self._snapshot_safe(payload)
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(safe_payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.logger.log_alert(
                "warning",
                f"Failed to write step snapshot for step {state.step_id}: {exc}",
                recommendation="Check snapshot directory permissions.",
                step_id=state.step_id,
            )

    @staticmethod
    def _snapshot_default(obj: Any) -> Any:
        """JSON 序列化 fallback：将 torch.Tensor 转为轻量描述，其他对象转字符串。"""
        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
        if isinstance(obj, (KVCacheEntry, StepConfig)):
            return str(obj)
        return str(obj)

    @staticmethod
    def _snapshot_safe(obj: Any) -> Any:
        """递归地将 step 快照数据转为 JSON 安全结构。

        - torch.Tensor 转为轻量元数据描述
        - dict 的 key 强制转为 str（兼容 tuple-key 的 reuse map）
        - list/tuple 递归处理元素
        """
        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
        if isinstance(obj, dict):
            return {str(k): Orchestrator._snapshot_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Orchestrator._snapshot_safe(v) for v in obj]
        if isinstance(obj, (KVCacheEntry, StepConfig)):
            return str(obj)
        return obj

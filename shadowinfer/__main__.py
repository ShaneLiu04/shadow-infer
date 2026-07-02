"""ShadowInfer CLI 入口。

使用方法：
    python -m shadowinfer profiler --model Fast-dLLM-v2-7B --config configs/profiler_full.yaml
    python -m shadowinfer optimize --model Fast-dLLM-v2-7B --config configs/optimize_full.yaml
    python -m shadowinfer serve --model Fast-dLLM-v2-7B --config configs/optimize_full.yaml

对应文档：ARCHITECTURE.md §4.2 端到端推理流程
版本：v3.1
"""

__version__ = "3.2.2"

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from shadowinfer.core.model_backend import ModelBackend


def _backend_choices() -> list[str]:
    """Return all registered backend names plus the ``auto`` alias."""
    from shadowinfer.core.backends.registry import get_backend_registry

    registry = get_backend_registry()
    return ["auto"] + registry.list_names()


def _default_device() -> str:
    """根据 PyTorch 可用性返回默认设备。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="ShadowInfer — Diffusion LLM 推理优化与 Profiling")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # profiler 命令
    profiler_parser = subparsers.add_parser("profiler", help="运行性能分析")
    profiler_parser.add_argument("--model", required=True, help="模型名称")
    profiler_parser.add_argument(
        "--config", default="configs/profiler_full.yaml", help="配置文件路径"
    )
    profiler_parser.add_argument("--output", default="outputs/profiling", help="输出目录")
    profiler_parser.add_argument("--prompt", default="Hello world", help="测试提示")
    profiler_parser.add_argument("--num-steps", type=int, default=50, help="denoising 步数")
    profiler_parser.add_argument(
        "--backend",
        choices=_backend_choices(),
        default="auto",
        help="模型后端（默认 auto，使用 PyTorchModelBackend）。可用后端见 shadowinfer --help 或 registry 文档",
    )
    profiler_parser.add_argument(
        "--device", default=_default_device(), help="运行设备（默认 cuda 若可用否则 cpu）"
    )
    profiler_parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于可复现的模型初始化（默认 42）"
    )
    profiler_parser.add_argument(
        "--scheduler", action="store_true", help="启用学习调度器覆盖 Q-drift 的 dispatch"
    )
    profiler_parser.add_argument(
        "--scheduler-model",
        default="outputs/scheduler_model.pt",
        help="学习调度器模型保存/加载路径（默认 outputs/scheduler_model.pt）",
    )

    # optimize 命令
    optimize_parser = subparsers.add_parser("optimize", help="运行优化推理")
    optimize_parser.add_argument("--model", required=True, help="模型名称")
    optimize_parser.add_argument(
        "--config", default="configs/optimize_full.yaml", help="配置文件路径"
    )
    optimize_parser.add_argument("--baseline", help="基线数据路径（用于对比）")
    optimize_parser.add_argument("--output", default="outputs/optimization", help="输出目录")
    optimize_parser.add_argument("--prompt", default="Hello world", help="测试提示")
    optimize_parser.add_argument("--num-steps", type=int, default=50, help="denoising 步数")
    optimize_parser.add_argument(
        "--backend",
        choices=_backend_choices(),
        default="auto",
        help="模型后端（默认 auto，使用 PyTorchModelBackend）。可用后端见 shadowinfer --help 或 registry 文档",
    )
    optimize_parser.add_argument(
        "--device", default=_default_device(), help="运行设备（默认 cuda 若可用否则 cpu）"
    )
    optimize_parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于可复现的模型初始化（默认 42）"
    )
    optimize_parser.add_argument(
        "--scheduler", action="store_true", help="启用学习调度器覆盖 Q-drift 的 dispatch"
    )
    optimize_parser.add_argument(
        "--scheduler-model",
        default="outputs/scheduler_model.pt",
        help="学习调度器模型保存/加载路径（默认 outputs/scheduler_model.pt）",
    )

    # compare 命令
    compare_parser = subparsers.add_parser("compare", help="对比基线和优化结果")
    compare_parser.add_argument("--baseline", required=True, help="基线结果路径")
    compare_parser.add_argument("--optimized", required=True, help="优化结果路径")
    compare_parser.add_argument("--output", default="outputs/comparison.html", help="输出报告路径")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动 serving 模式")
    serve_parser.add_argument("--model", required=True, help="模型名称")
    serve_parser.add_argument("--config", default="configs/optimize_full.yaml", help="配置文件路径")
    serve_parser.add_argument(
        "--backend",
        choices=_backend_choices(),
        default="pytorch",
        help="模型后端（默认 pytorch）。可用后端见 shadowinfer --help 或 registry 文档",
    )
    serve_parser.add_argument(
        "--device", default=_default_device(), help="运行设备（默认 cuda 若可用否则 cpu）"
    )
    serve_parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于可复现的模型初始化（默认 42）"
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="服务绑定地址（默认 127.0.0.1）")
    serve_parser.add_argument("--port", type=int, default=8000, help="服务端口（默认 8000）")
    serve_parser.add_argument(
        "--serving-config",
        help="serving 运行时配置文件路径（YAML 格式）",
    )
    serve_parser.add_argument(
        "--num-steps", type=int, default=20, help="默认 denoising 步数（默认 20）"
    )
    serve_parser.add_argument(
        "--strategy",
        choices=["baseline", "optimized", "ab"],
        default="optimized",
        help="默认 serving 策略（默认 optimized）",
    )
    serve_parser.add_argument("--prompt", default="Hello world", help="单次测试提示")
    serve_parser.add_argument(
        "--one-shot",
        action="store_true",
        help="运行单次生成后退出，不启动常驻 HTTP 服务（用于冒烟测试）",
    )

    # dashboard 命令
    dashboard_parser = subparsers.add_parser("dashboard", help="启动交互式 Web Profiler")
    dashboard_parser.add_argument(
        "--port", type=int, default=8501, help="Streamlit 端口（默认 8501）"
    )

    # benchmark 命令
    benchmark_parser = subparsers.add_parser("benchmark", help="运行 benchmark 并记录到回归历史")
    benchmark_parser.add_argument("--model", required=True, help="模型名称")
    benchmark_parser.add_argument(
        "--config", default="configs/optimize_full.yaml", help="配置文件路径"
    )
    benchmark_parser.add_argument("--output", default="benchmarks/results", help="输出目录")
    benchmark_parser.add_argument("--prompt", default="Hello world", help="测试提示")
    benchmark_parser.add_argument("--num-steps", type=int, default=20, help="denoising 步数")
    benchmark_parser.add_argument(
        "--backend",
        choices=_backend_choices(),
        default="auto",
        help="模型后端（默认 auto）。可用后端见 shadowinfer --help 或 registry 文档",
    )
    benchmark_parser.add_argument(
        "--device", default=_default_device(), help="运行设备（默认 cuda 若可用否则 cpu）"
    )
    benchmark_parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于可复现的模型初始化（默认 42）"
    )
    benchmark_parser.add_argument(
        "--history",
        default="benchmarks/results/regression_history.jsonl",
        help="回归历史文件路径",
    )
    benchmark_parser.add_argument(
        "--scheduler", action="store_true", help="启用学习调度器覆盖 Q-drift 的 dispatch"
    )
    benchmark_parser.add_argument(
        "--scheduler-model",
        default="outputs/scheduler_model.pt",
        help="学习调度器模型保存/加载路径（默认 outputs/scheduler_model.pt）",
    )

    return parser


def _default_model_config(model_name: str) -> Dict[str, Any]:
    """返回 CLI 使用的默认模型配置（小型 CPU 可运行配置）。"""
    return {
        "name": model_name,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 32,
        "hidden_dim": 128,
        "intermediate_dim": 512,
        "vocab_size": 1000,
        "max_seq_len": 128,
        "batch_size": 1,
        "seq_len": 128,
    }


def _resolve_backend_name(name: str) -> str:
    """Resolve the ``auto`` alias to the default backend name."""
    return "pytorch" if name == "auto" else name


def _create_model_backend(args, backend_args: Optional[Dict[str, Any]] = None) -> ModelBackend:
    """根据 CLI 参数创建并加载一个 ModelBackend 实例。

    Args:
        args: argparse 解析后的参数对象。
        backend_args: 额外传递给 backend 构造函数的参数字典（例如 serving 配置中的 args）。

    Returns:
        已加载的 ModelBackend 实例。
    """
    from shadowinfer.core.backends.registry import get_backend_registry

    registry = get_backend_registry()
    name = _resolve_backend_name(args.backend)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    kwargs: Dict[str, Any] = dict(backend_args or {})
    kwargs.setdefault("model_config", _default_model_config(args.model))
    kwargs.setdefault("seed", args.seed)

    backend = registry.create(name, **kwargs)
    backend.load(args.model, device=args.device)
    return backend


def _setup_orchestrator_backend(orch, args) -> None:
    """根据 CLI 参数为 Orchestrator 配置模型后端并初始化 Agent。

    Args:
        orch: Orchestrator 实例。
        args: argparse 解析后的参数对象。
    """
    from shadowinfer.core.backends.registry import get_backend_registry

    registry = get_backend_registry()
    name = _resolve_backend_name(args.backend)

    if name == "mock":
        orch.initialize(model_config={"name": args.model})
        return

    if args.seed is not None:
        torch.manual_seed(args.seed)

    backend = registry.create(
        name,
        model_config=_default_model_config(args.model),
        seed=args.seed,
    )
    backend.load(args.model, device=args.device)
    orch.set_model_backend(backend)
    orch.initialize(model_config=backend.get_model_config())


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "profiler":
        run_profiler(args)
    elif args.command == "optimize":
        run_optimize(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_profiler(args):
    """运行性能分析。

    对应 ARCHITECTURE.md §4.2 Profiling Phase。

    Args:
        args: argparse 解析后的参数对象。
    """
    from shadowinfer.orchestrator import Orchestrator

    orch = Orchestrator(config_path=args.config)
    _setup_orchestrator_backend(orch, args)
    if args.scheduler:
        orch.enable_learned_scheduler(model_path=args.scheduler_model)

    with orch:
        result = orch.run_baseline(prompt=args.prompt, num_steps=args.num_steps)
        print(f"Profiling completed. Results saved to {args.output}")
        print(f"End-to-end latency: {result.get('latency', {}).get('e2e_ms', 'N/A')} ms")


def run_optimize(args):
    """运行优化推理。

    对应 ARCHITECTURE.md §4.2 Optimization Phase。

    Args:
        args: argparse 解析后的参数对象。
    """
    from shadowinfer.orchestrator import Orchestrator

    orch = Orchestrator(config_path=args.config)
    _setup_orchestrator_backend(orch, args)
    if args.scheduler:
        orch.enable_learned_scheduler(model_path=args.scheduler_model)

    with orch:
        result = orch.run_full_pipeline(prompt=args.prompt, num_steps=args.num_steps)
        print(f"Optimization completed. Results saved to {args.output}")
        print(f"Speedup: {result.get('speedup', 'N/A')}x")
        print(f"Accuracy drop: {result.get('accuracy_drop', 'N/A')}")


def run_compare(args):
    """对比基线和优化结果。

    对应 ARCHITECTURE.md §4.2 对比验证阶段。

    Args:
        args: argparse 解析后的参数对象。
    """
    import json

    from shadowinfer.profiler.reporter import HTMLReporter

    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.optimized, "r", encoding="utf-8") as f:
        optimized = json.load(f)

    reporter = HTMLReporter()
    reporter.generate(baseline, optimized, args.output)
    print(f"Comparison report saved to {args.output}")


def run_serve(args):
    """启动 serving 模式。

    Args:
        args: argparse 解析后的参数对象。
    """
    from shadowinfer.serving import ServingBackend, ServingConfig, serve_forever

    serving_config = ServingConfig()
    if args.serving_config:
        serving_config = ServingConfig.from_yaml(args.serving_config)
    serving_config.default_num_steps = args.num_steps
    serving_config.default_strategy = args.strategy

    # Allow serving.yaml to override the default backend unless the user
    # explicitly passed --backend on the CLI.
    if args.backend == "pytorch" and serving_config.backend_type is not None:
        args.backend = serving_config.backend_type

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model_backend = _create_model_backend(args, backend_args=serving_config.backend_args)

    serving_backend = ServingBackend(
        model_backend=model_backend,
        config=serving_config,
        config_path=args.config,
        device=args.device,
    )
    print(
        f"[serve] ShadowInfer serving mode. "
        f"Model={args.model}, backend={args.backend}, device={args.device}, "
        f"strategy={args.strategy}"
    )
    if args.one_shot:
        response = serving_backend.generate(
            prompt=args.prompt,
            num_steps=args.num_steps,
            strategy=args.strategy,
        )
        print("[serve] One-shot generation completed.")
        print(json.dumps(response, ensure_ascii=False, default=str))
        return
    print(f"[serve] Starting HTTP server at http://{args.host}:{args.port}")
    serve_forever(serving_backend, host=args.host, port=args.port)


def run_dashboard(args):
    """启动 Streamlit Web Profiler。

    Args:
        args: argparse 解析后的参数对象。
    """
    import subprocess

    app_path = Path(__file__).parent / "web_profiler" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
    print(f"Launching ShadowInfer Web Profiler at http://localhost:{args.port}")
    subprocess.run(cmd, check=False)


def run_benchmark(args):
    """运行 benchmark 并记录到回归历史。

    Args:
        args: argparse 解析后的参数对象。
    """
    import json

    from shadowinfer.benchmarking.regression import (
        RegressionTracker,
        record_from_pipeline_summary,
    )
    from shadowinfer.orchestrator import Orchestrator

    orch = Orchestrator(config_path=args.config)
    _setup_orchestrator_backend(orch, args)
    if args.scheduler:
        orch.enable_learned_scheduler(model_path=args.scheduler_model)

    with orch:
        summary = orch.run_full_pipeline(prompt=args.prompt, num_steps=args.num_steps)

    result = record_from_pipeline_summary(
        summary,
        history_path=args.history,
        backend=args.backend,
        metadata={"device": args.device, "seed": args.seed},
    )
    tracker = RegressionTracker(history_path=args.history)
    report = tracker.detect_regression(result)

    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "regression_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"Benchmark completed. Report saved to {report_path}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    print(f"Speedup: {result.speedup:.2f}x")
    print(f"Accuracy drop: {result.accuracy_drop:.4f}")
    if report["has_regression"]:
        print("WARNING: Performance regression detected.")
    else:
        print("No regression detected.")


if __name__ == "__main__":
    main()

"""AR004 baseline microbenchmark: FFN on_step hot path (fast, CPU)."""
import cProfile
import io
import pstats
import sys
import time

import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shadowinfer.core.structs import StepConfig
from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent

torch.manual_seed(42)
HID, INT, LAYERS = 256, 1024, 4
agent = FFNOptimizerAgent(
    {
        "hidden_dim": HID,
        "intermediate_dim": INT,
        "channel_pruning_enabled": False,
        "channel_pruning_ratio": 0.0,
        "ffn_pack_group_size": 64,
        "use_packed_weights": True,
    }
)
agent.on_init({"hidden_dim": HID, "intermediate_dim": INT, "num_layers": LAYERS})

weight_up = torch.randn(INT, HID) * 0.02
weight_down = torch.randn(HID, INT) * 0.02
x = torch.randn(1, 32, HID)
x_prev = x + 0.01 * torch.randn_like(x)
out_prev = torch.zeros(1, 32, HID)


def run_steps(ag, n=3):
    for step in range(n):
        for layer in range(LAYERS):
            ag.on_step(
                StepConfig(step_id=step, total_steps=n, noise_level=0.5, shadowkv_mode="balanced", ffn_mode="mixed"),
                {
                    "ffn_input_current": x,
                    "ffn_input_previous": x_prev,
                    "ffn_output_previous": out_prev,
                    "weights": {"up": weight_up, "down": weight_down},
                    "layer_id": layer,
                    "qdrift_signal": {"sensitivity_score": 0.5, "ffn_mode": "mixed"},
                },
            )


# warmup (first-call packing/dequant)
run_steps(agent, 1)

prof = cProfile.Profile()
prof.enable()
run_steps(agent, 3)
prof.disable()
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(12)
text = buf.getvalue()
print("\n".join(text.splitlines()[:26]))

# wall-time: 3 repeats, per-step median (warmup per agent before timing)
times = []
for _ in range(3):
    agent2 = FFNOptimizerAgent(
        {
            "hidden_dim": HID,
            "intermediate_dim": INT,
            "channel_pruning_enabled": False,
            "channel_pruning_ratio": 0.0,
            "use_packed_weights": True,
        }
    )
    agent2.on_init({"hidden_dim": HID, "intermediate_dim": INT, "num_layers": LAYERS})
    run_steps(agent2, 1)  # warmup
    t0 = time.perf_counter()
    run_steps(agent2, 3)
    times.append((time.perf_counter() - t0) * 1000.0 / 12)  # ms per on_step
print("per_on_step_ms:", [round(t, 2) for t in times], "median:", round(sorted(times)[1], 2))

# fingerprint overhead probe
pw = agent.weight_up_quantized
if pw is not None:
    t0 = time.perf_counter()
    for _ in range(100):
        pw._precision_plan_fingerprint()
    print("fingerprint_us:", round((time.perf_counter() - t0) * 1e6 / 100, 1))
    cached = pw.dequantize_for_matmul()
    t0 = time.perf_counter()
    for _ in range(100):
        pw.dequantize_for_matmul()
    print("dequant_cached_us:", round((time.perf_counter() - t0) * 1e6 / 100, 1))


# ---------------------------------------------------------------------------
# Defect repro (1): _mixed_compute + channel_mask + packed -> shape mismatch
# ---------------------------------------------------------------------------
def repro_mixed_mask_defect():
    ag = FFNOptimizerAgent(
        {
            "hidden_dim": HID,
            "intermediate_dim": INT,
            "channel_pruning_enabled": True,
            "channel_pruning_ratio": 0.3,
            "ffn_pack_group_size": 64,
            "use_packed_weights": True,
        }
    )
    ag.on_init({"hidden_dim": HID, "intermediate_dim": INT, "num_layers": LAYERS})
    try:
        out = ag.on_step(
            StepConfig(step_id=0, total_steps=1, noise_level=0.5, shadowkv_mode="balanced", ffn_mode="mixed"),
            {
                "ffn_input_current": x,
                "ffn_input_previous": x,
                "ffn_output_previous": out_prev,
                "weights": {"up": weight_up, "down": weight_down},
                "layer_id": 0,
                "qdrift_signal": {"sensitivity_score": 0.5, "ffn_mode": "mixed"},
            },
        )
        print("mask_defect: no error, path=", out.get("compute_path"))
    except RuntimeError as exc:
        print("mask_defect: RuntimeError:", str(exc)[:120])


# ---------------------------------------------------------------------------
# Defect repro (2): FFN-P1 shared layer state (layer 1 silently reuses layer 0)
# ---------------------------------------------------------------------------
def repro_layer_state_defect():
    w1_up = torch.randn(INT, HID) * 0.02
    w1_down = torch.randn(HID, INT) * 0.02
    w2_up = torch.randn(INT, HID) * 0.02
    w2_down = torch.randn(HID, INT) * 0.02
    ag = FFNOptimizerAgent(
        {
            "hidden_dim": HID,
            "intermediate_dim": INT,
            "channel_pruning_enabled": False,
            "channel_pruning_ratio": 0.0,
            "use_packed_weights": True,
        }
    )
    ag.on_init({"hidden_dim": HID, "intermediate_dim": INT, "num_layers": LAYERS})
    cfg = StepConfig(step_id=0, total_steps=1, noise_level=0.5, shadowkv_mode="balanced", ffn_mode="full")
    ag.on_step(
        cfg,
        {
            "ffn_input_current": x,
            "weights": {"up": w1_up, "down": w1_down},
            "layer_id": 0,
            "qdrift_signal": {"sensitivity_score": 0.9, "ffn_mode": "full"},
        },
    )
    out_layer1 = ag.on_step(
        cfg,
        {
            "ffn_input_current": x,
            "weights": {"up": w2_up, "down": w2_down},
            "layer_id": 1,
            "qdrift_signal": {"sensitivity_score": 0.9, "ffn_mode": "full"},
        },
    )["output"]
    direct = torch.nn.functional.linear(
        torch.nn.functional.gelu(torch.nn.functional.linear(x, w2_up)), w2_down
    )
    shared = torch.nn.functional.linear(
        torch.nn.functional.gelu(torch.nn.functional.linear(x, w1_up)), w1_down
    )
    print(
        "layer_state_defect: |out-direct(layer2)|=",
        round(float((out_layer1 - direct).abs().max()), 4),
        "|out-shared(layer1)|=",
        round(float((out_layer1 - shared).abs().max()), 4),
    )


# ---------------------------------------------------------------------------
# Same-process A/B (rule 1.1): structural metrics at 256 and 1024 channels
# ---------------------------------------------------------------------------
from shadowinfer.ffn_optimizer.ffn_optimizer_agent import (  # noqa: E402
    FFNOptimizerAgent as _ABAgent,
)
from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight  # noqa: E402


def _median(values):
    return sorted(values)[len(values) // 2]


def structural_ab():
    def legacy_fingerprint(pw):
        return tuple((ch, d["precision"]) for ch, d in sorted(pw._data.items()))

    agent_ab = _ABAgent({"use_packed_weights": False, "num_layers": 4})
    for channels, dim in ((256, 256), (1024, 256)):
        torch.manual_seed(0)
        w_up = torch.randn(channels, dim)
        w_down = torch.randn(dim, channels)
        imp = torch.linspace(0.0, 1.0, channels)
        pmap = {i: ("fp16" if i % 4 == 0 else ("int8" if i % 4 == 1 else "int4")) for i in range(channels)}
        pw = PackedFFNWeight.pack(w_up, pmap, group_size=64)
        pw.dequantize_for_matmul()
        pw._unpack_reference()

        vec_fp, legacy_fp, vec_up, ref_up, vec_q, ref_q = [], [], [], [], [], []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(50):
                pw._precision_plan_fingerprint()
            vec_fp.append((time.perf_counter() - t0) / 50)
            t0 = time.perf_counter()
            for _ in range(50):
                legacy_fingerprint(pw)
            legacy_fp.append((time.perf_counter() - t0) / 50)
            t0 = time.perf_counter()
            for _ in range(3):
                pw.unpack()
            vec_up.append((time.perf_counter() - t0) / 3)
            t0 = time.perf_counter()
            for _ in range(3):
                pw._unpack_reference()
            ref_up.append((time.perf_counter() - t0) / 3)
            t0 = time.perf_counter()
            for _ in range(3):
                agent_ab.quantize_mixed_precision(w_up, w_down, imp)
            vec_q.append((time.perf_counter() - t0) / 3)
            t0 = time.perf_counter()
            for _ in range(3):
                agent_ab._quantize_mixed_precision_reference(w_up, w_down, imp)
            ref_q.append((time.perf_counter() - t0) / 3)

        print(
            f"AB ch={channels}: fingerprint_us vec={_median(vec_fp)*1e6:.1f} "
            f"legacy={_median(legacy_fp)*1e6:.1f} ratio={_median(vec_fp)/_median(legacy_fp):.3f}"
        )
        print(
            f"AB ch={channels}: unpack_ms vec={_median(vec_up)*1e3:.2f} "
            f"ref={_median(ref_up)*1e3:.2f} ratio={_median(vec_up)/_median(ref_up):.3f}"
        )
        print(
            f"AB ch={channels}: quantize_ms vec={_median(vec_q)*1e3:.2f} "
            f"ref={_median(ref_q)*1e3:.2f} ratio={_median(vec_q)/_median(ref_q):.3f}"
        )


structural_ab()
repro_mixed_mask_defect()
repro_layer_state_defect()

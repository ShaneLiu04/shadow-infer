"""AR002-T002 M6 性能指标真实化测试（TDD：只含测试，不含实现）。

依据：
- specs/changes/AR002-hotpath-metrics-drift/srs.md FR3.1（M6 需求与验收标准）
- specs/changes/AR002-hotpath-metrics-drift/design.md §3 功能点 1/2/3/8、
  §4.1 M6 指标口径、§6.1/6.3/6.4 测试设计
- specs/changes/AR002-hotpath-metrics-drift/baseline-profile.md §5（零值指标实录）

M6 实现落地前，_run_profiler 的 perf_data 仍为硬编码零值
（latency_ms=0.0 / tokens_per_sec=0.0），_compute_total_latency 仍有
0.0 -> 10.0 造假兜底，global_state 缺少 latency_ms 数据源，
本文件针对真实计时/派生指标的断言预期失败（红灯），用于驱动实现任务。

计时类断言一律使用容差比较（pytest.approx / 上下界），不做精确相等，
避免 CI flaky。
"""

import time

import pytest

from shadowinfer.core.scheduler import LearnedScheduler, StepExperience
from shadowinfer.orchestrator import InferenceResult, Orchestrator

# e2e 与 sum(per_step) 的一致性容差（design §6.3：e2e 与分步累加一致）
E2E_VS_PER_STEP_TOLERANCE = 0.05
# tokens_per_sec 派生关系容差（design §4.1：tokens_per_sec = tokens_per_step / latency）
TOKENS_DERIVATION_TOLERANCE = 0.05


def _make_experience(latency_ms: float, memory_mb: float = 100.0) -> StepExperience:
    """构造仅 latency/memory 不同的 StepExperience（奖励纯单元测试用）。"""
    return StepExperience(
        step_id=0,
        total_steps=10,
        noise_level=0.1,
        sensitivity_score=0.3,
        drift_score=0.2,
        prev_latency_ms=10.0,
        prev_memory_mb=100.0,
        prev_accuracy_drop=0.0,
        shadowkv_mode="balanced",
        ffn_mode="mixed",
        latency_ms=latency_ms,
        memory_mb=memory_mb,
        accuracy_drop=0.0,
    )


def _make_step_result(e2e_ms: float) -> dict:
    """构造只含 latency.e2e_ms 的 step_results 条目（_compute_total_latency 用）。"""
    return {"profiler": {"performance_metrics": {"latency": {"e2e_ms": e2e_ms}}}}


def _profiler_step_latency_log_values(orch: Orchestrator) -> list:
    """从 profiler 结构化日志缓存提取 step_latency_ms 指标值。"""
    values = []
    for record in orch.profiler.logger.get_logs():
        if record.get("event") == "metric" and record.get("metric_name") == "step_latency_ms":
            values.append(record.get("value"))
    return values


def _has_cpu_device_annotation(orch: Orchestrator) -> bool:
    """检查 profiler 日志是否附带 {"device": "cpu"} 口径标注（design §4.1）。"""
    for record in orch.profiler.logger.get_logs():
        tags = record.get("tags") or {}
        data = record.get("data") or {}
        if tags.get("device") == "cpu" or data.get("device") == "cpu":
            return True
    return False


@pytest.fixture(scope="module")
def optimized_run():
    """共享的 2-step run_optimized 结果。

    只跑一次供本模块多个断言复用，控制测试总耗时；
    wall_ms 为包住整个 run_optimized 的外部墙钟（防 flaky 的一致性下界用）。
    """
    orch = Orchestrator(config="configs/optimize_full.yaml")
    wall_start = time.perf_counter()
    result = orch.run_optimized(prompt="m6-metrics-test", num_steps=2)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    yield orch, result, wall_ms
    orch.shutdown()


class TestM6StepRealTiming:
    """功能点 1（M6 step 真实计时）：per-step 实测延迟与派生吞吐（验收点 A）。"""

    def test_per_step_latency_positive_in_run_result(self, optimized_run):
        """run 结果 latency.per_step_ms 每个 step 均应为正实测值（替换 0.0）。"""
        _, result, _ = optimized_run
        per_step = result["latency"]["per_step_ms"]
        assert len(per_step) == 2
        for sid, lat in per_step.items():
            assert lat > 0.0, f"step {sid} per_step_ms 应为正实测值，实际 {lat}"

    def test_e2e_consistent_with_per_step_sum(self, optimized_run):
        """e2e_ms 应与 sum(per_step_ms) 一致（容差 5%）。

        baseline-profile §5：现状 per_step_ms 全 0.0 而 e2e_ms 为真实值，
        两者互相矛盾——M6 后两者必须出自同一实测口径。
        """
        _, result, _ = optimized_run
        e2e = result["latency"]["e2e_ms"]
        per_step_sum = sum(result["latency"]["per_step_ms"].values())
        assert e2e > 0.0
        assert per_step_sum > 0.0
        assert e2e == pytest.approx(per_step_sum, rel=E2E_VS_PER_STEP_TOLERANCE)

    def test_profiler_step_latency_positive(self, optimized_run):
        """每个 step 的 profiler performance_metrics.latency.e2e_ms 应为正实测值。"""
        orch, _, _ = optimized_run
        assert len(orch.step_results) == 2
        for sr in orch.step_results:
            perf = sr["profiler"]["performance_metrics"]
            lat = perf["latency"]["e2e_ms"]
            assert lat > 0.0, f"profiler e2e_ms 应为正实测值，实际 {lat}"
            for value in perf["latency"]["per_step_ms"].values():
                assert value > 0.0

    def test_profiler_step_latency_metric_log_positive(self, optimized_run):
        """profiler 日志的 step_latency_ms 指标应为正实测值（替换 0.0）。"""
        orch, _, _ = optimized_run
        values = _profiler_step_latency_log_values(orch)
        assert len(values) == 2
        for value in values:
            assert value > 0.0

    def test_tokens_per_step_equals_batch_times_seq_len(self, optimized_run):
        """tokens_per_step 应为 batch*seq_len（替换硬编码 1.0）。"""
        orch, _, _ = optimized_run
        mc = orch._model_config
        expected = mc.get("batch_size", 1) * mc.get("seq_len", 128)
        for sr in orch.step_results:
            throughput = sr["profiler"]["performance_metrics"]["throughput"]
            assert throughput["tokens_per_step"] == expected

    def test_tokens_per_sec_positive_and_derived_from_measured_latency(self, optimized_run):
        """tokens_per_sec 应为正，且由实测 latency 派生（容差 5%）。"""
        orch, _, _ = optimized_run
        for sr in orch.step_results:
            perf = sr["profiler"]["performance_metrics"]
            lat = perf["latency"]["e2e_ms"]
            assert lat > 0.0
            tokens_per_step = perf["throughput"]["tokens_per_step"]
            tokens_per_sec = perf["throughput"]["tokens_per_sec"]
            assert tokens_per_sec > 0.0
            expected = tokens_per_step / (lat / 1000.0)
            assert tokens_per_sec == pytest.approx(expected, rel=TOKENS_DERIVATION_TOLERANCE)

    def test_gpu_utilization_zero_with_cpu_device_annotation(self, optimized_run):
        """CPU fallback 下 gpu_utilization 保持 0.0，且日志附带 device=cpu 口径标注。"""
        orch, _, _ = optimized_run
        for sr in orch.step_results:
            gpu = sr["profiler"]["performance_metrics"]["gpu_utilization"]
            assert gpu["compute_util"] == 0.0
        assert _has_cpu_device_annotation(
            orch
        ), "profiler 日志应附带 {'device': 'cpu'} 口径标注（design §4.1）"

    def test_e2e_consistent_with_external_wall_clock(self, optimized_run):
        """e2e_ms 应落在整个 run 外部墙钟的宽松上下界内。

        srs FR3.1 要求与外部墙钟误差 < 2%（由 T008 实测落档验收）；
        本测试用宽松界防止 CI flaky：step 执行应主导 run 的墙钟。
        """
        _, result, wall_ms = optimized_run
        e2e = result["latency"]["e2e_ms"]
        assert e2e <= wall_ms
        assert e2e >= 0.5 * wall_ms


class TestM6GlobalStateLatency:
    """功能点 1/3（M6 step 真实计时 + 调度真实奖励）：global_state 数据源（验收点 B）。"""

    def test_global_state_latency_ms_positive_after_run(self, optimized_run):
        """run 结束后 global_state["latency_ms"] 应为本 step 实测值（> 0）。"""
        orch, _, _ = optimized_run
        assert orch.global_state.get("latency_ms", 0.0) > 0.0

    def test_policy_context_latency_ratio_positive(self, optimized_run):
        """_build_policy_context 的 latency_ratio 应基于实测 latency（> 0）。

        修复前 global_state 无 latency_ms 数据源，latency_ratio 恒为 0。
        """
        orch, _, _ = optimized_run
        context = orch._build_policy_context([])
        assert context.get("latency_ratio", 0.0) > 0.0


class TestM6ComputeTotalLatency:
    """功能点 1（M6）：_compute_total_latency 移除 0.0 -> 10.0 造假兜底（验收点 C）。"""

    def test_all_zero_steps_return_zero_without_fake_fallback(self):
        """全零 step_results 应返回 0.0，而非逐 step 兜底 10.0（旧实现 3 步得 30.0）。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        result = InferenceResult()
        result.step_results = [_make_step_result(0.0) for _ in range(3)]
        assert orch._compute_total_latency(result) == 0.0

    def test_zero_steps_excluded_but_others_accumulate(self):
        """零值 step 不计入总和，非零 step 仍正常累加。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        result = InferenceResult()
        result.step_results = [
            _make_step_result(0.0),
            _make_step_result(15.0),
            _make_step_result(0.0),
            _make_step_result(20.0),
        ]
        assert orch._compute_total_latency(result) == 35.0

    def test_zero_latency_logs_warning(self):
        """零值 step 计入总延迟计算时应记 warning 级结构化日志。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        before = len(orch.logger.get_logs(level="warning"))
        result = InferenceResult()
        result.step_results = [_make_step_result(0.0), _make_step_result(5.0)]
        orch._compute_total_latency(result)
        after = len(orch.logger.get_logs(level="warning"))
        assert after > before


class TestStepExperienceRewardMonotonic:
    """功能点 3（M6 调度真实奖励）：StepExperience.reward 单调性纯单元测试（验收点 D）。"""

    def test_reward_non_increasing_as_latency_grows(self):
        """latency 增大时 reward 单调不增。"""
        latencies = [0.0, 1.0, 5.0, 10.0, 100.0, 1000.0]
        rewards = [_make_experience(lat).reward() for lat in latencies]
        for r_prev, r_next in zip(rewards, rewards[1:]):
            assert r_next <= r_prev

    def test_reward_strictly_decreasing_for_strictly_increasing_latency(self):
        """latency 严格增大（memory 固定）时 reward 严格减小。"""
        latencies = [1.0, 2.0, 4.0, 8.0, 16.0]
        rewards = [_make_experience(lat, memory_mb=100.0).reward() for lat in latencies]
        for r_prev, r_next in zip(rewards, rewards[1:]):
            assert r_next < r_prev

    def test_reward_negative_for_nonzero_latency(self):
        """非零延迟（无论 memory 取值）下 reward 应为负。"""
        for lat in (0.001, 1.0, 500.0):
            for mem in (0.0, 100.0, 16383.5625):
                assert _make_experience(lat, memory_mb=mem).reward() < 0.0

    def test_reward_zero_only_when_latency_and_memory_zero(self):
        """仅当 latency 与 memory 同时为零时 reward 为 0。"""
        assert _make_experience(0.0, memory_mb=0.0).reward() == 0.0


class TestM6SchedulerConsumesRealLatency:
    """功能点 3（M6 调度真实奖励）：调度器经验消费真实延迟（验收点 D）。"""

    def test_scheduler_experiences_consume_real_latency(self):
        """启用学习调度器运行后，经验应记录正的实测 latency 且 reward < 0。

        修复前 profiler performance_metrics.latency.e2e_ms 恒为 0.0，
        调度器吃到零延迟奖励（baseline-profile §5）。
        """
        orch = Orchestrator(config="configs/optimize_full.yaml")
        scheduler = LearnedScheduler()
        orch.enable_learned_scheduler(scheduler=scheduler)
        try:
            orch.run_optimized(prompt="m6-scheduler-test", num_steps=2)
            assert len(scheduler.experiences) == 2
            for exp in scheduler.experiences:
                assert exp.latency_ms > 0.0
                assert exp.reward() < 0.0
            ordered = sorted(scheduler.experiences, key=lambda e: e.latency_ms)
            rewards = [e.reward() for e in ordered]
            for r_small_lat, r_large_lat in zip(rewards, rewards[1:]):
                assert r_large_lat <= r_small_lat
        finally:
            orch.shutdown()

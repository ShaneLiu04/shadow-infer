"""Tests for the production serving scaffold."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict

import pytest

from shadowinfer.core.model_backend import MockModelBackend
from shadowinfer.serving import ServingBackend, ServingConfig, create_server


def _small_model_config() -> Dict[str, Any]:
    return {
        "name": "test-model",
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 8,
        "hidden_dim": 16,
        "intermediate_dim": 32,
        "batch_size": 1,
        "seq_len": 8,
        "vocab_size": 100,
        "max_seq_len": 16,
    }


@pytest.fixture
def serving_backend() -> ServingBackend:
    backend = MockModelBackend(model_config=_small_model_config())
    cfg = ServingConfig(
        max_concurrent_requests=2,
        rate_limit_rps=1000.0,
        rate_limit_burst=1000,
    )
    return ServingBackend(
        model_backend=backend,
        config=cfg,
        config_path=None,
        device="cpu",
    )


@pytest.fixture
def server_url(serving_backend: ServingBackend) -> Any:
    server = create_server(serving_backend, host="127.0.0.1", port=0)
    server.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.stop()


def _get(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _post(url: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_endpoint(server_url: str) -> None:
    body = _get(f"{server_url}/health")
    assert body["status"] == "ok"
    assert body["backend"] == "MockModelBackend"
    assert "config" in body


def test_root_endpoint(server_url: str) -> None:
    body = _get(server_url)
    assert body["service"] == "ShadowInfer"
    assert "/generate" in body["endpoints"]
    assert "/metrics" in body["endpoints"]


def test_metrics_endpoint(server_url: str) -> None:
    # Trigger a request first so metrics are non-empty.
    _post(f"{server_url}/generate", {"prompt": "metrics", "num_steps": 2})
    body = _get_text(f"{server_url}/metrics")
    assert "shadowinfer_serving_requests_total" in body


def test_generate_optimized(server_url: str) -> None:
    status, body = _post(
        f"{server_url}/generate",
        {"prompt": "hello", "num_steps": 2, "strategy": "optimized"},
    )
    assert status == 200
    assert body["prompt"] == "hello"
    assert body["strategy"] == "optimized"
    assert body["num_steps"] == 2
    assert "serving_latency_ms" in body
    assert "result" in body


def test_generate_baseline(server_url: str) -> None:
    status, body = _post(
        f"{server_url}/generate",
        {"prompt": "world", "num_steps": 2, "strategy": "baseline"},
    )
    assert status == 200
    assert body["strategy"] == "baseline"
    assert "result" in body


def test_generate_ab_picks_a_strategy(server_url: str) -> None:
    status, body = _post(
        f"{server_url}/generate",
        {"prompt": "ab test", "num_steps": 2, "strategy": "ab"},
    )
    assert status == 200
    assert body["strategy"] in {"baseline", "optimized"}


def test_generate_invalid_prompt(server_url: str) -> None:
    status, body = _post(
        f"{server_url}/generate",
        {"prompt": "", "num_steps": 2, "strategy": "optimized"},
    )
    assert status == 400
    assert "error" in body


def test_not_found(server_url: str) -> None:
    status, body = _post(f"{server_url}/unknown", {})
    assert status == 404
    assert "error" in body


def test_backend_generate_directly() -> None:
    backend = MockModelBackend(model_config=_small_model_config())
    serving = ServingBackend(
        model_backend=backend,
        config=ServingConfig(max_concurrent_requests=2, rate_limit_rps=1000.0),
        config_path=None,
        device="cpu",
    )
    result = serving.generate(prompt="direct", num_steps=2, strategy="optimized")
    assert result["prompt"] == "direct"
    assert result["strategy"] == "optimized"
    assert "result" in result
    assert "serving_latency_ms" in result


def test_rate_limiter_blocks_excess() -> None:
    backend = MockModelBackend(model_config=_small_model_config())
    serving = ServingBackend(
        model_backend=backend,
        config=ServingConfig(
            max_concurrent_requests=2,
            rate_limit_rps=1.0,
            rate_limit_burst=1,
        ),
        config_path=None,
        device="cpu",
    )
    # First call should pass.
    serving.generate(prompt="first", num_steps=1, strategy="optimized")
    # Second call immediately after should be rate limited.
    with pytest.raises(RuntimeError, match="rate limit"):
        serving.generate(prompt="second", num_steps=1, strategy="optimized")


def test_concurrency_limiter_blocks_excess() -> None:
    backend = MockModelBackend(model_config=_small_model_config())
    serving = ServingBackend(
        model_backend=backend,
        config=ServingConfig(
            max_concurrent_requests=1,
            rate_limit_rps=1000.0,
            request_timeout=0.05,
        ),
        config_path=None,
        device="cpu",
    )

    acquired = []

    def block() -> None:
        serving._concurrency_limiter.acquire()
        acquired.append(1)
        time.sleep(0.2)
        serving._concurrency_limiter.release()

    t = threading.Thread(target=block)
    t.start()
    time.sleep(0.05)  # Ensure the first thread holds the semaphore.
    try:
        with pytest.raises(RuntimeError, match="concurrency"):
            serving.generate(prompt="blocked", num_steps=1, strategy="optimized")
    finally:
        t.join()


def test_ab_weights_selection() -> None:
    backend = MockModelBackend(model_config=_small_model_config())
    serving = ServingBackend(
        model_backend=backend,
        config=ServingConfig(
            ab_weights={"baseline": 1.0, "optimized": 0.0},
            rate_limit_rps=1000.0,
        ),
        config_path=None,
        device="cpu",
    )
    for _ in range(5):
        assert serving.pick_ab_strategy() == "baseline"


def test_hot_config_reload(tmp_path) -> None:
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        "max_concurrent_requests: 3\nrate_limit_rps: 5.0\nrate_limit_burst: 10\n",
        encoding="utf-8",
    )
    backend = MockModelBackend(model_config=_small_model_config())
    serving = ServingBackend(
        model_backend=backend,
        config=ServingConfig(hot_config_path=str(config_path), hot_config_interval=0.1),
        config_path=None,
        device="cpu",
    )
    try:
        # Wait for the background reload to pick up the file.
        time.sleep(0.3)
        assert serving.config.max_concurrent_requests == 3
        assert serving.config.rate_limit_rps == 5.0
    finally:
        serving.stop_hot_reload()

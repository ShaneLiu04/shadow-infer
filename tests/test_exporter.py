"""
Tests for Prometheus exporter module.
"""

import time

import pytest

from exporter import PrometheusExporter
from shadowinfer.observability import MetricsRegistry


class TestPrometheusExporter:
    def test_exporter_creation(self):
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9999)

        assert exporter.registry is registry
        assert exporter.port == 9999
        assert exporter.server is None

    def test_exporter_start_stop(self):
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9998)

        exporter.start()
        assert exporter.server is not None
        assert exporter.thread is not None
        assert exporter.thread.is_alive()

        time.sleep(0.5)  # Wait for server to start

        exporter.stop()
        # Thread should be daemon, so it will exit when main thread exits

    def test_metrics_endpoint(self):
        import urllib.request

        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test counter")
        counter.inc(5)

        exporter = PrometheusExporter(registry, port=9997)
        exporter.start()
        time.sleep(0.5)

        try:
            response = urllib.request.urlopen("http://127.0.0.1:9997/metrics", timeout=2)
            content = response.read().decode()

            assert "test_counter" in content
            assert "5.0" in content
        finally:
            exporter.stop()

    def test_health_endpoint(self):
        import urllib.request

        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9996)
        exporter.start()
        time.sleep(0.5)

        try:
            response = urllib.request.urlopen("http://127.0.0.1:9996/health", timeout=2)
            content = response.read().decode()

            assert "healthy" in content
        finally:
            exporter.stop()

    def test_404_endpoint(self):
        import urllib.error
        import urllib.request

        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9995)
        exporter.start()
        time.sleep(0.5)

        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://127.0.0.1:9995/unknown", timeout=2)
            assert exc_info.value.code == 404
        finally:
            exporter.stop()

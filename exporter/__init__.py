"""
Prometheus Metrics Exporter

版本：v3.0

HTTP server exposing Prometheus-compatible /metrics endpoint.

Usage:
    from shadowinfer.exporter import start_exporter
    start_exporter(port=9090)

Or from CLI:
    python -m exporter --port 9090
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from shadowinfer.observability import _PROMETHEUS_AVAILABLE, MetricsRegistry

logger = logging.getLogger(__name__)


def _format_metrics(registry: Optional[MetricsRegistry]) -> bytes:
    """生成 /metrics 响应内容。

    优先使用 ``prometheus_client`` 官方 exposition 格式；不可用时降级为
    自定义格式。
    """
    if registry is None:
        return b"# No metrics available\n"
    text = registry.expose()
    return text.encode("utf-8")


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics endpoint."""

    registry: Optional[MetricsRegistry] = None

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(_format_metrics(self.registry))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.info(format % args)


class PrometheusExporter:
    """
    Prometheus metrics exporter.

    Serves metrics on HTTP endpoint for Prometheus scraping.
    """

    def __init__(self, registry: MetricsRegistry, port: int = 9090):
        self.registry = registry
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start exporter in background thread."""
        MetricsHandler.registry = self.registry
        self.server = HTTPServer(("0.0.0.0", self.port), MetricsHandler)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Prometheus exporter started on port {self.port}")

    def stop(self) -> None:
        """Stop exporter."""
        if self.server is not None:
            self.server.shutdown()
            logger.info("Prometheus exporter stopped")


def start_exporter(registry: MetricsRegistry, port: int = 9090) -> PrometheusExporter:
    """Convenience function to start exporter."""
    exporter = PrometheusExporter(registry, port)
    exporter.start()
    return exporter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    registry = MetricsRegistry()
    exporter = start_exporter(registry, args.port)

    print(f"Exporter running on http://0.0.0.0:{args.port}/metrics")
    print("Press Ctrl+C to stop")

    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        exporter.stop()

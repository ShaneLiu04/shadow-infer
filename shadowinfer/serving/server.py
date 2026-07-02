"""HTTP server for ShadowInfer production serving.

对应文档：ROADMAP.md §3.6 Production & Serving
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from shadowinfer.serving.backend import ServingBackend


class _QuietHandler(BaseHTTPRequestHandler):
    """Base handler that suppresses default request logging."""

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def _make_request_handler(backend: ServingBackend) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to *backend*."""

    class Handler(_QuietHandler):
        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "backend": type(backend.model_backend).__name__,
                        "config": {
                            "max_concurrent_requests": backend.config.max_concurrent_requests,
                            "rate_limit_rps": backend.config.rate_limit_rps,
                            "ab_weights": backend.config.ab_weights,
                        },
                    },
                )
            elif parsed.path == "/":
                self._send_json(
                    200,
                    {
                        "service": "ShadowInfer",
                        "version": "3.1",
                        "endpoints": ["/health", "/generate", "/metrics"],
                    },
                )
            elif parsed.path == "/metrics":
                self._send_text(200, backend.metrics.expose(), "text/plain; charset=utf-8")
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/generate":
                self._send_json(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json(400, {"error": "invalid json", "detail": str(exc)})
                return

            prompt = payload.get("prompt", "")
            num_steps = payload.get("num_steps")
            strategy = payload.get("strategy")

            if not isinstance(prompt, str) or not prompt:
                self._send_json(400, {"error": "prompt must be a non-empty string"})
                return

            try:
                response = backend.generate(
                    prompt=prompt,
                    num_steps=int(num_steps) if num_steps is not None else None,
                    strategy=str(strategy) if strategy is not None else None,
                )
                self._send_json(200, response)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "rate limit" in message:
                    self._send_json(429, {"error": "rate limit exceeded"})
                elif "concurrency" in message:
                    self._send_json(503, {"error": "service overloaded"})
                else:
                    self._send_json(500, {"error": "generation failed", "detail": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json(500, {"error": "generation failed", "detail": str(exc)})

        def _send_text(self, status: int, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class ServingServer:
    """Production serving HTTP server with graceful shutdown."""

    def __init__(
        self,
        backend: ServingBackend,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.backend = backend
        self.host = host
        self.port = port
        self._httpd = ThreadingHTTPServer((host, port), _make_request_handler(backend))
        self._thread: Optional[threading.Thread] = None

    @property
    def server_address(self) -> tuple[str, int]:
        return self._httpd.server_address

    def start(self) -> None:
        """Start the server in a background thread."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server gracefully."""
        self.backend.stop_hot_reload()
        self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._httpd.server_close()

    def serve_forever(self) -> None:
        """Run the server in the current thread until interrupted."""
        print(f"ShadowInfer serving at http://{self.host}:{self.port}")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down serving server...")
        finally:
            self.stop()


def create_server(
    backend: ServingBackend,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ServingServer:
    """Create a serving server bound to *backend*."""
    return ServingServer(backend, host=host, port=port)


def serve_forever(
    backend: ServingBackend,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start a blocking serving server."""
    server = create_server(backend, host=host, port=port)
    server.serve_forever()

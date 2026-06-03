from __future__ import annotations

from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
import pickle
import time
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np


def sanitize_pickle_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: sanitize_pickle_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(sanitize_pickle_value(item) for item in value)
    return value


class RemoteHttpRpcClient:
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        retries: int = 3,
        retry_sleep: float = 1.0,
        keep_alive: bool = True,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
            raise ValueError(f"remote URL must be http://host:port, got {url!r}")
        self.url = url.rstrip("/")
        self.host = parsed.hostname
        self.port = parsed.port
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_sleep = float(retry_sleep)
        self.keep_alive = bool(keep_alive)
        self._conn: HTTPConnection | None = None

    def call(self, method: str, **kwargs) -> Any:
        payload = pickle.dumps({"method": method, "kwargs": sanitize_pickle_value(kwargs)}, protocol=pickle.HIGHEST_PROTOCOL)
        last_error: Exception | None = None
        for attempt in range(max(1, self.retries)):
            try:
                conn = self._connection()
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(payload)),
                }
                conn.request("POST", "/rpc", body=payload, headers=headers)
                response = conn.getresponse()
                data = response.read()
                if response.status != 200:
                    raise RuntimeError(f"RPC {method} failed with HTTP {response.status}: {data[:256]!r}")
                result = pickle.loads(data)
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(f"RPC {method} failed: {result['error']}")
                return result.get("result") if isinstance(result, dict) else result
            except Exception as exc:  # pragma: no cover - retry path is timing dependent.
                last_error = exc
                self.close()
                if attempt + 1 < max(1, self.retries):
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"RPC {method} failed after {self.retries} attempt(s): {last_error}") from last_error

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> HTTPConnection:
        if not self.keep_alive or self._conn is None:
            self._conn = HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self._conn


def make_pickle_rpc_handler(dispatch: Callable[[str, dict[str, Any]], Any]) -> type[BaseHTTPRequestHandler]:
    class PickleRpcHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
            if self.path != "/rpc":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = pickle.loads(self.rfile.read(length))
                result = dispatch(request["method"], request.get("kwargs", {}))
                payload = pickle.dumps({"result": sanitize_pickle_value(result)}, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_response(200)
            except Exception as exc:  # pragma: no cover - exercised through client error text.
                payload = pickle.dumps({"error": repr(exc)}, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_response(500)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
            return

    return PickleRpcHandler

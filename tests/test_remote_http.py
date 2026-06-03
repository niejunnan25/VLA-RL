from __future__ import annotations

from http.server import HTTPServer
import threading

import numpy as np

from vla_rl.runtime.remote_http import RemoteHttpRpcClient, make_pickle_rpc_handler


def test_remote_http_pickle_rpc_roundtrip():
    state = {"count": 0}

    def dispatch(method, kwargs):
        if method == "add":
            state["count"] += int(kwargs["value"])
            return {"count": state["count"], "array": np.array([state["count"]], dtype=np.float32)}
        raise ValueError(method)

    server = HTTPServer(("127.0.0.1", 0), make_pickle_rpc_handler(dispatch))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RemoteHttpRpcClient(f"http://127.0.0.1:{server.server_port}", timeout=5.0)
        result = client.call("add", value=np.int64(3))
        assert result["count"] == 3
        assert result["array"].shape == (1,)
    finally:
        client.close()
        server.shutdown()
        server.server_close()

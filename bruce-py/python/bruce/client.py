"""CLIENT-001 — typed Python client for bruce-server.

Replaces raw urllib calls with a clean OO interface:

    from bruce.client import BruceClient
    c = BruceClient("http://127.0.0.1:8080")
    c.write("fact1", k=[1.0, 0.0], v=[10.0, 20.0], owner="alice")
    k, v = c.read("fact1")
    out = c.attention(x=[1.0, 0.0], eps=1.0, sim="dot")
    c.delete("fact1", owner="alice")
    print(c.info(), c.metrics())

All methods are blocking. For high-throughput async use, see
`bruce.client.AsyncBruceClient` (TODO).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ServerInfo:
    """Decoded `/info` payload."""
    version: str
    d_k: int
    d_v: int
    alive: int
    total: int
    audit_len: int


class BruceClientError(Exception):
    """Raised on non-2xx HTTP responses from bruce-server."""

    def __init__(self, status: int, body: str, op: str):
        super().__init__(f"bruce-server {op} returned {status}: {body}")
        self.status = status
        self.body = body
        self.op = op


class BruceClient:
    """Typed synchronous client for bruce-server.

    Usage:
        c = BruceClient("http://127.0.0.1:8080")
        c.write("fact1", k=..., v=..., owner="alice")
        ...
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- low-level ---
    def _req(self, method: str, path: str, body: Any = None) -> Any:
        url = self.base_url + path
        data = (json.dumps(body).encode() if body is not None else None)
        headers = ({"content-type": "application/json"}
                    if body is not None else {})
        req = urllib.request.Request(url, data=data, headers=headers,
                                       method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = r.read().decode()
                # /metrics returns plain text, not JSON
                if path == "/metrics":
                    return txt
                return json.loads(txt) if txt else None
        except urllib.error.HTTPError as e:
            raise BruceClientError(e.code, e.read().decode(errors="ignore"),
                                     op=f"{method} {path}")
        except urllib.error.URLError as e:
            raise BruceClientError(-1, str(e), op=f"{method} {path}")

    # --- KvMemory CRUD ---
    def write(self, fact_id: str, k: list[float], v: list[float],
              owner: str) -> None:
        self._req("POST", "/facts", {
            "fact_id": fact_id, "k": list(k), "v": list(v), "owner": owner,
        })

    def read(self, fact_id: str) -> tuple[list[float], list[float]] | None:
        try:
            r = self._req("GET", f"/facts/{fact_id}")
        except BruceClientError as e:
            if e.status == 404:
                return None
            raise
        return list(r["k"]), list(r["v"])

    def delete(self, fact_id: str, owner: str) -> None:
        self._req("DELETE", f"/facts/{fact_id}?owner={owner}")

    # --- F_ε query ---
    def attention(self, x: list[float], eps: float,
                  sim: str = "dot") -> list[float]:
        return self._req("POST", "/query/attention", {
            "x": list(x), "eps": float(eps), "sim": sim,
        })

    # --- server inspection ---
    def info(self) -> ServerInfo:
        r = self._req("GET", "/info")
        return ServerInfo(version=r["version"], d_k=r["d_k"], d_v=r["d_v"],
                            alive=r["alive"], total=r["total"],
                            audit_len=r["audit_len"])

    def health(self) -> bool:
        try:
            self._req("GET", "/health")
            return True
        except BruceClientError:
            return False

    def audit_root(self) -> str:
        return self._req("GET", "/audit/root")

    def audit_length(self) -> int:
        return self._req("GET", "/audit/length")

    def metrics(self) -> dict[str, float]:
        """Parse Prometheus /metrics into a dict of metric name → value.
        Comment lines and blank lines are skipped."""
        txt = self._req("GET", "/metrics")
        out: dict[str, float] = {}
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return out

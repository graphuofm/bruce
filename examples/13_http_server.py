"""Bruce HTTP server — client round-trip.

Demonstrates how to talk to a deployed `bruce-server` over HTTP. This
is the same KvMemory + F_ε + audit log you get in-process, just behind
a network boundary so an agent (or a fleet of agents) can share one
memory.

Run:
    # one terminal: start the server
    cargo run -p bruce-server --release -- \\
        --addr 127.0.0.1:18080 --d-k 4 --d-v 2

    # another terminal:
    python examples/13_http_server.py

If `bruce-server` is not running we print a hint and skip — this file
also serves as a smoke test that the surface we ship matches the
single-node `KvMemory` API.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request

import json


HOST = "http://127.0.0.1:18080"


def get(path: str):
    with urllib.request.urlopen(HOST + path, timeout=3) as r:
        return json.loads(r.read())


def post(path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


def delete(path: str):
    req = urllib.request.Request(HOST + path, method="DELETE")
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


def main() -> None:
    try:
        info = get("/info")
    except (urllib.error.URLError, ConnectionRefusedError):
        print(f"bruce-server not reachable at {HOST}.")
        print(f"Start it with:")
        print(f"    cargo run -p bruce-server --release -- "
              f"--addr 127.0.0.1:18080 --d-k 4 --d-v 2")
        sys.exit(0)

    print(f"Connected to bruce-server v{info['version']}  "
          f"d_k={info['d_k']}  d_v={info['d_v']}")

    # write a tiny memory
    post("/facts", dict(fact_id="f1", k=[1, 0, 0, 0], v=[10, 0], owner="alice"))
    post("/facts", dict(fact_id="f2", k=[0, 1, 0, 0], v=[0, 20], owner="alice"))
    post("/facts", dict(fact_id="f3", k=[1, 1, 0, 0], v=[ 5,  5], owner="bob"))

    print(f"  wrote 3 facts; alive = {get('/info')['alive']}")

    # read one back exactly
    rec = get("/facts/f1")
    print(f"  GET /facts/f1 → k={rec['k']}  v={rec['v']}")

    # ε=0 indicator: x exactly equals k of f1
    out0 = post("/query/attention",
                dict(x=[1, 0, 0, 0], eps=0.0, sim="indicator"))
    # ε=1 softmax-dot: the textbook attention
    out1 = post("/query/attention",
                dict(x=[1, 0, 0, 0], eps=1.0, sim="dot"))
    print(f"  ε=0 indicator   → {out0}   (exact pick of f1)")
    print(f"  ε=1 softmax-dot → {[round(x,3) for x in out1]}")

    # owner-enforced delete: bob can delete f3, alice cannot
    delete("/facts/f3?owner=bob")
    print(f"  bob deleted f3; alive = {get('/info')['alive']}")
    try:
        delete("/facts/f1?owner=bob")
        raise AssertionError("should have been 403")
    except urllib.error.HTTPError as e:
        print(f"  bob deleting f1 → HTTP {e.code} (correct: owned by alice)")

    # audit log root after all ops
    print(f"  audit_len = {get('/audit/length')}")
    print(f"  audit_root = {get('/audit/root')[:20]}...")
    print("done — same surface as in-process KvMemory, just over HTTP.")


if __name__ == "__main__":
    main()

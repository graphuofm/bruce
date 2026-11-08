#!/usr/bin/env python
"""PostgreSQL call-sequence replay of the ladder rungs S2 / S3 / S4 (target/release/ladder_replay).

NOT an engine run: PostgreSQL 17.10 could not be rebuilt on this machine (README "Not run"), so
the values PostgreSQL would deliver are replayed outside the server over the same Rust
arithmetic, in PG's call order (see replay/src/main.rs). No latency is reported for replays.

  window : same grid as sqlite_window.py; errors vs the same numpy float64 reference
  ivm    : (a) the 10,000-op INSERT/DELETE/UPDATE stream of ../postgres/maint_ivm/run.py
               (IMDb all rows, score s0, RandomState(7), first 29 ops delete each facet's argmax row)
           (b) the time-ordered retention streams of ../timeorder/maint_ts.py on Amazon
               (T0 = 2020-01-01, 50,000 inserts, R = 30 and 2 days; time order and the random control)

usage: replay.py window|ivm
"""
import json, struct, subprocess, sys
import numpy as np
from common import (EPS, HERE, KS, NQ, REPLAY_BIN, WORK, errors, load, machine_state, reference, vmax_frames)

IVM_EPS = EPS


def run_window():
    out = HERE / "replay_window.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            if "rung" in r:
                done.add((r["dataset"], r["k"], r["eps"], r["q"]))
    inb, outb = WORK / "replay_in.bin", WORK / "replay_out.bin"
    with open(out, "a") as f:
        f.write(json.dumps(dict(header=True, machine_at_start=machine_state(),
                                note="PG call-order replay outside the server; no latency")) + "\n")
        for ds in ["imdb_all", "imdb_ge", "amazon"]:
            d = load(ds); n = len(d["value"])
            for k in KS:
                vmax = vmax_frames(d["value"], k)
                for eps in EPS:
                    for q in range(NQ):
                        if (ds, k, eps, q) in done:
                            continue
                        s = d[f"s{q}"]
                        with open(inb, "wb") as g:
                            g.write(struct.pack("<Q", n)); g.write(s.astype("<f8").tobytes()); g.write(d["value"].astype("<f8").tobytes())
                        p = subprocess.run([str(REPLAY_BIN), "window", str(inb), str(k), repr(eps), str(outb)],
                                           capture_output=True, text=True, check=True)
                        st = json.loads(p.stdout)
                        o = np.fromfile(outb, dtype="<f8").reshape(3, n)
                        ref = reference(d["value"], s, eps, k)
                        for i, rung in enumerate(["S2", "S3", "S4"]):
                            cell = dict(engine="postgres_replay", dataset=ds, k=k, eps=eps, q=q, rung=rung,
                                        **errors(o[i], ref, vmax))
                            if rung == "S4":
                                cell["stats"] = st
                            if p.stderr:
                                cell["stderr"] = p.stderr[:500]
                            f.write(json.dumps(cell) + "\n")
                        f.flush()
                        print(ds, k, eps, q, [json.loads(x) for x in []], st, flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")
    for p in (inb, outb):
        p.unlink(missing_ok=True)


# ------------------------------------------------------------------ ivm
def write_rows(path, facet, v, s):
    with open(path, "wb") as g:
        g.write(struct.pack("<Q", len(v))); g.write(facet.astype("<i4").tobytes())
        g.write(v.astype("<f8").tobytes()); g.write(s.astype("<f8").tobytes())


def write_init(path, ids):
    with open(path, "wb") as g:
        g.write(struct.pack("<Q", len(ids))); g.write(np.asarray(ids, dtype="<u4").tobytes())


def write_ops(path, ops):
    """ops: list of (kind, [id...]) or ('update', id, v, s)."""
    with open(path, "wb") as g:
        g.write(struct.pack("<Q", len(ops)))
        for op in ops:
            if op[0] == "update":
                g.write(struct.pack("<BIIdd", 2, 1, op[1], op[2], op[3]))
            else:
                ids = op[1]
                g.write(struct.pack("<BI", 0 if op[0] == "insert" else 1, len(ids)))
                g.write(np.asarray(ids, dtype="<u4").tobytes())


def maint_ivm_stream(d, N=10_000):
    """../postgres/maint_ivm/run.py make_stream, over the same rows (scores: data.py's s0)."""
    ids = np.asarray(d["id"]); fac = d["facet"]; s = d["s0"]; v = d["value"]
    order = np.lexsort((-s, fac))  # per facet, highest s0 first
    first = np.r_[True, fac[order][1:] != fac[order][:-1]]
    argmax = [int(ids[i]) for i in order[first]]
    argmax = [int(x) for x in sorted(argmax, key=lambda i: fac[i])]
    rng = np.random.RandomState(7)
    live = set(ids.tolist()); dead = []
    ops = []
    for i in argmax:
        ops.append(("delete", [i])); live.discard(i); dead.append(i)
    while len(ops) < N:
        u = rng.rand()
        if u < 0.45 or not dead:
            while True:
                i = int(ids[rng.randint(len(ids))])
                if i in live:
                    break
            ops.append(("delete", [i])); live.discard(i); dead.append(i)
        elif u < 0.90:
            i = dead.pop(rng.randint(len(dead)))
            ops.append(("insert", [i])); live.add(i)
        else:
            while True:
                i = int(ids[rng.randint(len(ids))])
                if i in live:
                    break
            donor = int(ids[rng.randint(len(ids))])
            ops.append(("update", i, float(v[donor]), float(s[donor])))
    return ops, len(argmax)


def retention_streams(d, T0_ms, N, R_days):
    """../timeorder/maint_ts.py build_streams over Amazon (rows sorted by ts, id)."""
    ts = d["ts_ms"]; ids = d["id"]
    R = int(R_days * 86400 * 1000)
    init = ids[(ts >= T0_ms - R) & (ts < T0_ms)]
    fsel = np.flatnonzero(ts >= T0_ms)[:N]
    fut_ids, fut_ts = ids[fsel], ts[fsel]
    live_ts = list(zip(ts[(ts >= T0_ms - R) & (ts < T0_ms)].tolist(), init.tolist()))
    head = 0; time_ops = []
    for t, i in zip(fut_ts.tolist(), fut_ids.tolist()):
        time_ops.append(("insert", [int(i)])); live_ts.append((t, i))
        cut = t - R; expired = []
        while live_ts[head][0] < cut:
            expired.append(int(live_ts[head][1])); head += 1
        if expired:
            time_ops.append(("delete", expired))
    rng = np.random.RandomState(5)
    perm = list(fut_ids[rng.permutation(len(fut_ids))])
    live = list(init.astype(int)); pos = {x: k for k, x in enumerate(live)}
    rand_ops = []; it = iter(perm)

    def remove_random(cnt):
        out = []
        for _ in range(cnt):
            k = rng.randint(len(live)); x = live[k]
            last = live.pop()
            if k < len(live):
                live[k] = last; pos[last] = k
            del pos[x]; out.append(int(x))
        return out
    for op in time_ops:
        if op[0] == "insert":
            x = int(next(it)); rand_ops.append(("insert", [x])); pos[x] = len(live); live.append(x)
        else:
            rand_ops.append(("delete", remove_random(len(op[1]))))
    return init, fut_ids, time_ops, rand_ops


def run_ivm():
    res = dict(machine_at_start=machine_state(), note="PG softavg_ivm trigger replay outside the server; no latency", runs=[])
    rows_b, init_b, ops_b = WORK / "ivm_rows.bin", WORK / "ivm_init.bin", WORK / "ivm_ops.bin"

    def go(label, setup, ops, eps_list, v_scale, cp_every):
        write_ops(ops_b, ops)
        for eps in eps_list:
            p = subprocess.run([str(REPLAY_BIN), "ivm", str(rows_b), str(init_b), str(ops_b), repr(eps), str(v_scale),
                                str(cp_every)], capture_output=True, text=True, check=True)
            r = json.loads(p.stdout)
            r.update(stream=label, eps=eps, setup=setup)
            if p.stderr:
                r["stderr"] = p.stderr[:500]
            res["runs"].append(r)
            print(label, eps, {a: (x["groups_recomputed"], x["per_statement_touched"]["max_abs_err_over_vscale"],
                                   x["per_statement_touched"]["wrong"]) for a, x in r["arms"].items()}, flush=True)

    # (a) maint_ivm stream on IMDb
    d = load("imdb_all")
    write_rows(rows_b, d["facet"], d["value"], d["s0"]); write_init(init_b, d["id"])
    ops, n_arg = maint_ivm_stream(d)
    mix = {k: sum(o[0] == k for o in ops) for k in ("delete", "insert", "update")}
    go("maint_ivm_10k", dict(corpus="imdb", rows=int(len(d["id"])), n_ops=len(ops), mix=mix,
                             first_ops_delete_argmax_rows=n_arg, seed=7, checkpoint_every=1000, v_scale=10.0),
       ops, IVM_EPS, 10.0, 1000)
    # (b) Amazon retention streams
    d = load("amazon")
    n_all = int(d["id"].max()) + 1
    fac = np.zeros(n_all, np.int32); v = np.zeros(n_all); s = np.zeros(n_all)
    fac[d["id"]] = d["facet"]; v[d["id"]] = d["value"]; s[d["id"]] = d["s0"]
    write_rows(rows_b, fac, v, s)
    T0 = int(np.datetime64("2020-01-01T00:00:00", "ms").astype(np.int64))
    for R in (30, 2):
        init, fut, t_ops, r_ops = retention_streams(d, T0, 50_000, R)
        write_init(init_b, init)
        setup = dict(corpus="amazon", t0="2020-01-01", retention_days=R, initial_rows=int(len(init)), n_inserts=int(len(fut)),
                     statements=len(t_ops), delete_statements=sum(o[0] == "delete" for o in t_ops),
                     rows_deleted=sum(len(o[1]) for o in t_ops if o[0] == "delete"), v_scale=5.0,
                     checkpoint_every_statements=2000)
        go(f"amazon_time_R{R}", setup, t_ops, IVM_EPS, 5.0, 2000)
        go(f"amazon_random_R{R}", setup, r_ops, IVM_EPS, 5.0, 2000)
    res["machine_at_end"] = machine_state()
    (HERE / "replay_ivm.json").write_text(json.dumps(res, indent=1))
    for p in (rows_b, init_b, ops_b):
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    {"window": run_window, "ivm": run_ivm}[sys.argv[1]]()

"""Minimal eps-algebra frontend: parse -> plan -> EXPLAIN -> execute.

Grammar (one query shape, deliberately minimal):

    SELECT <group_col>, SOFTAVG(<val_col> WEIGHT sim(<emb_col>, :q) TEMP <eps>)
    FROM <table>
    [WHERE <col> >= <const>]
    GROUP BY <group_col>

The point is the compilation: the exact predicate and the soft aggregate
land in ONE plan, the optimizer pushes the eps=0 filter below the eps>0
fold, and the physical operator is the tiled online-softmax fold
(PartialTriple combine) that the Rust kernel executes via masked_attention.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from . import _bruce as _k

_QUERY_RE = re.compile(
    r"SELECT\s+(?P<group>\w+)\s*,\s*"
    r"SOFTAVG\(\s*(?P<val>\w+)\s+WEIGHT\s+sim\(\s*(?P<emb>\w+)\s*,\s*:q\s*\)\s+"
    r"TEMP\s+(?P<eps>[\w.+-]+)\s*\)\s+"
    r"FROM\s+(?P<table>\w+)\s*"
    r"(?:WHERE\s+(?P<fcol>\w+)\s*>=\s*(?P<fval>[\d.]+)\s*)?"
    r"GROUP\s+BY\s+(?P<gcol>\w+)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class SoftAggQuery:
    table: str
    group_col: str
    val_col: str
    emb_col: str
    eps: float
    filter_col: str | None
    filter_val: float | None

    @staticmethod
    def parse(sql: str) -> "SoftAggQuery":
        m = _QUERY_RE.match(sql.strip())
        if not m:
            raise ValueError(f"cannot parse query: {sql!r}")
        if m.group("group") != m.group("gcol"):
            raise ValueError("SELECT column must match GROUP BY column")
        eps_txt = m.group("eps").lower()
        eps = float("inf") if eps_txt in ("inf", "infinity") else float(eps_txt)
        return SoftAggQuery(
            table=m.group("table"),
            group_col=m.group("group"),
            val_col=m.group("val"),
            emb_col=m.group("emb"),
            eps=eps,
            filter_col=m.group("fcol"),
            filter_val=float(m.group("fval")) if m.group("fval") else None,
        )

    # -- planning ------------------------------------------------------
    def logical_plan(self) -> str:
        f = (
            f"  Filter[eps=0]  {self.filter_col} >= {self.filter_val}\n"
            if self.filter_col
            else ""
        )
        return (
            f"GroupSoftAvg[eps={self.eps}]  key={self.group_col}, "
            f"val={self.val_col}, weight=exp(sim({self.emb_col}, :q)/eps)\n"
            + f + f"  Scan  {self.table}"
        )

    def physical_plan(self, n_rows: int | None = None, n_groups: int | None = None) -> str:
        nr = f" rows~{n_rows}" if n_rows else ""
        ng = f" groups~{n_groups}" if n_groups else ""
        lines = [f"Fold: fused grouped online-softmax (m,num,den), kernel=grouped_softavg,"]
        lines.append(f"      eps={self.eps}, order-invariant, single pass, O(groups) state{ng}")
        lines.append(f"MaskStream: (group_id, row_id) pairs from GROUP BY {self.group_col}{nr}")
        if self.filter_col:
            lines.append(
                f"  pushed-down Filter[eps=0]: {self.filter_col} >= {self.filter_val}"
                f"  (exact predicate evaluated BELOW the soft fold; rows never scored)"
            )
        lines.append(f"Scan {self.table}: columns [{self.group_col}, {self.val_col}, {self.emb_col}]")
        return "\n".join(lines)

    def explain(self, **kw) -> str:
        return (
            "== logical plan ==\n" + self.logical_plan()
            + "\n== physical plan ==\n" + self.physical_plan(**kw)
        )

    # -- execution -----------------------------------------------------
    def execute_grouped(
        self,
        group_codes: np.ndarray,
        n_groups: int,
        values: np.ndarray,
        embs: np.ndarray,
        query_emb: np.ndarray,
        filter_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run the plan on a dictionary-encoded grouping column.

        `group_codes` is the storage layer's job: encode once at load,
        not per query. The exact filter is fused into the kernel scan
        (`sel`), so filtered rows are never scored. Returns
        (covered_mask, answers, timings).
        """
        t = {}
        t0 = time.perf_counter()
        out, covered = _k.grouped_softavg(
            np.ascontiguousarray(query_emb, dtype=np.float64),
            np.ascontiguousarray(embs, dtype=np.float64),
            np.ascontiguousarray(values.reshape(-1, 1), dtype=np.float64),
            np.ascontiguousarray(group_codes, dtype=np.uint32),
            n_groups,
            eps=self.eps,
            sel=None if filter_mask is None
                else np.ascontiguousarray(filter_mask, dtype=bool),
        )
        t["fused_scan_s"] = time.perf_counter() - t0
        return np.asarray(covered), out[:, 0], t

    def execute(
        self,
        group_keys: np.ndarray,
        values: np.ndarray,
        embs: np.ndarray,
        query_emb: np.ndarray,
        filter_mask: np.ndarray | None = None,
    ) -> tuple[list, np.ndarray, dict]:
        """Run the plan on column arrays. Returns (group_labels, answers, timings)."""
        t = {}
        t0 = time.perf_counter()
        if filter_mask is not None:
            keep = np.flatnonzero(filter_mask)
        else:
            keep = np.arange(len(values))
        t["filter_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        labels, gid = np.unique(np.asarray(group_keys)[keep], return_inverse=True)
        pairs = np.column_stack([gid, keep]).astype(np.int64)
        t["mask_stream_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        q = np.repeat(query_emb.reshape(1, -1), len(labels), axis=0).astype(np.float64)
        out, nonempty = _k.masked_attention(
            np.ascontiguousarray(q),
            np.ascontiguousarray(embs, dtype=np.float64),
            np.ascontiguousarray(values.reshape(-1, 1), dtype=np.float64),
            pairs,
            eps=self.eps,
        )
        t["fold_s"] = time.perf_counter() - t0
        return list(labels), out[:, 0], t


# ----------------------------------------------------------------------
# TemperaturePlan: the whole running example as ONE plan.
# ----------------------------------------------------------------------

@dataclass
class TemperaturePlan:
    """Three temperature stages compiled as one plan over one store:

        (1) eps = 0   exact filter          (the mask)
        (2) eps > 0   soft group-aggregate  (the fold; SoftAggQuery)
        (3) eps = 1   attention read over the winning group's rows
                      (the external-memory input a model runtime reads)

    What one plan buys that a glued stack cannot express:

      * the similarity column is computed ONCE and shared by stage (2)'s
        weights, stage (3)'s read, and the truncation contract;
      * stage (2)'s fold statistics certify a truncated stage-(3) read:
        the omitted softmax mass delta bounds the read's error, so the
        optimizer may substitute top-k under an explicit error contract
        (never by silently changing semantics);
      * one deletion updates the maintained state of BOTH stages.
    """

    agg: SoftAggQuery
    read_eps: float = 1.0
    read_top_k: int | None = None   # truncated stage-3 read; None = exact

    @staticmethod
    def parse(sql: str, read_eps: float = 1.0,
              read_top_k: int | None = None) -> "TemperaturePlan":
        return TemperaturePlan(SoftAggQuery.parse(sql), read_eps, read_top_k)

    # -- planning ------------------------------------------------------
    def explain(self) -> str:
        a = self.agg
        read = (f"top-{self.read_top_k} rows, contract: |err| <= "
                f"delta*(1 + 1/(1-delta))*max|v|, delta = omitted weight mass"
                if self.read_top_k else "all winning rows (exact)")
        return "\n".join([
            "== TemperaturePlan (one store, one similarity pass) ==",
            f"(3) AttentionRead[eps={self.read_eps}]  x=:read_q, "
            f"K=V={a.emb_col} of winning group; {read}",
            f"(2) GroupSoftAvg[eps={a.eps}]  key={a.group_col}, "
            f"val={a.val_col}, weights exp(sim/eps); state (m,num,den)/group",
            (f"(1) Filter[eps=0]  {a.filter_col} >= {a.filter_val}  "
             f"(pushed below all scoring)" if a.filter_col else
             "(1) Filter[eps=0]  none"),
            f"    Scan {a.table}: [{a.group_col}, {a.val_col}, {a.emb_col}]"
            f" -> sim({a.emb_col}, :q) computed once, shared by (2), (3),"
            f" and the contract",
        ])

    # -- execution -----------------------------------------------------
    def execute(
        self,
        group_keys: np.ndarray,
        values: np.ndarray,
        embs: np.ndarray,
        query_emb: np.ndarray,
        filter_mask: np.ndarray | None = None,
        read_query: np.ndarray | None = None,
    ) -> dict:
        """Run all three stages. Returns a dict with per-stage outputs,
        the shared statistics, and (if read_top_k) the error contract."""
        t: dict[str, float] = {}
        a = self.agg

        # (1) eps=0 filter
        t0 = time.perf_counter()
        keep = (np.flatnonzero(filter_mask) if filter_mask is not None
                else np.arange(len(values)))
        t["filter_s"] = time.perf_counter() - t0

        # shared similarity pass
        t0 = time.perf_counter()
        sims = embs[keep].astype(np.float64) @ query_emb.astype(np.float64)
        t["sim_s"] = time.perf_counter() - t0

        # (2) eps>0 fold over precomputed sims (1-d keys: kernel sim == sims)
        t0 = time.perf_counter()
        labels, gid = np.unique(np.asarray(group_keys)[keep], return_inverse=True)
        pairs = np.column_stack([gid, np.arange(len(keep))]).astype(np.int64)
        q1 = np.ones((len(labels), 1), dtype=np.float64)
        out, _ = _k.masked_attention(
            q1, np.ascontiguousarray(sims.reshape(-1, 1)),
            np.ascontiguousarray(values[keep].reshape(-1, 1).astype(np.float64)),
            pairs, eps=a.eps)
        answers = out[:, 0]
        t["fold_s"] = time.perf_counter() - t0

        # winning group
        win = int(np.argmax(answers))
        wrows = np.flatnonzero(gid == win)          # indices into keep[]
        ws = sims[wrows]

        # (3) eps=read_eps attention read over the winning group's rows
        x_read = (read_query if read_query is not None else query_emb)
        K = embs[keep][wrows].astype(np.float64)
        rs = K @ x_read.astype(np.float64)
        t0 = time.perf_counter()
        w = np.exp((rs - rs.max()) / self.read_eps)
        read_full = (w[:, None] * K).sum(0) / w.sum()
        t["read_s"] = time.perf_counter() - t0

        res = {"labels": list(labels), "answers": answers,
               "winning_group": labels[win], "winning_rows": len(wrows),
               "read_full": read_full, "timings": t}

        if self.read_top_k:
            k = min(self.read_top_k, len(wrows))
            top = np.argsort(rs)[::-1][:k]
            t0 = time.perf_counter()
            wt = w[top]
            read_top = (wt[:, None] * K[top]).sum(0) / wt.sum()
            t["read_topk_s"] = time.perf_counter() - t0
            delta = 1.0 - float(wt.sum() / w.sum())
            bound = delta * (1.0 + 1.0 / (1.0 - delta)) * np.abs(K).max()
            res.update(read_topk=read_top, omitted_mass=delta,
                       certified_bound=bound,
                       measured_err=float(np.abs(read_top - read_full).max()))
        return res


class MaintainedPlan:
    """Maintained state for a TemperaturePlan: one IncrementalMemory per
    group for stage (2) (d_v = 1) and one for stage (3) over the winning
    group's rows (d_v = dim). A single ``delete(key_id)`` updates both
    states, so the aggregate AND the model-facing attention read refresh
    from one operation. (Deleting a score that ties a memory's running
    max triggers that memory's one-pass re-anchor, per the kernel
    contract.)"""

    def __init__(self, plan: TemperaturePlan, ids, group_keys, values,
                 embs, query_emb, winning_group,
                 read_query=None, groups=None):
        self.plan = plan
        a = plan.agg
        ids = np.asarray(ids).astype(str)
        gk = np.asarray(group_keys)
        embs = np.asarray(embs, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64)
        x_read = np.asarray(read_query if read_query is not None
                            else query_emb, dtype=np.float64)
        self._group_of = dict(zip(ids.tolist(), gk.tolist()))
        self.win = winning_group
        want = set(groups) if groups is not None else None

        self.mem2: dict[str, object] = {}
        for g in np.unique(gk):
            if want is not None and g not in want:
                continue
            m = _k.IncrementalMemory(
                query=np.ascontiguousarray(query_emb, dtype=np.float64),
                eps=a.eps, d_v=1, sim="dot")
            gm = gk == g
            m.insert_many(ids[gm].tolist(),
                          np.ascontiguousarray(embs[gm]),
                          np.ascontiguousarray(vals[gm].reshape(-1, 1)))
            self.mem2[g] = m

        wm = gk == winning_group
        self.mem3 = _k.IncrementalMemory(
            query=np.ascontiguousarray(x_read), eps=plan.read_eps,
            d_v=embs.shape[1], sim="dot")
        self.mem3.insert_many(ids[wm].tolist(),
                              np.ascontiguousarray(embs[wm]),
                              np.ascontiguousarray(embs[wm]))

    def delete(self, key_id: str) -> dict:
        """Delete one record; both maintained stages refresh."""
        g = self._group_of[str(key_id)]
        if g in self.mem2:
            self.mem2[g].delete(str(key_id))
        if g == self.win:
            self.mem3.delete(str(key_id))
        return self.outputs(g)

    def outputs(self, group=None) -> dict:
        group = group if group is not None else self.win
        out = {"read": self.mem3.output()}
        if group in self.mem2:
            out["agg"] = float(self.mem2[group].output()[0])
        return out

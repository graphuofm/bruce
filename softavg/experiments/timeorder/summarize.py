#!/usr/bin/env python
"""Merge this directory's raw outputs into results.json and render README.md's
generated block (no number in README.md is typed by hand). Server must be up."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import EPS_GRID, TIMING_STATUS, environment, server_env  # noqa: E402

BEGIN, END = "<!-- BEGIN GENERATED (summarize.py) -->", "<!-- END GENERATED -->"


def jsonl(p):
    head = foot = None; rows = []
    for line in open(p):
        c = json.loads(line)
        if c.get("header"):
            head = c
        elif c.get("footer"):
            foot = c
        else:
            rows.append(c)
    return head, foot, rows


def ms(x):
    return "-" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def e(x):
    return "-" if x is None else ("0" if x == 0 else f"{x:.1e}")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def amazon_stats():
    import pandas as pd
    d = pd.read_parquet(HERE / "amazon_ts.parquet")
    return dict(rows=int(len(d)), verified_rating_and_year=int(d.ok.sum()), ts_min=str(d.ts.min()), ts_max=str(d.ts.max()),
                distinct_ts=int(d.ts.nunique()), rows_per_year={int(k): int(v) for k, v in d.ts.dt.year.value_counts().sort_index().items()},
                source="McAuley-Lab/Amazon-Reviews-2023 raw/review_categories/<cat>.jsonl, streamed by build_amazon_ts.py "
                       "(2026-11-06) up to the corpus' last line per category; nothing written to disk but amazon_ts.parquet",
                load=json.loads((HERE / "load_amazon_ts.json").read_text()))


def assemble():
    prev = json.loads((HERE / "results.json").read_text())
    imdb_build = prev.get("imdb_runs_build") or dict(server=prev["server"], environment=prev["environment"])
    h, f, cells = jsonl(HERE / "window_ts.jsonl")
    ha, fa, ca = jsonl(HERE / "window_ts_amazon.jsonl")
    res = dict(
        description="Time-ordered windows and maintenance in PostgreSQL on real timestamps: IMDb release dates and "
                    "Amazon review timestamps (recovered 2026-11-06): sliding windows ORDER BY ts with ROWS and time-based "
                    "RANGE frames vs ORDER BY id on the same rows; softavg_ivm under a timestamp-order insert stream "
                    "with sliding retention vs a random stream with the same statement and row counts.",
        timing_status=TIMING_STATUS, environment=environment(), server=server_env("rel"),
        builds_note="`server`/`environment` = the build of the Amazon runs (after softagg-core's B1 bound change); "
                    "the IMDb runs (window, maint) used the earlier build recorded in `imdb_runs_build`",
        imdb_runs_build=imdb_build,
        server_build_per_arm={"window_ts": "release", "maint_ts": "release", "window_ts_amazon": "release",
                              "maint_ts_amazon": "release"},
        sources=dict(imdb_ts=json.loads((HERE / "imdb_ts_stats.json").read_text()),
                     amazon=amazon_stats(),
                     stackexchange="donfu/oa-stackexchange has no CreationDate / post id (METADATA: answer_score, "
                                   "question_score, tags) -> no timestamps recoverable from the source used"),
        window=dict(header=h, machine_at_end=f and f["machine_at_end"], cells=cells),
        maint=[json.loads(p.read_text()) for p in sorted(HERE.glob("results_maint_ts_R*.json"),
                                                         key=lambda p: float(p.stem.split("_R")[1]))],
        window_amazon=dict(header=ha, machine_at_end=fa and fa["machine_at_end"], cells=ca),
        maint_amazon=[json.loads(p.read_text()) for p in sorted(HERE.glob("results_maint_ts_amazon_R*.json"),
                                                                key=lambda p: -float(p.stem.split("_R")[1]))],
    )
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    return res


def range_label(k, unit):
    if unit == "days":
        return f"{k} days"
    return {3600: "1 hour", 86400: "1 day"}.get(k, f"{k // 86400} days" if k % 86400 == 0 else f"{k} s")


def window_section(w, eps, title):
    unit = w["header"].get("range_unit", "days")
    out = ["", title, "", "softavg_moving: ms, abs err / max|v|, punts / inverses. Baselines: ms (max abs err / max|v|) or the error."]
    rows = []
    idx = {(c["order"], c["frame"], c["k"], c["eps"], c["plan"]): c for c in w["cells"]}
    frames = list(dict.fromkeys((c["order"], c["frame"], c["k"]) for c in w["cells"]))
    for o, kind, k in frames:
        fw = idx[(o, kind, k, eps[0], "softavg_moving")]["frame_width"]
        label = f"{o} {kind} {range_label(k, unit) if kind == 'range' else k} (width mean {fw['mean']:.0f}, max {fw['max']})"
        for p in ("softavg_moving", "softavg_fwd", "naive_window"):
            row = [label, p]
            for ep in eps:
                c = idx.get((o, kind, k, ep, p))
                if c is None:
                    row.append("-"); continue
                if not c["sql_ok"]:
                    row.append("ERR " + (c["error_message"] or "").replace("ERROR:  value out of range: ", "")); continue
                txt = f"{ms(c['median_ms'])}, {e(c.get('max_abs_err_over_vmax'))}"
                if p == "softavg_moving":
                    txt += f", {c['punts']}/{c['inverses']}"
                if not c.get("protocol_window_ok", True):
                    txt += " (OVER BOUND)"
                if c.get("n_nonfinite"):
                    txt += f" nonfinite {c['n_nonfinite']}"
                row.append(txt)
            rows.append(row)
    return out + ["", table(["frame", "plan"] + eps, rows)]


def maint_section(M, what):
    st = M["setup"]
    out = ["", f"### {what}: softavg_ivm, time-ordered stream with sliding retention R = {st['retention_days']:g} days vs random stream", "",
           f"initial table: {st['initial_rows']:,} rows (timestamp in [{st['t0']} - {st['retention_days']:g} d, {st['t0']})); "
           f"{st['n_inserts']:,} inserts ({st['inserted_ts_range'][0][:10]} .. {st['inserted_ts_range'][1][:10]}); time stream: "
           f"{st['delete_statements']:,} retention DELETE statements removing {st['rows_deleted']:,} rows; the random stream "
           f"issues the same statements with the same row counts (deleted rows uniform over live rows); {st['commit']}; "
           f"v_scale {st['v_scale']:g}; machine at start quiet={M['machine_at_start'].get('quiet')}", ""]
    rows = []
    for r in M["runs"]:
        sd, si = r["statements"].get("delete", {}), r["statements"]["insert"]
        rows.append([r["eps"], r["stream"], sd.get("rows", 0), r["argmax_rows_deleted"], sd.get("recomputed", 0), si["recomputed"],
                     f"{ms(si['median_ms'])} / {ms(sd.get('median_ms'))}", f"{ms(si['p99_ms'])} / {ms(sd.get('p99_ms'))}",
                     e(r["max_abs_err_over_vscale"]), r["group_mismatch_total"]])
    return out + [table(["eps", "stream", "rows deleted", "argmax rows deleted", "groups recomputed (delete stmts)", "(insert stmts)",
                         "ms median ins / del stmt", "ms p99 ins / del stmt", "max drift / v_scale", "group mismatches"], rows)]


def render(R):
    eps = [str(x) for x in EPS_GRID]
    s = R["sources"]["imdb_ts"]; a = R["sources"]["amazon"]; w = R["window"]; wa = R["window_amazon"]
    env = R["server"]; ienv = R["imdb_runs_build"]["server"]; iext = R["imdb_runs_build"]["environment"]["extension"]
    out = ["_All timings: smoke, machine shared._", "",
           f"- Amazon runs: {env['pg_version'].split(',')[0]}, build **{env['server_build']}**, bruce_pg.so md5 `{env['bruce_pg_so_md5']}`, "
           f"extension build **{R['environment']['extension']['build_type']}**, source sha256 `{R['environment']['extension']['source_sha256'][:16]}...` "
           f"(softagg-core with the B1 bound change)",
           f"- IMDb runs: bruce_pg.so md5 `{ienv['bruce_pg_so_md5']}`, build **{ienv['server_build']}** / **{iext['build_type']}**, "
           f"source sha256 `{iext['source_sha256'][:16]}...` (before B1)",
           f"- Amazon review timestamps: {a['rows']:,} rows recovered, {a['verified_rating_and_year']:,} pass the rating + year check; "
           f"range {a['ts_min'][:10]} .. {a['ts_max'][:10]}; {a['distinct_ts']:,} distinct timestamps (ms); "
           f"score s0 = query row {a['load']['query_row']} (\"{a['load']['query_text']}\")",
           f"- IMDb release dates: {s['items_with_ts']:,} of {s['items']:,} movies ({s['coverage']:.1%}) have a parsable release date; "
           f"precision {s['precision']}; range {s['ts_min'][:10]} .. {s['ts_max'][:10]}, median {s['ts_quantiles']['0.5'][:10]}",
           f"- machine at start of the window runs: IMDb quiet={w['header']['machine_at_start'].get('quiet')}, "
           f"Amazon quiet={wa['header']['machine_at_start'].get('quiet')}"]
    out += window_section(wa, eps, f"### Amazon: sliding windows, ORDER BY id vs ORDER BY ts (same {wa['header']['rows']:,} reviews with "
                                   f"timestamp in [{wa['header']['t0']}, {wa['header']['t1']}), score s0)")
    for M in R["maint_amazon"]:
        out += maint_section(M, "Amazon")
    out += window_section(w, eps, f"### IMDb: sliding windows, ORDER BY id vs ORDER BY ts (same {w['header']['rows']:,} rows: "
                                  f"day-precision release dates in [{w['header']['t0']}, {w['header']['t1']}), score s0)")
    for M in R["maint"]:
        out += maint_section(M, "IMDb")
    return "\n".join(out)


if __name__ == "__main__":
    R = assemble()
    txt = render(R)
    p = HERE / "README.md"
    s = p.read_text()
    a, b = s.index(BEGIN) + len(BEGIN), s.index(END)
    p.write_text(s[:a] + "\n" + txt + "\n" + s[b:])
    print(txt)

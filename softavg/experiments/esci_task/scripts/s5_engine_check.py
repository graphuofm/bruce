"""Cross-check: the softavg numbers used by s4 equal what the shipped DuckDB
extension (engines/duckdb/build/softavg.duckdb_extension, over
softagg-core) returns for the same GROUP BY inside DuckDB.

  SELECT query_id, <facet>, softavg(rating, s, eps) FROM task GROUP BY ALL

is run for every eps in the grid on the full task table (labelled +
distractor pool) for both facets, and compared with the numpy estimator of
s4 (Setting.softavg).  Writes data/engine_check.json.
"""
import hashlib, json, sys, time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from s4_evaluate import EPS, Setting  # noqa: E402

EXT = Path("/home/jding/bruce/paper_icde/engines/duckdb/build/softavg.duckdb_extension")
T = pd.read_parquet(HERE / "data" / "task.parquet")
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{EXT}'")
con.register("task", T)
out = dict(extension=str(EXT),
           extension_sha256_16=hashlib.sha256(EXT.read_bytes()).hexdigest()[:16],
           duckdb_version=duckdb.__version__, checks=[])
for facet in ["brand", "main_category"]:
    st = Setting(T, facet, "rating", ["E"])
    for eps in EPS + [float("inf")]:
        t0 = time.time()
        lit = "'infinity'" if eps == float("inf") else repr(eps)
        r = con.execute(f"""select query_id, {facet} as f, softavg(rating, s, {lit}::double) as sa
                            from task where {facet} is not null and rating is not null
                            group by all""").df()
        dt = time.time() - t0
        # align: s4 groups are sorted by (query_id, facet)
        r = r.sort_values(["query_id", "f"], kind="mergesort").reset_index(drop=True)
        ref = st.softavg(eps)
        assert len(r) == st.G and np.array_equal(r.query_id.values, st.gq)
        rel = np.abs(r.sa.values - ref) / np.maximum(np.abs(ref), 1e-300)
        out["checks"].append(dict(facet=facet, eps=eps, groups=int(st.G),
                                  max_rel_diff=float(rel.max()), seconds_smoke=round(dt, 3)))
        print(out["checks"][-1], flush=True)
(HERE / "data" / "engine_check.json").write_text(json.dumps(out, indent=1))

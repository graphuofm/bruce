"""Join coverage of ESCI US products against Amazon Reviews 2023 item metadata
(match: ESCI product_id == AR2023 parent_asin).  Writes data/coverage.json."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"
e = pd.read_parquet(D / "esci_us.parquet", columns=["query_id", "product_id", "esci_label"])
m = (pd.read_parquet(D / "meta_join.parquet")
       .sort_values("rating_number", ascending=False).drop_duplicates("product_id"))
rep = json.loads((D / "meta_join_report.json").read_text())
prods = pd.Index(e.product_id.unique())
pm = m.set_index("product_id").reindex(prods)
j = e.merge(m, on="product_id", how="left")
g = j.groupby("query_id").agg(n_rated=("average_rating", lambda x: int(x.notna().sum())),
                              n_priced=("price", lambda x: int(x.notna().sum())))
cov = dict(
    match_rule="ESCI product_id (ASIN) == Amazon Reviews 2023 meta parent_asin",
    meta_source="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_*.jsonl.gz (streamed, not stored)",
    meta_files=len(rep), meta_lines_scanned=int(sum(r["lines"] for r in rep)),
    meta_uncompressed_gb_streamed=round(sum(r["uncompressed_bytes"] for r in rep) / 1e9, 1),
    esci_us_judgments=int(len(e)), esci_us_queries=int(e.query_id.nunique()),
    esci_us_products=int(len(prods)),
    products_matched=int(pm.mc_category_file.notna().sum()),
    products_with_rating=int(pm.average_rating.notna().sum()),
    products_with_price=int(pm.price.notna().sum()),
    frac_products_with_rating=float(pm.average_rating.notna().mean()),
    frac_products_with_price=float(pm.price.notna().mean()),
    frac_judgments_with_rating=float(j.average_rating.notna().mean()),
    frac_judgments_with_price=float(j.price.notna().mean()),
    frac_judgments_with_rating_by_label={k: float(v) for k, v in
                                         j.groupby("esci_label").average_rating
                                          .apply(lambda x: x.notna().mean()).items()},
    queries_with_at_least_k_rated={k: int((g.n_rated >= k).sum()) for k in [1, 4, 8, 12, 16]},
    queries_with_at_least_k_priced={k: int((g.n_priced >= k).sum()) for k in [1, 4, 8, 12, 16]},
    per_meta_file=rep,
)
(D / "coverage.json").write_text(json.dumps(cov, indent=1))
print(json.dumps({k: v for k, v in cov.items() if k != "per_meta_file"}, indent=1))

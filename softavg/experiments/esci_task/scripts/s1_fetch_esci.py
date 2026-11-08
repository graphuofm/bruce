"""Step 1a: pull the US-locale ESCI judgments and US product catalogue.

Source: the official release, github.com/amazon-science/esci-data
(shopping_queries_dataset_{examples,products}.parquet, 51 MB + 1.1 GB).
DuckDB reads the remote parquet with column projection, so the long
description / bullet-point columns of the product file are never fetched.
(The HF mirror tasksource/esci rate-limits anonymous clients with 429.)

Outputs
  data/esci_us.parquet      one row per (query, product) judgment, US locale:
                            query_id, query, product_id, esci_label,
                            small_version, split, product_title, product_brand
  data/catalog_us.parquet   all US products: product_id, product_title,
                            product_brand (used for distractors)
"""
import time
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"
D.mkdir(exist_ok=True)
GH = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset"
EX = f"{GH}/shopping_queries_dataset_examples.parquet"
PR = f"{GH}/shopping_queries_dataset_products.parquet"

t0 = time.time()
con = duckdb.connect()
con.sql("set http_retries = 10")
con.sql(f"""copy (select product_id, product_title, product_brand
                  from '{PR}' where product_locale = 'us')
            to '{D}/catalog_us.parquet' (format parquet, compression zstd)""")
con.sql(f"""copy (select e.query_id, e.query, e.product_id, e.esci_label,
                         e.small_version, e.split, p.product_title, p.product_brand
                  from '{EX}' e join '{D}/catalog_us.parquet' p using (product_id)
                  where e.product_locale = 'us')
            to '{D}/esci_us.parquet' (format parquet, compression zstd)""")
print(con.sql(f"select count(*) n, count(distinct query_id) q, count(distinct product_id) p "
              f"from '{D}/esci_us.parquet'"))
print(con.sql(f"select esci_label, count(*) from '{D}/esci_us.parquet' group by 1 order by 1"))
print(con.sql(f"select count(*) from '{D}/catalog_us.parquet'"))
print(f"{time.time()-t0:.0f}s")

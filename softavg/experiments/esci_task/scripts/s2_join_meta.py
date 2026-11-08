"""Step 1b: join ESCI US products to Amazon Reviews 2023 item metadata.

Each meta_<category>.jsonl.gz (McAuley lab mirror; the HF copy rate-limits
anonymous clients) is streamed through `curl | gunzip`; nothing is written
to disk except the rows whose parent_asin is an ESCI US product_id.
Only the regex-located parent_asin is checked per line; matching lines
are JSON-parsed.

Output: data/meta_join.parquet
  product_id, mc_category_file, main_category, store, average_rating,
  rating_number, price
and data/meta_join_report.json (per-file lines scanned / matched / bytes).
"""
import json, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"
CATS = """All_Beauty Amazon_Fashion Appliances Arts_Crafts_and_Sewing Automotive
Baby_Products Beauty_and_Personal_Care Books CDs_and_Vinyl Cell_Phones_and_Accessories
Clothing_Shoes_and_Jewelry Digital_Music Electronics Gift_Cards Grocery_and_Gourmet_Food
Handmade_Products Health_and_Household Health_and_Personal_Care Home_and_Kitchen
Industrial_and_Scientific Kindle_Store Magazine_Subscriptions Movies_and_TV
Musical_Instruments Office_Products Patio_Lawn_and_Garden Pet_Supplies Software
Sports_and_Outdoors Subscription_Boxes Tools_and_Home_Improvement Toys_and_Games
Unknown Video_Games""".split()
PA = re.compile(rb'"parent_asin": "([^"]+)"')


def parse_price(p):
    if p is None:
        return None
    if isinstance(p, (int, float)):
        return float(p)
    m = re.search(r"[\d,]+\.?\d*", str(p))
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def run(cat, ids_path):
    ids = set(Path(ids_path).read_text().split())
    t0 = time.time()
    cmd = f"curl -sL --retry 5 {BASE}/meta_{cat}.jsonl.gz | gunzip"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1 << 22)
    rows, n, nbytes = [], 0, 0
    for line in p.stdout:
        n += 1
        nbytes += len(line)
        m = PA.search(line)
        if not m or m.group(1).decode() not in ids:
            continue
        d = orjson.loads(line)
        rows.append(dict(product_id=d["parent_asin"], mc_category_file=cat,
                         main_category=d.get("main_category"), store=d.get("store"),
                         average_rating=d.get("average_rating"),
                         rating_number=d.get("rating_number"),
                         price=parse_price(d.get("price"))))
    p.wait()
    return cat, rows, dict(category=cat, lines=n, uncompressed_bytes=nbytes,
                           matched=len(rows), seconds=round(time.time() - t0, 1),
                           returncode=p.returncode)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cats = sys.argv[2].split(",") if len(sys.argv) > 2 else CATS
    ids_path = HERE / "data" / "esci_us_ids.txt"
    if not ids_path.exists():
        ids = pd.read_parquet(HERE / "data" / "esci_us.parquet",
                              columns=["product_id"]).product_id.unique()
        ids_path.write_text("\n".join(ids))
    out_dir = HERE / "data" / "meta_parts"
    out_dir.mkdir(exist_ok=True)
    todo = [c for c in cats if not (out_dir / f"{c}.json").exists()]
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(run, c, str(ids_path)) for c in todo]
        for f in as_completed(futs):
            cat, rows, rep = f.result()
            (out_dir / f"{cat}.json").write_text(json.dumps(dict(report=rep, rows=rows)))
            print(json.dumps(rep), flush=True)
    rows, reps = [], []
    for c in cats:
        f = out_dir / f"{c}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows += d["rows"]
            reps.append(d["report"])
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "data" / "meta_join.parquet", index=False)
    (HERE / "data" / "meta_join_report.json").write_text(json.dumps(reps, indent=1))
    print(len(df), df.product_id.nunique())

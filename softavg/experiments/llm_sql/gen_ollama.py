#!/usr/bin/env python3
"""Local-model arm: send every prompt of prompts_v1.json to a local ollama model.

Per prompt: 5 samples at the model's default sampling settings (no temperature
passed; seed = sample index so the run is reproducible) + 1 sample at
temperature 0. Raw replies are stored verbatim in outputs/<model>.jsonl
(':' in the model tag becomes '-'). Re-running skips (prompt_id, sample_idx)
pairs that are already in the file.

Usage: python3 gen_ollama.py qwen2.5:7b [--host http://127.0.0.1:11434]
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts_v1.json"
OUT = HERE / "outputs"
N_DEFAULT = 5          # samples at default temperature (sample_idx 0..4)
IDX_T0 = 5             # sample_idx of the temperature-0 sample
NUM_CTX = 16384        # room for qwen3's thinking trace
NUM_PREDICT = 12000

# Effective default temperature (ollama 0.17.5): qwen2.5 modelfiles set no
# sampling parameters -> ollama's built-in default 0.8; qwen3:14b's modelfile sets 0.6.
DEFAULT_T = {"qwen2.5:14b": 0.8, "qwen2.5:7b": 0.8, "qwen3:14b": 0.6}


def extract_sql(text):
    """Last ```sql block; else last fenced block; else the whole reply."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```sql[ \t]*\n(.*?)```", text, flags=re.S | re.I)
    if not blocks:
        blocks = re.findall(r"```[a-zA-Z]*[ \t]*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip()


def chat(host, model, prompt, options):
    body = dict(model=model, messages=[{"role": "user", "content": prompt}], stream=False,
                options=options, keep_alive="10m")
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    OUT.mkdir(exist_ok=True)
    path = OUT / (a.model.replace(":", "-") + ".jsonl")
    done = set()
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["prompt_id"], r["sample_idx"]))
    with urllib.request.urlopen(a.host + "/api/version") as r:
        ollama_version = json.loads(r.read())["version"]
    with urllib.request.urlopen(urllib.request.Request(a.host + "/api/show", data=json.dumps({"model": a.model}).encode())) as r:
        show = json.loads(r.read())
    digest = None
    with urllib.request.urlopen(a.host + "/api/tags") as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == a.model:
                digest = m["digest"]
    for p in prompts:
        for idx in range(N_DEFAULT + 1):
            if (p["id"], idx) in done:
                continue
            opts = dict(num_ctx=NUM_CTX, num_predict=NUM_PREDICT, seed=idx)
            if idx == IDX_T0:
                opts["temperature"] = 0.0
            t0 = time.time()
            res = chat(a.host, a.model, p["prompt"], opts)
            msg = res.get("message", {})
            raw = msg.get("content", "")
            rec = dict(model=a.model, prompt_id=p["id"], task=p["task"], engine=p["engine"],
                       sample_idx=idx,
                       temperature=(0.0 if idx == IDX_T0 else "default"),
                       temperature_effective=(0.0 if idx == IDX_T0 else DEFAULT_T.get(a.model)),
                       seed=idx, raw_text=raw, thinking=msg.get("thinking"),
                       extracted_sql=extract_sql(raw),
                       done_reason=res.get("done_reason"), eval_count=res.get("eval_count"),
                       prompt_eval_count=res.get("prompt_eval_count"),
                       wall_s=round(time.time() - t0, 1),
                       ollama_version=ollama_version, model_digest=digest,
                       modelfile_parameters=show.get("parameters"),
                       options=opts, prompts_version=1,
                       at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(p["id"], idx, rec["done_reason"], rec["eval_count"], rec["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()

"""Summarizes bench/run_bench.ps1 results (JSON lines) into markdown tables comparing arcinferzero with llama.cpp."""
import json
import sys
from collections import defaultdict

ENGINES = ["arcinferzero", "vulkan", "sycl"]
LABEL = {"arcinferzero": "arcinferzero", "vulkan": "llama.cpp Vulkan", "sycl": "llama.cpp SYCL"}


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def best(rows, model, engine, test, key, prompt=None, higher=True):
    vals = [r[key] for r in rows if r["model"] == model and r["engine"] == engine and r["test"] == test
            and (prompt is None or r.get("prompt") == prompt) and r.get(key)]
    if not vals:
        return None
    return max(vals) if higher else min(vals)


def fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def ratio(a, b):
    return "–" if a is None or b is None or b == 0 else f"{a / b:.2f}×"


def main():
    rows = load(sys.argv[1:])
    models = sorted({r["model"] for r in rows}, key=lambda m: [r["model"] for r in rows].index(m))
    out = []
    out.append("## Throughput (tokens/s, higher is better)\n")
    out.append("| Model | Test | arcinferzero | llama.cpp Vulkan | llama.cpp SYCL | vs Vulkan | vs SYCL |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for m in models:
        for t in ["pp512", "pp2048", "tg64"]:
            v = {e: best(rows, m, e, t, "tps") for e in ENGINES}
            if all(x is None for x in v.values()):
                continue
            out.append(f"| {m} | {t} | {fmt(v['arcinferzero'])} | {fmt(v['vulkan'])} | {fmt(v['sycl'])} | "
                       f"{ratio(v['arcinferzero'], v['vulkan'])} | {ratio(v['arcinferzero'], v['sycl'])} |")
    out.append("\n## Time to first token on real code prompts (seconds, lower is better)\n")
    out.append("| Model | Prompt | Tokens | arcinferzero | llama.cpp Vulkan | llama.cpp SYCL | speed-up vs Vulkan | vs SYCL | "
               "gen t/s at that context (arc / Vulkan / SYCL) |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for m in models:
        prompts = sorted({r.get("prompt") for r in rows if r["model"] == m and r["test"] == "ttft"})
        for p in prompts:
            v = {e: best(rows, m, e, "ttft", "ttft_s", p, higher=False) for e in ENGINES}
            g = {e: best(rows, m, e, "ttft", "tg_tps", p) for e in ENGINES}
            n = next((r["n_prompt"] for r in rows if r["model"] == m and r.get("prompt") == p), "")
            out.append(f"| {m} | {p} | {n} | {fmt(v['arcinferzero'])} | {fmt(v['vulkan'])} | {fmt(v['sycl'])} | "
                       f"{ratio(v['vulkan'], v['arcinferzero'])} | {ratio(v['sycl'], v['arcinferzero'])} | "
                       f"{fmt(g['arcinferzero'], 2)} / {fmt(g['vulkan'], 2)} / {fmt(g['sycl'], 2)} |")
    out.append("\n## Perplexity (ctx 512, lower is better)\n")
    out.append("| Model | arcinferzero | llama.cpp Vulkan | llama.cpp SYCL |")
    out.append("|---|---:|---:|---:|")
    for m in models:
        v = {e: best(rows, m, e, "ppl", "ppl", higher=False) for e in ENGINES}
        if all(x is None for x in v.values()):
            continue
        out.append(f"| {m} | {fmt(v['arcinferzero'], 4)} | {fmt(v['vulkan'], 4)} | {fmt(v['sycl'], 4)} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()

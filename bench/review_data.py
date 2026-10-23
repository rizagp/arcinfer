#!/usr/bin/env python3
# Writes the llama.cpp comparison tables of docs/review.md (the block between <!-- DATA-BEGIN --> and
# <!-- DATA-END -->) from bench/results/results.jsonl: per model / engine / test / prompt the best of the repeated
# runs (highest tok/s, shortest time to first token), as the review's method says.
# usage: python bench/review_data.py [results.jsonl] [review.md]
import json, os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "bench", "results", "results.jsonl")
page = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "docs", "review.md")

best = {}
for line in open(res, encoding="utf-8-sig"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    key = (r.get("model"), r.get("engine"), r.get("test"), r.get("prompt"))
    old = best.get(key)
    if old is None:
        best[key] = r
    elif "ttft_s" in r:
        if r["ttft_s"] < old["ttft_s"]:
            best[key] = r
    elif "tps" in r:
        if r["tps"] > old["tps"]:
            best[key] = r
    elif r.get("time", "") >= old.get("time", ""):
        best[key] = r


def get(model, engine, test, prompt=None):
    return best.get((model, engine, test, prompt))


ENGINES = [("arcinferzero", "arcinferzero"), ("vulkan", "llama.cpp Vulkan"), ("sycl", "llama.cpp SYCL")]
MODELS = [("4b", "Qwen3.5 4B"), ("9b", "Qwen3.5 9B"), ("27b", "Qwen3.8 27B"), ("g12", "Gemma 4 12B"),
          ("35b", "Qwen3.6 35B-A3B (GSQ 2-bit)")]
PROMPTS = [("code-11k", "code 11K"), ("code-34k", "code 34K")]
NOT_RUN = "not run"


def num(v, d):
    return f"{v:,.{d}f}"


def cell(r, field, d):
    return NOT_RUN if r is None or r.get(field) is None else num(float(r[field]), d)


out = []
date = max(r.get("time", "")[:10] for r in best.values())
out.append(f"Measured {date} with the OpenCL engine; best of the repeated runs per engine. llama.cpp cannot load the "
           "GSQ 2-bit Qwen3.6 35B-A3B, so its row has arcinferzero only.")
out.append("")

out.append("Time to first token (seconds) on the code prompts, and generation speed right after the prompt (tok/s):")
out.append("")
out.append("| Model | Prompt | Tokens | " + " | ".join(f"{n} TTFT" for _, n in ENGINES) + " | "
           + " | ".join(f"{n} gen" for _, n in ENGINES) + " |")
out.append("|---|---|---:|" + "---:|" * (2 * len(ENGINES)))
for m, mn in MODELS:
    for p, pn in PROMPTS:
        rs = [get(m, e, "ttft", p) for e, _ in ENGINES]
        n = next((r.get("n_prompt") for r in rs if r and r.get("n_prompt")), None)
        if n is None:
            n = {"code-11k": 11240, "code-34k": 33937}[p] if m != "g12" else {"code-11k": 13921, "code-34k": 35957}[p]
        out.append(f"| {mn} | {pn} | {n:,} | " + " | ".join(cell(r, "ttft_s", 1) for r in rs) + " | "
                   + " | ".join(cell(r, "tg_tps", 2) for r in rs) + " |")
out.append("")

out.append("Synthetic throughput (llama-bench style; tok/s): prompt processing over 512 and 2,048 tokens, generation "
           "of 64 tokens after a short prompt:")
out.append("")
out.append("| Model | " + " | ".join(f"{n} pp512" for _, n in ENGINES) + " | "
           + " | ".join(f"{n} pp2048" for _, n in ENGINES) + " | " + " | ".join(f"{n} tg64" for _, n in ENGINES) + " |")
out.append("|---|" + "---:|" * (3 * len(ENGINES)))
for m, mn in MODELS:
    row = []
    for t in ("pp512", "pp2048", "tg64"):
        row += [cell(get(m, e, t), "tps", 1) for e, _ in ENGINES]
    out.append(f"| {mn} | " + " | ".join(row) + " |")
out.append("")

out.append("Perplexity (40 chunks of 512 tokens of llama.cpp documentation, second half of each chunk scored; lower is "
           "better). Gemma 4's raw-text perplexity is not a usable accuracy metric on this corpus (see Accuracy); the "
           "35B row is the GSQ file, which llama.cpp cannot load:")
out.append("")
out.append("| Model | " + " | ".join(n for _, n in ENGINES) + " |")
out.append("|---|" + "---:|" * len(ENGINES))
for m, mn in MODELS:
    out.append(f"| {mn} | " + " | ".join(cell(get(m, e, "ppl"), "ppl", 4) for e, _ in ENGINES) + " |")

# speculative decoding runs of the benchmark script, when there are any
spec = [(m, mn, get(m, "arcinferzero", "mtp", "code-11k")) for m, mn in MODELS + [("e4b", "Gemma 4 E4B")]]
spec = [s for s in spec if s[2]]
if spec:
    out.append("")
    out.append("Speculative decoding after the 11K code prompt (tok/s):")
    out.append("")
    out.append("| Model | plain | drafting | drafts accepted | tokens per pass |")
    out.append("|---|---:|---:|---:|---:|")
    for m, mn, r in spec:
        out.append(f"| {mn} | {cell(r, 'plain_tps', 2)} | {cell(r, 'mtp_tps', 2)} | {cell(r, 'accept_pct', 1)} % "
                   f"| {cell(r, 'tokens_per_pass', 2)} |")

# one task on every model (bench/prompt_bench.py): the latest run per model, linked to labelled copies of the pages
land_dir = os.path.join(root, "bench", "results", "landing")
pub_dir = os.path.join(os.path.dirname(page), "landing")
names = {"27b": "Qwen3.8 27B", "35b": "Qwen3.6 35B-A3B", "9b": "Qwen3.5 9B", "4b": "Qwen3.5 4B",
         "g12": "Gemma 4 12B", "e4b": "Gemma 4 E4B"}
if os.path.exists(os.path.join(land_dir, "summary.jsonl")):
    runs = {}
    for line in open(os.path.join(land_dir, "summary.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            runs[r["model"]] = r
    rows = []
    for m in ("27b", "35b", "9b", "4b", "g12", "e4b"):
        r = runs.get(m)
        if not r:
            continue
        src = os.path.join(land_dir, f"{m}.html")
        link = names[m]
        if os.path.exists(src) and r.get("html_chars"):
            os.makedirs(pub_dir, exist_ok=True)
            html = open(src, encoding="utf-8").read()
            label = (f'<div style="position:fixed;left:8px;bottom:8px;z-index:2147483647;font:12px/1.4 system-ui,sans-serif;'
                     f'background:#111;color:#fff;padding:6px 10px;border-radius:6px;opacity:.85">'
                     f'Output of {names[m]} on arcinferzero (benchmark prompt, fictional shop)</div>')
            html = (re.sub(r"(<body[^>]*>)", lambda mm: mm.group(1) + label, html, count=1, flags=re.I)
                    if re.search(r"<body", html, re.I) else label + html)
            open(os.path.join(pub_dir, f"{m}.html"), "w", encoding="utf-8", newline="\n").write(html)
            link = f"[{names[m]}](landing/{m}.html)"
        f = lambda k, d: NOT_RUN if r.get(k) is None else num(float(r[k]), d)
        rows.append(f"| {link} | {f('ttft_s', 1)} | {f('total_s', 1)} | {f('completion_tokens', 0)} | {f('gen_tps', 1)} |")
    if rows:
        out.append("")
        out.append("One task on every model through the server (bench/prompt_bench.py):")
        out.append("")
        out.append("| Model | first token (s) | total (s) | tokens | tok/s |")
        out.append("|---|---:|---:|---:|---:|")
        out += rows

blob = "\n".join(out)
s = open(page, encoding="utf-8").read()
s2, n = re.subn(r"<!-- DATA-BEGIN -->.*?<!-- DATA-END -->",
                lambda _: "<!-- DATA-BEGIN -->\n" + blob + "\n<!-- DATA-END -->", s, flags=re.S)
assert n == 1, "data markers not found"
open(page, "w", encoding="utf-8", newline="\n").write(s2)
print(f"filled {page} ({len(out)} lines of tables)")

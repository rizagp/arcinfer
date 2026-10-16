#!/usr/bin/env python3
# Fills the DATA block of docs/review.html from bench/results/results.jsonl (the latest measurement of every
# model / engine / test / prompt), plus the accuracy and memory figures of docs/review_notes.md.
# usage: python bench/review_data.py [results.jsonl] [review.html]
import json, os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "bench", "results", "results.jsonl")
page = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "docs", "review.html")

latest = {}
for line in open(res, encoding="utf-8-sig"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    key = (r.get("model"), r.get("engine"), r.get("test"), r.get("prompt"))
    if key not in latest or r.get("time", "") >= latest[key].get("time", ""):
        latest[key] = r


def get(model, engine, test, prompt=None):
    return latest.get((model, engine, test, prompt))


def rnd(v, d):
    return None if v is None else round(float(v), d)


ENGINES = ["arcinfer", "vulkan", "sycl"]
MODELS = [("4b", "Qwen3.5 4B"), ("9b", "Qwen3.5 9B"), ("27b", "Qwen3.8 27B"), ("g12", "Gemma 4 12B"),
          ("35b", "Qwen3.6 35B-A3B · llama.cpp on UD-IQ2_M")]
PROMPTS = [("code-11k", {"en": "code 11K", "id": "kode 11K"}), ("code-34k", {"en": "code 34K", "id": "kode 34K"})]

data = {
    "date": max(r.get("time", "")[:10] for r in latest.values()),
    "engines": [{"id": "arcinfer", "name": "arcinfer", "color": "var(--s1)"},
                {"id": "vulkan", "name": "llama.cpp Vulkan", "color": "var(--s2)"},
                {"id": "sycl", "name": "llama.cpp SYCL", "color": "var(--s3)"}],
    "models": [{"id": m, "name": n, **({"skip": ["sycl"]} if m == "35b" else {})} for m, n in MODELS],
    "prompts": [{"id": p, "name": n} for p, n in PROMPTS],
    "ttft": {}, "pp512": {}, "pp2048": {}, "tg64": {}, "ppl": {},
}
for m, _ in MODELS + [("35b-iq2m", "")]:
    data["ttft"][m] = {}
    for p, _ in PROMPTS:
        data["ttft"][m][p] = {}
        for e in ENGINES:
            r = get(m, e, "ttft", p)
            if r:
                data["ttft"][m][p][e] = {"s": rnd(r["ttft_s"], 2), "tg": rnd(r.get("tg_tps"), 2), "n": r.get("n_prompt")}
    for k in ("pp512", "pp2048", "tg64"):
        data[k][m] = {e: rnd(get(m, e, k)["tps"], 1) for e in ENGINES if get(m, e, k)}
    data["ppl"][m] = {e: rnd(get(m, e, "ppl")["ppl"], 4) for e in ENGINES if get(m, e, "ppl")}
data["pplModels"] = [{"id": m, "name": n} for m, n in MODELS if m != "35b"]  # 35B: two different quantizations


def moe(model, engine):
    t11 = data["ttft"][model]["code-11k"].get(engine)
    t34 = data["ttft"][model]["code-34k"].get(engine)
    return {"pp512": data["pp512"][model].get(engine), "pp2048": data["pp2048"][model].get(engine),
            "tg64": data["tg64"][model].get(engine), "ttft11": t11, "ttft34": t34, "ppl": data["ppl"][model].get(engine)}


data["moe"] = {"arcinfer": moe("35b", "arcinfer"), "ref": moe("35b-iq2m", "vulkan")}
# the 35B in the charts: arcinfer on the GSQ 2-bit model, llama.cpp Vulkan on Unsloth's UD-IQ2_M (llama.cpp cannot
# load GSQ); the facet name says so
for p, _ in PROMPTS:
    if "vulkan" in data["ttft"]["35b-iq2m"][p]:
        data["ttft"]["35b"][p]["vulkan"] = data["ttft"]["35b-iq2m"][p]["vulkan"]
for k in ("pp512", "pp2048", "tg64"):
    if "vulkan" in data[k]["35b-iq2m"]:
        data[k]["35b"]["vulkan"] = data[k]["35b-iq2m"]["vulkan"]
for k in ("ttft", "pp512", "pp2048", "tg64", "ppl"):
    data[k].pop("35b-iq2m", None)

names = {"27b": "Qwen3.8 27B", "9b": "Qwen3.5 9B", "4b": "Qwen3.5 4B", "g12": "Gemma 4 12B", "e4b": "Gemma 4 E4B"}
data["spec"] = []
for m in ("27b", "9b", "4b", "g12", "e4b"):
    r = get(m, "arcinfer", "mtp", "code-11k")
    if r:
        data["spec"].append({"model": m, "name": names[m], "kind": "mtp" if m in ("27b", "9b", "4b") else "assistant",
                             "plain": rnd(r["plain_tps"], 2), "spec": rnd(r["mtp_tps"], 2),
                             "accept": rnd(r["accept_pct"], 1), "tpp": rnd(r["tokens_per_pass"], 2)})
data["spec_ctx"] = "code-11k"
if not data["spec"]:
    # the 11K-context drafting runs of the final benchmark did not take place (it was stopped); these are the
    # 10 October runs after the decode work: 128 tokens after a short story prompt (12B: bin/run.exe plain and
    # --mtp; E4B: plain with bin/run.exe, drafting through the server, 200 tokens, 80 of 198 drafts accepted)
    data["spec_ctx"] = "short"
    data["spec"] = [
        {"model": "g12", "name": names["g12"], "kind": "assistant", "plain": 7.66, "spec": 8.74, "accept": 50.0, "tpp": 1.94},
        # E4B: 12 October, after the IQ4_XS GEMV / fused FFN work: 128 tokens after the 160-token code prompt
        # (bench/prompts/gemma-code160), bin/run.exe plain and --mtp (67 of 115 drafts, 61 passes)
        {"model": "e4b", "name": names["e4b"], "kind": "assistant", "plain": 21.2, "spec": 30.7, "accept": 58.3, "tpp": 2.10},
    ]

# accuracy (tests/run_tests.ps1 and docs/review_notes.md: KL vs the f32 reference, mean over the prompt positions)
data["kl"] = [
    {"name": "Qwen3.5 4B", "prompt": {"en": "chat, 20 positions", "id": "chat, 20 posisi"}, "kl": 0.0021,
     "note": {"en": "int8 activations, attention and KV cache (0.0006 with f32 attention)",
              "id": "aktivasi, attention dan cache KV int8 (0,0006 dengan attention f32)"}},
    {"name": "Qwen3.6 35B-A3B", "prompt": {"en": "chat, 20 positions", "id": "chat, 20 posisi"}, "kl": 0.0008,
     "note": {"en": "the reference decodes the 2-bit expert layout on its own",
              "id": "referensi men-decode layout expert 2-bit secara terpisah"}},
    {"name": "Gemma 4 12B", "prompt": {"en": "chat, 20 positions", "id": "chat, 20 posisi"}, "kl": 0.029,
     "note": {"en": "0.032 with one V scale per row", "id": "0,032 dengan satu skala V per baris"}},
    {"name": "Gemma 4 12B", "prompt": {"en": "code, 160 positions", "id": "kode, 160 posisi"}, "kl": 0.018,
     "note": {"en": "0.23 with one V scale per row, including a flipped prediction (KL 27 at one position)",
              "id": "0,23 dengan satu skala V per baris, termasuk satu prediksi yang berbalik (KL 27 di satu posisi)"}},
    {"name": "Gemma 4 12B", "prompt": {"en": "raw text, 120 positions", "id": "teks mentah, 120 posisi"}, "kl": 0.035,
     "note": {"en": "0.080 with one V scale per row", "id": "0,080 dengan satu skala V per baris"}},
    {"name": "Gemma 4 12B", "prompt": {"en": "code, 160 positions, token by token", "id": "kode, 160 posisi, token demi token"},
     "kl": 0.013,
     "note": {"en": "the generation path (run --kl-decode) after the 10 October decode work; 0.020 before it. Runs differ by "
                    "float rounding at a few near-tied tokens",
              "id": "jalur generasi (run --kl-decode) setelah perbaikan decode 10 Oktober; 0,020 sebelumnya. Selisih antar-run "
                    "berasal dari pembulatan float di beberapa token yang hampir seri"}},
]
data["mem"] = [
    {"name": "Qwen3.8 27B", "ctx": 49152, "gib": 16.0, "note": {"en": "was 40,960 before the segmented prefill", "id": "sebelumnya 40.960, sebelum prefill bersegmen"}},
    {"name": "Qwen3.6 35B-A3B", "ctx": 262144, "gib": 14.8, "note": {"en": "trained maximum", "id": "batas latih"}},
    {"name": "Qwen3.5 9B", "ctx": 262144, "gib": 12.4, "note": {"en": "trained maximum", "id": "batas latih"}},
    {"name": "Qwen3.5 4B", "ctx": 262144, "gib": 11.9, "note": {"en": "trained maximum", "id": "batas latih"}},
    {"name": "Gemma 4 12B", "ctx": 262144, "gib": 12.7, "note": {"en": "trained maximum", "id": "batas latih"}},
    {"name": "Gemma 4 E4B", "ctx": 131072, "gib": 10.8, "note": {"en": "trained maximum", "id": "batas latih"}},
]

# one task on every model (bench/prompt_bench.py): the latest run per model, and labelled copies of the pages
land_dir = os.path.join(root, "bench", "results", "landing")
pub_dir = os.path.join(os.path.dirname(page), "landing")
data["landing"] = []
if os.path.exists(os.path.join(land_dir, "summary.jsonl")):
    runs = {}
    for line in open(os.path.join(land_dir, "summary.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            runs[r["model"]] = r
    for m in ("27b", "35b", "9b", "4b", "g12", "e4b"):
        r = runs.get(m)
        if not r:
            continue
        src = os.path.join(land_dir, f"{m}.html")
        file = None
        if os.path.exists(src) and r.get("html_chars"):
            os.makedirs(pub_dir, exist_ok=True)
            html = open(src, encoding="utf-8").read()
            label = (f'<div style="position:fixed;left:8px;bottom:8px;z-index:2147483647;font:12px/1.4 system-ui,sans-serif;'
                     f'background:#111;color:#fff;padding:6px 10px;border-radius:6px;opacity:.85">'
                     f'Output of {names[m]} on arcinfer (benchmark prompt, fictional shop)</div>')
            html = re.sub(r"(<body[^>]*>)", lambda mm: mm.group(1) + label, html, count=1, flags=re.I) if re.search(r"<body", html, re.I) else label + html
            open(os.path.join(pub_dir, f"{m}.html"), "w", encoding="utf-8").write(html)
            file = f"landing/{m}.html"
        data["landing"].append({"model": m, "name": names[m], "ttft": r.get("ttft_s"), "answer": r.get("answer_start_s"),
                                "total": r.get("total_s"), "tokens": r.get("completion_tokens"), "tps": r.get("gen_tps"),
                                "drafts": r.get("draft_n"), "accepted": r.get("draft_accepted"), "file": file})

s = open(page, encoding="utf-8").read()
blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
s2, n = re.subn(r"/\*DATA-BEGIN\*/.*?/\*DATA-END\*/", lambda _: "/*DATA-BEGIN*/" + blob + "/*DATA-END*/", s, flags=re.S)
assert n == 1, "data markers not found"
open(page, "w", encoding="utf-8", newline="\n").write(s2)
missing = [f"{m}/{p}/{e}" for m, _ in MODELS for p, _ in PROMPTS for e in ENGINES
           if e not in data["ttft"][m][p] and not (m == "35b" and e == "sycl")]
print(f"filled {page} ({len(blob)} bytes of data); missing TTFT: {', '.join(missing) or 'none'}")

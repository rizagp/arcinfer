#!/usr/bin/env python3
# One real task on every model: sends the same chat prompt through arc-server (each model with its own sampling
# defaults, reasoning off, a fixed seed), streams the answer and records time to first token, answer tokens,
# generation speed and draft acceptance. Saves the raw answer and the HTML page it contains.
# usage: python bench/prompt_bench.py [--models 35b,g12,9b,4b,e4b,27b] [--out bench/results/landing]
import argparse, json, os, re, subprocess, sys, time, urllib.request

PROMPT = ("buat html landing page professional toko coklat enak di pondok gede dalam satu file yang ringkas, "
          "interaksi dengan animasi, vanilla css, dan vanilla js dengan struktur mvvm")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="35b,g12,9b,4b,e4b,27b")
ap.add_argument("--out", default=os.path.join(root, "bench", "results", "landing"))
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--max-tokens", type=int, default=6144)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
base = f"http://127.0.0.1:{args.port}"
server_ps1 = os.path.join(root, "scripts", "arc-server.ps1")


def ps(*a):
    return subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", server_ps1, *a],
                          capture_output=True, text=True)


def extract_html(text):
    m = re.search(r"```html\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<!DOCTYPE html.*?</html>)", text, re.S | re.I)
    return m.group(1).strip() if m else None


for model in args.models.split(","):
    r = ps("start", "-Model", model, "-Quiet")
    if r.returncode != 0:
        print(f"{model}: server failed to start\n{r.stdout}{r.stderr}")
        continue
    alias = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
    time.sleep(10)  # a short idle period before the measurement
    body = {"model": alias, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": args.max_tokens,
            "seed": 1234, "reasoning_effort": "none", "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_answer = None
    reasoning, content, timings, usage, finish = [], [], {}, {}, None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    t_first = t_first or time.time()
                if d.get("content"):
                    content.append(d["content"])
                    t_first = t_first or time.time()
                    t_answer = t_answer or time.time()
                finish = ch.get("finish_reason") or finish
    t1 = time.time()
    text, think = "".join(content), "".join(reasoning)
    page = extract_html(text)
    open(os.path.join(args.out, f"{model}.md"), "w", encoding="utf-8").write(
        (f"<!-- reasoning -->\n{think}\n<!-- answer -->\n" if think else "") + text)
    if page:
        open(os.path.join(args.out, f"{model}.html"), "w", encoding="utf-8").write(page)
    rec = {"model": model, "alias": alias, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "finish": finish,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
           "ttft_s": round((t_first or t1) - t0, 2), "answer_start_s": round((t_answer or t1) - t0, 2),
           "total_s": round(t1 - t0, 1), "gen_tps": round(timings.get("predicted_per_second", 0), 2),
           "draft_n": timings.get("draft_n"), "draft_accepted": timings.get("draft_n_accepted"),
           "reasoning_chars": len(think), "answer_chars": len(text), "html_chars": len(page) if page else 0}
    with open(os.path.join(args.out, "summary.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
ps("stop")

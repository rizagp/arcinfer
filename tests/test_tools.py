"""Exercises arcinferzero-server's OpenAI API: streaming + reasoning + tool calls + multi-turn prefix reuse."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from the workspace",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"},
                                                          "max_lines": {"type": "integer"}}, "required": ["path"]},
    },
}]


def stream_chat(messages, **extra):
    body = {"model": "x", "messages": messages, "tools": TOOLS, "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 1024, "temperature": 0.6}
    body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    reasoning, content, calls, finish, usage, timings = "", "", [], None, None, None
    tool_events = 0
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        buf = b""
        for chunk in iter(lambda: r.read(1), b""):
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                ev = ev.decode().strip()
                if not ev.startswith("data:"):
                    continue
                data = ev[5:].strip()
                if data == "[DONE]":
                    continue
                j = json.loads(data)
                if j.get("usage"):
                    usage = j["usage"]
                if j.get("timings"):
                    timings = j["timings"]
                for ch in j.get("choices", []):
                    d = ch.get("delta", {})
                    reasoning += d.get("reasoning_content") or ""
                    content += d.get("content") or ""
                    # tool calls arrive in pieces (OpenAI streaming): id and name first, then the arguments JSON
                    for tc in d.get("tool_calls") or []:
                        tool_events += 1
                        k = tc.get("index", len(calls))
                        while len(calls) <= k:
                            calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            calls[k]["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            calls[k]["function"]["name"] = fn["name"]
                        calls[k]["function"]["arguments"] += fn.get("arguments") or ""
                    finish = ch.get("finish_reason") or finish
    if calls:
        print(f"  tool call deltas: {tool_events}")
    return reasoning, content, calls, finish, usage, timings, time.time() - t0


msgs = [{"role": "user", "content": "Tolong baca file README.md lalu ringkas isinya."}]
r, c, calls, fin, usage, tim, dt = stream_chat(msgs)
print(f"turn 1: finish={fin} reasoning={len(r)} chars content={c[:80]!r} calls={json.dumps(calls)} "
      f"usage={usage} {dt:.1f}s")
assert fin == "tool_calls" and calls, "expected a tool call"
args = json.loads(calls[0]["function"]["arguments"])
print("  parsed arguments:", args)
msgs.append({"role": "assistant", "content": c, "reasoning_content": r, "tool_calls": calls})
msgs.append({"role": "tool", "tool_call_id": calls[0]["id"],
             "content": "# Supreme Search\nMesin pencari dokumen internal berbasis FastAPI dan SQLite FTS5.\n"
                        "Fitur: indeks PDF, pencarian full-text, ranking BM25, API REST."})
r2, c2, calls2, fin2, usage2, tim2, dt2 = stream_chat(msgs)
print(f"turn 2: finish={fin2} content={c2[:160]!r} usage={usage2} cache_n={tim2 and tim2['cache_n']} {dt2:.1f}s")

"""Compares minja rendering in arcinferzero (chat_render.exe --jinja) with jinja2 for GGUF chat templates that arcinferzero
renders through minja (Gemma 4). usage: python test_chat_jinja.py chat_render.exe template.jinja [template ...]"""
import json
import os
import subprocess
import sys
import tempfile

import jinja2

exe, templates = sys.argv[1], sys.argv[2:]


def raise_exception(msg):
    raise RuntimeError(msg)


env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)
env.globals["raise_exception"] = raise_exception

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file",
                                      "parameters": {"type": "object",
                                                     "properties": {"path": {"type": "string", "description": "File path"},
                                                                    "offset": {"type": "integer"},
                                                                    "mode": {"type": "string", "enum": ["text", "bytes"]},
                                                                    "lines": {"type": "array", "items": {"type": "integer"}}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                                                                                      "timeout": {"type": "number", "nullable": True}}}}},
]

CASES = {
    "simple": {"messages": [{"role": "user", "content": "Halo, apa kabar?"}]},
    "system": {"messages": [{"role": "system", "content": "Kamu asisten yang ringkas."},
                            {"role": "user", "content": "Sebutkan 3 buah."}]},
    "thinking": {"messages": [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "2+2?"}],
                 "enable_thinking": True},
    "no_thinking": {"messages": [{"role": "user", "content": "Hi"}], "enable_thinking": False},
    "multiturn": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "<|channel>thought\nadd<channel|>2"},
        {"role": "user", "content": "kali 3?"}]},
    "tools": {"tools": TOOLS, "enable_thinking": True, "messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Baca README.md"},
        {"role": "assistant", "content": "", "reasoning_content": "I should read the file.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"path": "README.md", "offset": 3}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "# Title\ntext"},
        {"role": "assistant", "content": "Let me also list files.",
         "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls -la"}}},
                        {"id": "c3", "type": "function", "function": {"name": "read", "arguments": {"path": "a.py", "lines": [1, 2]}}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "a.py\nb.py"},
        {"role": "tool", "tool_call_id": "c3", "content": "print(1)"}]},
    "tool_pending": {"tools": TOOLS, "messages": [
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "tool_call_id": "x", "content": "a\nb"}]},
    "content_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part A. "}, {"type": "text", "text": "Part B."}]}]},
    "image": {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                                        {"type": "text", "text": "Apa isi gambar ini?"}]}]},
}

ok = True
for path in templates:
    tmpl = env.from_string(open(path, encoding="utf-8").read())
    for name, case in CASES.items():
        ref = tmpl.render(messages=case["messages"], tools=case.get("tools"), add_generation_prompt=True,
                          enable_thinking=case.get("enable_thinking", False), bos_token="<bos>", eos_token="<eos>")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"messages": case["messages"], "tools": case.get("tools"),
                       "enable_thinking": case.get("enable_thinking", False)}, f, ensure_ascii=False)
            fn = f.name
        try:
            got = subprocess.run([exe, fn, "--jinja", path], capture_output=True).stdout.decode("utf-8")
        finally:
            os.unlink(fn)
        same = got == ref
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {os.path.basename(path)} / {name}")
        if not same:
            i = next((k for k in range(min(len(got), len(ref))) if got[k] != ref[k]), min(len(got), len(ref)))
            print("   first difference at", i)
            print("   jinja2:", repr(ref[max(0, i - 60):i + 80]))
            print("   minja :", repr(got[max(0, i - 60):i + 80]))
print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")

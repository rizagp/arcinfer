"""Compares arcinferzero's native chat rendering (chat_render.exe) with the GGUF Jinja templates rendered by jinja2."""
import json
import os
import subprocess
import sys
import tempfile

import jinja2

exe, tmpl_q35, tmpl_q38 = sys.argv[1], sys.argv[2], sys.argv[3]


def tojson(x, indent=None):
    return json.dumps(x, ensure_ascii=False, indent=indent)


def raise_exception(msg):
    raise RuntimeError(msg)


env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                                      "offset": {"type": "integer"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
]

CASES = {
    "simple": {"messages": [{"role": "user", "content": "Halo, apa kabar?"}]},
    "system": {"messages": [{"role": "system", "content": "Kamu asisten yang ringkas."},
                            {"role": "user", "content": "Sebutkan 3 buah."}]},
    "multiturn": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "Simple addition."},
        {"role": "user", "content": "kali 3?"}]},
    "no_think": {"messages": [{"role": "user", "content": "Hi"}], "enable_thinking": False},
    "effort_low": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                   "reasoning_effort": "low"},
    "tools": {"tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Baca README.md"},
        {"role": "assistant", "content": "", "reasoning_content": "I should read the file.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"path": "README.md", "offset": 3}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "# Title\ntext"},
        {"role": "assistant", "content": "Let me also list files.",
         "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls -la"}}},
                        {"id": "c3", "type": "function", "function": {"name": "read", "arguments": {"path": "a.py"}}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "a.py\nb.py"},
        {"role": "tool", "tool_call_id": "c3", "content": "print(1)"}]},
    "content_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part A. "}, {"type": "text", "text": "Part B."}]}]},
}

ok = True
for style, path in (("q35", tmpl_q35), ("q38", tmpl_q38)):
    template = env.from_string(open(path, encoding="utf-8-sig").read())
    for name, case in CASES.items():
        kwargs = dict(messages=case["messages"], tools=case.get("tools"), add_generation_prompt=True)
        if "enable_thinking" in case:
            kwargs["enable_thinking"] = case["enable_thinking"]
        if "reasoning_effort" in case:
            kwargs["reasoning_effort"] = case["reasoning_effort"]
        expected = template.render(**kwargs)
        c = dict(case, q38=(style == "q38"))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)
            fname = f.name
        got = subprocess.run([exe, fname], capture_output=True).stdout.decode("utf-8")
        os.unlink(fname)
        same = got == expected
        ok &= same
        print(f"{style:4} {name:14} {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            i = next((k for k in range(min(len(got), len(expected))) if got[k] != expected[k]), min(len(got), len(expected)))
            print("   expected:", repr(expected[max(0, i - 60):i + 80]))
            print("   got     :", repr(got[max(0, i - 60):i + 80]))
print("ALL IDENTICAL" if ok else "SOME DIFFER")

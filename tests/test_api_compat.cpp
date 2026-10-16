// CPU-only tests of the API helpers: the constrained-JSON prefix checker (tools/json_constraint.h) and the request
// translation of the other API formats (tools/api_compat.h). Build: .\build.ps1 tests\test_api_compat.cpp
#include "../tools/api_compat.h"
#include "../tools/json_constraint.h"
#include "../tools/tool_constraint.h"
#include "../src/chat_gemma4.h"
#include <cstdio>

static int fails = 0;
static void expect(bool ok, const char* what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

int main() {
  using jcon::Checker;
  // ---- json_object
  {
    Checker c(Json::object(), true);
    expect(c.check("") == jcon::PARTIAL, "object: empty is a prefix");
    expect(c.check("  {") == jcon::PARTIAL, "object: open brace");
    expect(c.check("{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\\ny\"}") == jcon::DONE, "object: nested value done");
    expect(c.check("[1]") == jcon::FAIL, "object: array at top level refused");
    expect(c.check("{\"a\": 1}}") == jcon::FAIL, "object: trailing brace refused");
    expect(c.check("{\"a\": 01}") == jcon::FAIL, "object: leading zero refused");
    expect(c.check("{\"a\": tru") == jcon::PARTIAL, "object: literal prefix");
    expect(c.check("{\"a\": trux") == jcon::FAIL, "object: bad literal");
    expect(c.check("{\"a\": \"\\u00") == jcon::PARTIAL, "object: unicode escape prefix");
    expect(c.check("{\"a\": \"\\q\"}") == jcon::FAIL, "object: bad escape");
    expect(c.check("{\"a\":1}            ") == jcon::FAIL, "object: long whitespace after the value refused");
  }
  // ---- schema
  {
    Json sc = Json::parse(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},
      "tags":{"type":"array","items":{"enum":["a","bb"]},"maxItems":2},"kind":{"anyOf":[{"type":"null"},{"const":"x"}]}},
      "required":["name","age"]})");
    Checker c(sc, false);
    expect(c.check("{\"name\": \"Ann\", \"age\": 30}") == jcon::DONE, "schema: required keys");
    expect(c.check("{\"name\": \"Ann\"}") == jcon::FAIL, "schema: missing required key");
    expect(c.check("{\"na") == jcon::PARTIAL, "schema: key prefix");
    expect(c.check("{\"nx") == jcon::FAIL, "schema: unknown key prefix");
    expect(c.check("{\"name\": \"A\", \"name\": \"B\"") == jcon::FAIL, "schema: duplicate key");
    expect(c.check("{\"age\": 3.5") == jcon::FAIL, "schema: integer with a fraction");
    expect(c.check("{\"age\": \"3\"") == jcon::FAIL, "schema: wrong type");
    expect(c.check("{\"name\":\"a\",\"age\":1,\"tags\":[\"b") == jcon::PARTIAL, "schema: enum prefix");
    expect(c.check("{\"name\":\"a\",\"age\":1,\"tags\":[\"c") == jcon::FAIL, "schema: not an enum value");
    expect(c.check("{\"name\":\"a\",\"age\":1,\"tags\":[\"a\",\"bb\",\"a\"") == jcon::FAIL, "schema: maxItems");
    expect(c.check("{\"name\":\"a\",\"age\":1,\"kind\":null}") == jcon::DONE, "schema: anyOf null");
    expect(c.check("{\"name\":\"a\",\"age\":1,\"kind\":\"x\"}") == jcon::DONE, "schema: anyOf const");
    expect(c.check("{\"name\":\"a\",\"age\":1,\"kind\":\"y\"") == jcon::FAIL, "schema: anyOf no branch");
    Json ref = Json::parse(R"({"$defs":{"P":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]}},
      "type":"array","items":{"$ref":"#/$defs/P"}})");
    Checker r(ref, false);
    expect(r.check("[{\"x\": -1.5e3}, {\"x\": 2}]") == jcon::DONE, "schema: $ref items");
    expect(r.check("[{\"y\": 1}]") == jcon::FAIL, "schema: $ref unknown key");
  }
  // ---- request translation
  {
    Json out;
    std::string err;
    Json oll = Json::parse(R"({"model":"qwen3.6-35b-a3b:latest","messages":[{"role":"user","content":"hi","images":["QUJD"]},
      {"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},
      {"role":"tool","content":"42","tool_name":"f"}],"options":{"temperature":0.2,"num_predict":50},"think":false})");
    expect(apic::to_chat("ollama_chat", oll, out, err), "ollama: translated");
    expect(out["model"].str() == "qwen3.6-35b-a3b", "ollama: tag stripped");
    expect(out["messages"][0]["content"][1]["type"].str() == "image_url", "ollama: image part");
    expect(out["messages"][1]["tool_calls"][0]["function"]["arguments"].str() == "{\"a\":1}", "ollama: tool call arguments");
    expect(out["messages"][2]["tool_call_id"].str() == out["messages"][1]["tool_calls"][0]["id"].str(), "ollama: tool result id");
    expect(out["max_tokens"].as_int() == 50 && out["enable_thinking"].is_bool() && out["stream"].as_bool(), "ollama: options");
    Json ant = Json::parse(R"({"model":"m","max_tokens":100,"system":[{"type":"text","text":"S"}],"messages":[
      {"role":"user","content":[{"type":"text","text":"q"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"QUJD"}}]},
      {"role":"assistant","content":[{"type":"text","text":"t"},{"type":"tool_use","id":"tu1","name":"g","input":{"b":2}}]},
      {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"ok"}]}],
      "tools":[{"name":"g","description":"d","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"},
      "thinking":{"type":"enabled","budget_tokens":512},"stop_sequences":["END"]})");
    expect(apic::to_chat("anthropic", ant, out, err), "anthropic: translated");
    expect(out["messages"][0]["role"].str() == "system" && out["messages"][0]["content"].str() == "S", "anthropic: system");
    expect(out["messages"][1]["content"][1]["image_url"]["url"].str().rfind("data:image/jpeg;base64,", 0) == 0, "anthropic: image");
    expect(out["messages"][2]["tool_calls"][0]["id"].str() == "tu1", "anthropic: tool_use");
    expect(out["messages"][3]["role"].str() == "tool" && out["messages"][3]["tool_call_id"].str() == "tu1", "anthropic: tool_result");
    expect(out["tools"][0]["function"]["parameters"]["type"].str() == "object", "anthropic: tools");
    expect(out["tool_choice"].str() == "required" && out["thinking_token_budget"].as_int() == 512, "anthropic: choice, thinking");
    Json rsp = Json::parse(R"({"model":"m","instructions":"I","input":[{"role":"user","content":[{"type":"input_text","text":"a"}]},
      {"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},{"type":"function_call","call_id":"c2","name":"f","arguments":"{}"},
      {"type":"function_call_output","call_id":"c1","output":"r1"}],"tools":[{"type":"function","name":"f","parameters":{}}],
      "text":{"format":{"type":"json_schema","name":"o","schema":{"type":"object"}}},"max_output_tokens":9})");
    expect(apic::to_chat("responses", rsp, out, err), "responses: translated");
    expect(out["messages"][2]["tool_calls"].size() == 2, "responses: consecutive calls merged");
    expect(out["messages"][3]["tool_call_id"].str() == "c1", "responses: call output");
    expect(out["response_format"]["type"].str() == "json_schema" && out["max_tokens"].as_int() == 9, "responses: format, tokens");
    Json cmp = Json::parse(R"({"model":"m","prompt":"def f(","max_tokens":8})");
    expect(apic::to_chat("completions", cmp, out, err) && out["__raw_prompt"].str() == "def f(", "completions: raw prompt");
  }
  // ---- Gemma 4 output split: thought blocks anywhere, tool calls inside a thought, stray closes; the stream (one
  // character at a time) adds up to the whole-output split
  {
    struct Case { const char* name; std::string text, reason, content; bool tool; };
    const std::vector<Case> cases = {
        {"gemma: plain", "Hello there.", "", "Hello there.", false},
        {"gemma: one thought", "<|channel>thought\nplan A<channel|>Answer.", "plan A", "Answer.", false},
        {"gemma: two thoughts", "<|channel>thought\nT1<channel|>Text <|channel>thought\nT2<channel|>More", "T1\n\nT2", "Text More", false},
        {"gemma: stray close", "<channel|>\n\n read file<channel|>", "", "read file", false},
        {"gemma: tool in thought", "<|channel>thought\nI will read<|tool_call>call:read{path:x}<tool_call|>", "I will read", "", true},
        {"gemma: tool after text", "Sure.\n<|tool_call>call:f{}<tool_call|>", "", "Sure.\n", true},
    };
    for (const Case& c : cases) {
      const auto sp = gemma4::OutputParser::split(c.text, true);
      gemma4::OutputParser p;
      std::string r, t, dr, dc;
      for (size_t i = 1; i <= c.text.size(); ++i) {
        p.update(c.text.substr(0, i), i == c.text.size(), dr, dc);
        r += dr;
        t += dc;
      }
      const bool ok = sp.reason == c.reason && sp.content == c.content && sp.tool == c.tool && r == sp.reason && t == sp.content;
      if (!ok) printf("    got reason [%s] content [%s] streamed [%s] [%s]\n", sp.reason.c_str(), sp.content.c_str(), r.c_str(), t.c_str());
      expect(ok, c.name);
    }
  }
  // ---- Gemma 4 after a tool response: the prompt opened the thought block, the output starts inside it
  {
    const std::string text = "Wait, I already did that.\nLet me check.<channel|>\n\nDone.";
    const auto sp = gemma4::OutputParser::split(text, true, true);
    gemma4::OutputParser p(true);
    std::string r, t, dr, dc;
    for (size_t i = 1; i <= text.size(); ++i) {
      p.update(text.substr(0, i), i == text.size(), dr, dc);
      r += dr;
      t += dc;
    }
    expect(sp.reason == "Wait, I already did that.\nLet me check." && sp.content == "Done." && r == sp.reason && t == sp.content,
           "gemma: output starting inside a prefilled thought");
    const auto sp2 = gemma4::OutputParser::split("I will read it<|tool_call>call:read{}<tool_call|>", true, true);
    expect(sp2.reason == "I will read it" && sp2.content.empty() && sp2.tool, "gemma: prefilled thought, then a tool call");
  }
  // ---- Gemma 4 tool calls written with a broken nesting (long edits in Pi): repaired with the tool's schema
  {
    const Json tools = Json::parse(R"([{"type":"function","function":{"name":"edit","parameters":{"type":"object",
      "properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object",
      "properties":{"oldText":{"type":"string"},"newText":{"type":"string"}}}}},"required":["path","edits"]}}}])");
    const std::string Q = "<|\"|>";
    // "]" where "}]" belongs, then path
    const std::string t1 = "<|tool_call>call:edit{edits:[{oldText:" + Q + "a{x}" + Q + ",newText:" + Q + "b, c" + Q +
                           "},{oldText:" + Q + "d" + Q + ",newText:" + Q + "f\"{y}\"" + Q + "],path:" + Q + "x.py" + Q + "}<tool_call|>";
    auto c1 = gemma4::parse_tool_calls(t1, tools);
    expect(c1.size() == 1 && c1[0].arguments["path"].str() == "x.py" && c1[0].arguments["edits"].size() == 2 &&
               !c1[0].arguments["edits"][1].has("]") && c1[0].arguments["edits"][1]["newText"].str() == "f\"{y}\"",
           "gemma call: object closed by ']'");
    // path written inside the last edit
    const std::string t2 = "<|tool_call>call:edit{edits:[{oldText:" + Q + "a" + Q + ",newText:" + Q + "b" + Q + ",path:" + Q +
                           "x.py" + Q + "}]}<tool_call|>";
    auto c2 = gemma4::parse_tool_calls(t2, tools);
    expect(c2.size() == 1 && c2[0].arguments["path"].str() == "x.py" && !c2[0].arguments["edits"][0].has("path"),
           "gemma call: required path moved up");
    // streaming: the arguments come whole when the call is complete, equal to the final parse
    auto pc = gemma4::partial_tool_calls(t2.substr(0, t2.size() - 20), false, tools);
    auto pf = gemma4::partial_tool_calls(t2, false, tools);
    expect(pc.size() == 1 && pc[0].args.empty() && pf.size() == 1 && pf[0].args == c2[0].arguments.dump(),
           "gemma call: streamed arguments whole and repaired");
  }
  // ---- constrained Gemma 4 tool calls (tools/tool_constraint.h), with Pi-like tool schemas
  {
    const Json tools = Json::parse(R"([
      {"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},
        "edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
        "required":["oldText","newText"]}}},"required":["path","edits"]}}},
      {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"},
        "offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
      {"type":"function","function":{"name":"browser_navigate","parameters":{"type":"object","properties":{
        "action":{"type":"string","enum":["back","forward","reload"]}},"required":["action"]}}}])");
    tcon::GemmaCall g(tools);
    const std::string Q = "<|\"|>";
    const std::string ok = "call:edit{path:" + Q + "x.py" + Q + ",edits:[{oldText:" + Q + "a = {1, 2}\n\"q\"" + Q + ",newText:" + Q +
                           "b: [3]," + Q + "},{oldText:" + Q + "c" + Q + ",newText:" + Q + "d" + Q + "}]}<tool_call|>";
    bool prefixes = true;
    for (size_t n = 0; n < ok.size(); ++n) prefixes = prefixes && g.check(ok.substr(0, n)) == jcon::PARTIAL;
    expect(prefixes && g.check(ok) == jcon::DONE, "tool call: a valid edit call, every prefix partial");
    const std::string body = "call:edit{edits:[{oldText:" + Q + "a" + Q + ",newText:" + Q + "b" + Q + "}]";
    expect(g.check(body + "}") == jcon::FAIL && g.check(body + ",") == jcon::PARTIAL && g.check(body + ",pa") == jcon::PARTIAL,
           "tool call: path cannot be left out (only ',' and more keys)");
    expect(g.check("call:edit{edits:[{edits:") == jcon::FAIL, "tool call: edits nested in an edit refused");
    expect(g.check("call:edit{edits:[{32") == jcon::FAIL && g.check("call:edit{edits:[{old") == jcon::PARTIAL,
           "tool call: only the item's keys");
    expect(g.check("call:edit{edits:[{oldText:" + Q + "a" + Q + "}") == jcon::FAIL, "tool call: an edit needs newText");
    expect(g.check("call:edit{path:\"x.py\"") == jcon::FAIL, "tool call: strings only with <|\"|>");
    expect(g.check("call:edit{path:" + Q + "a" + Q + ",path:") == jcon::FAIL, "tool call: a key only once");
    expect(g.check("call:xyz") == jcon::FAIL && g.check("call:re") == jcon::PARTIAL && g.check("call:read") == jcon::PARTIAL &&
               g.check("call:edi{") == jcon::FAIL, "tool call: only the request's tools");
    const std::string rd = "call:read{path:" + Q + "f" + Q + ",offset:10}";
    expect(g.check(rd) == jcon::PARTIAL && g.check(rd + "<tool_") == jcon::PARTIAL && g.check(rd + "<tool_call|>") == jcon::DONE &&
               g.check(rd + "x") == jcon::FAIL && g.check(rd + "<tool_call|>x") == jcon::FAIL, "tool call: ends with <tool_call|>");
    expect(g.check("call:read{path:" + Q + "f" + Q + ",offset:1.5") == jcon::FAIL &&
               g.check("call:read{path:" + Q + "f" + Q + ",offset:" + Q) == jcon::FAIL, "tool call: integer arguments");
    expect(g.check("call:browser_navigate{action:" + Q + "rel") == jcon::PARTIAL &&
               g.check("call:browser_navigate{action:" + Q + "relo" + Q) == jcon::FAIL &&
               g.check("call:browser_navigate{action:" + Q + "reload" + Q + "}<tool_call|>") == jcon::DONE, "tool call: enum values");
    const std::string tx = "I will edit it.<|tool_call>call:ed";
    expect(g.open_call(tx) == tx.size() - 7 && g.open_call(tx + "<tool_call|>") == std::string::npos &&
               g.open_call("no call") == std::string::npos, "tool call: the open call is found");
    // ---- Qwen format (chat.h parse_tool_calls), same tools
    tcon::QwenCall qc(tools);
    const std::string qok = "\n<function=edit>\n<parameter=path>\nx.py\n</parameter>\n<parameter=edits>\n"
                            "[{\"oldText\": \"a = {1, 2}\", \"newText\": \"b\"}]\n</parameter>\n</function>\n</tool_call>";
    bool qpre = true;
    for (size_t n = 0; n < qok.size(); ++n) qpre = qpre && qc.check(qok.substr(0, n)) == jcon::PARTIAL;
    expect(qpre && qc.check(qok) == jcon::DONE, "qwen call: a valid edit call, every prefix partial");
    const std::string qb = "\n<function=edit>\n<parameter=edits>\n[{\"oldText\": \"a\", \"newText\": \"b\"}]\n</parameter>\n";
    expect(qc.check(qb + "</") == jcon::FAIL && qc.check(qb + "<") == jcon::PARTIAL && qc.check(qb + "<parameter=pa") == jcon::PARTIAL,
           "qwen call: path cannot be left out");
    expect(qc.check("\n<function=edit>\n<parameter=edits>\n[{\"edits\"") == jcon::FAIL &&
               qc.check("\n<function=edit>\n<parameter=edits>\n[{\"oldText\": \"a\"}]\n</parameter>") == jcon::FAIL,
           "qwen call: array items follow the schema");
    expect(qc.check("\n<function=read>\n<parameter=offset>\n1") == jcon::PARTIAL &&
               qc.check("\n<function=read>\n<parameter=offset>\nabc") == jcon::FAIL &&
               qc.check("\n<function=read>\n<parameter=path>\nf\n</parameter>\n<parameter=offset>\n10\n</par") == jcon::PARTIAL,
           "qwen call: integer parameters as JSON");
    expect(qc.check("\n<function=xyz>") == jcon::FAIL && qc.check("\n<function=re") == jcon::PARTIAL &&
               qc.check("\n<function=read>\n<parameter=nope>") == jcon::FAIL, "qwen call: only the request's tools and keys");
    expect(qc.check("\n<function=read>\n<parameter=path>\nline 1\n<b>{x}</b> \"q\"\n</parameter>\n</function>\n</tool_call>") == jcon::DONE &&
               qc.check("\n<function=read>\n<parameter=path>\nf <parameter=") == jcon::FAIL, "qwen call: string values are raw text");
    expect(qc.open_call("x<tool_call>\n<fun") == 12 && qc.open_call("x<tool_call>\n</tool_call>") == std::string::npos,
           "qwen call: the open call is found");
  }
  // ---- a comma only where another key can follow (else the answer dead-ends: "{"name": "x", "age": 34," cut off)
  {
    Checker c(Json::parse(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]})"), false);
    expect(c.check("{\"name\": \"x\", \"age\": 34,") == jcon::FAIL && c.check("{\"name\": \"x\",") == jcon::PARTIAL &&
               c.check("{\"name\": \"x\", \"age\": 34}") == jcon::DONE, "json: no comma after the last allowed key");
    const Json tl = Json::parse(R"([{"type":"function","function":{"name":"read","parameters":{"type":"object",
      "properties":{"path":{"type":"string"}},"required":["path"]}}}])");
    tcon::GemmaCall g2(tl);
    const std::string Q2 = "<|\"|>";
    expect(g2.check("call:read{path:" + Q2 + "f" + Q2 + ",") == jcon::FAIL &&
               g2.check("call:read{path:" + Q2 + "f" + Q2 + "}") == jcon::PARTIAL, "tool call: no comma after the last key");
  }
  // ---- Qwen template: earlier turns' reasoning dropped by default, kept with preserve_thinking
  {
    const Json msgs = Json::parse(R"([{"role":"user","content":"q1"},
      {"role":"assistant","content":"a1","reasoning_content":"r1"},{"role":"user","content":"q2"}])");
    ChatOptions co;
    const std::string dropped = chat::render(msgs, Json(), co, false);
    co.preserve_thinking = true;
    const std::string kept = chat::render(msgs, Json(), co, false);
    expect(dropped.find("r1") == std::string::npos && kept.find("<think>\nr1\n</think>\n\na1") != std::string::npos,
           "qwen template: preserve_thinking keeps earlier reasoning");
    // the prompt of the next turn extends the previous one (cache reuse) only with the reasoning kept
    const Json m2 = Json::parse(R"([{"role":"user","content":"q1"},{"role":"assistant","content":"a1","reasoning_content":"r1"}])");
    ChatOptions c2;
    c2.add_generation_prompt = false;
    c2.preserve_thinking = true;
    expect(kept.rfind(chat::render(m2, Json(), c2, false), 0) == 0, "qwen template: preserved prompt is append-only");
  }
  printf("%s (%d failed)\n", fails ? "FAILED" : "all passed", fails);
  return fails ? 1 : 0;
}

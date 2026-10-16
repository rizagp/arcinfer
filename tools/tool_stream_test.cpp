// Checks the streaming of tool calls (chat::partial_tool_calls, gemma4::partial_tool_calls) against the parsers of
// whole responses: for every prefix of a response, each call's streamed arguments must be a prefix of its final
// arguments, end on a whole UTF-8 character and only grow; the final arguments must be valid JSON equal to
// parse_tool_calls' (with a repeated key read as JSON readers do: the last value counts).
// usage: tool_stream_test [fuzz iterations]
#include "../src/chat_gemma4.h"
#include <cstdio>
#include <random>

static int failures = 0;
static void fail(const char* what, const std::string& text, size_t n, const std::string& detail) {
  if (++failures <= 10) fprintf(stderr, "FAIL %s at prefix %zu/%zu: %s\n  text: %s\n", what, n, text.size(), detail.c_str(), text.c_str());
}

// A parsed object with repeated keys rebuilt the way Json::set builds it (first position, last value).
static Json dedupe(const Json& v) {
  if (v.is_array()) {
    Json a = Json::array();
    for (const Json& e : v.items()) a.push(dedupe(e));
    return a;
  }
  if (!v.is_object()) return v;
  Json o = Json::object();
  for (const auto& [k, e] : v.entries()) o.set(k, dedupe(e));
  return o;
}

template <class Partial, class Whole>
static void check(const std::string& text, Partial partial, Whole whole) {
  const std::vector<chat::PartialToolCall> fin = partial(text, true);
  const std::vector<chat::ToolCall> ref = whole(text);
  if (fin.size() != ref.size()) return fail("call count", text, text.size(), std::to_string(fin.size()) + " vs " + std::to_string(ref.size()));
  for (size_t k = 0; k < fin.size(); ++k) {
    if (fin[k].name != ref[k].name) fail("name", text, text.size(), fin[k].name + " vs " + ref[k].name);
    Json parsed;
    if (!Json::try_parse(fin[k].args, parsed)) { fail("final JSON", text, text.size(), fin[k].args); continue; }
    if (dedupe(parsed).dump() != ref[k].arguments.dump()) fail("final arguments", text, text.size(), fin[k].args + " vs " + ref[k].arguments.dump());
  }
  std::vector<chat::PartialToolCall> prev;
  for (size_t n = 0; n <= text.size(); ++n) {
    const std::vector<chat::PartialToolCall> pc = partial(text.substr(0, n), false);
    if (pc.size() > fin.size() || pc.size() < prev.size()) { fail("partial count", text, n, std::to_string(pc.size())); break; }
    for (size_t k = 0; k < pc.size(); ++k) {
      const std::string& a = pc[k].args;
      if (pc[k].name != fin[k].name) fail("partial name", text, n, pc[k].name + " vs " + fin[k].name);
      if (fin[k].args.compare(0, a.size(), a) != 0) fail("not a prefix", text, n, a + "  |  " + fin[k].args);
      if (k < prev.size() && a.size() < prev[k].args.size()) fail("shrank", text, n, a);
      if (chat::utf8_end(a, 0, a.size()) != a.size()) fail("split UTF-8", text, n, a);
    }
    if (failures > 10) return;
    prev = pc;
  }
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 3000;
  const Json tools = Json::parse(R"([
    {"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}}}},
    {"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array"}}}}},
    {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}},
    {"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":["number","null"]}}}}}])");
  auto qp = [&](const std::string& t, bool f) { return chat::partial_tool_calls(t, tools, f); };
  auto qw = [&](const std::string& t) { return chat::parse_tool_calls(t, tools); };
  auto gp = [](const std::string& t, bool f) { return gemma4::partial_tool_calls(t, f); };
  auto gw = [](const std::string& t) { return gemma4::parse_tool_calls(t); };

  const std::vector<std::string> qwen = {
    "I'll write it.\n\n<tool_call>\n<function=write>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\n"
    "def f(x):\n\treturn \"x\\n\" + 'é 中文 😀'\n\n# </function> <parameter </tool </param\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=edit>\n<parameter=path>\nREADME.md\n</parameter>\n<parameter=edits>\n"
    "[{\"oldText\": \"a\\nb\", \"newText\": \"c\"}]\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=read>\n<parameter=path>\nx\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n\n</parameter>\n<parameter=timeout>\nsoon\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=path>\na\n<parameter=content>\nno closing tags\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=content>\ntruncated mid value \xF0\x9F\x98",
    "<tool_call>\n<function=bash>\n<parameter=command>\n\n</parameter>\n<parameter=other>\n{\"k\": [1, 2]}\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write>\n<parameter=path>\na\n</parameter>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\nno function here\n</tool_call>\n<tool_call>\n<function= write >\n<parameter= content >x</parameter></function></tool_call>",
    "text only, no tool call </tool_call> <function=x>",
    // an edits array that is not valid JSON (a raw line break inside a string): streamed, then repeated as a string
    "<tool_call>\n<function=edit>\n<parameter=path>\nR\n</parameter>\n<parameter=edits>\n"
    "[{\"oldText\": \"line1\nline2\", \"newText\": \"x\"}]\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=edit>\n<parameter=edits>\n[{\"oldText\": \"a\", \"newText\": 1.5e3, \"n\": null}, [true, false]]\n</parameter>\n</function>\n</tool_call>",
  };
  const std::string Q = "<|\"|>";
  const std::vector<std::string> gemma = {
    "<|tool_call>call:write{path:" + Q + "src/a.py" + Q + ",content:" + Q + "x = {'a': [1, 2]}\nprint(\"é 中文 😀\") # <| |> , : }" + Q + "}<tool_call|>",
    "<|tool_call>call:edit{path:" + Q + "R.md" + Q + ",edits:[{oldText:" + Q + "a\nb" + Q + ",newText:" + Q + "c" + Q + "},{oldText:" + Q + "d" + Q + ",newText:" + Q + Q + "}]}<tool_call|>",
    "<|tool_call>call:read{path:" + Q + "x" + Q + ", limit: 40, flag: true, none: None, f: 1.50, word: hello world, nested: [1, [2, 3], {a: 4}]}<tool_call|>"
    "<|tool_call>call:bash{\"command\": \"ls \\\"-la\\\"\", \"timeout\": null}<tool_call|>",
    "<|tool_call>call:write{path:" + Q + "a" + Q + ",content:" + Q + "unterminated \xE4\xB8\xAD",
    "<|tool_call>call:noargs<tool_call|><|tool_call>{x:1}<tool_call|><|tool_call> call:w {k:" + Q + "v" + Q + ",k:2}<tool_call|>",
    "<|tool_call>call:x{a:{b:{c:[" + Q + "d" + Q + ",]}},e:\"bad\\q\",}",
    "<|tool_call>call:x{a:[1}<tool_call|>",  // a stray '}' in an array made the parser loop forever
  };
  for (const auto& t : qwen) check(t, qp, qw);
  // array parameters stream while they are written (Pi's edit puts its whole change in `edits`): before "newText"
  // is generated, the first edit's oldText is already in the arguments
  {
    const std::string& t = qwen[1];
    const size_t cut = t.find("\"newText\"");
    const auto pc = qp(t.substr(0, cut), false);
    if (pc.empty() || pc[0].args.find("\"oldText\": \"a\\nb\"") == std::string::npos)
      fail("edits not streamed", t, cut, pc.empty() ? "none" : pc[0].args);
  }
  for (const auto& t : gemma) check(t, gp, gw);

  // Pi's tools (pi-coding-agent 1.1 with pi-unsloth-webtools, pi-browser-control and MCP codemode; parameter types as
  // Pi sends them): one call each, as Qwen writes them. Final arguments must have the schema's types, and long string
  // and array values must stream before their parameter is closed.
  {
    const Json pi = Json::parse(R"([
      {"name":"read","p":{"path":"string","offset":"number","limit":"number"}},
      {"name":"bash","p":{"command":"string","timeout":"number"}},
      {"name":"edit","p":{"path":"string","edits":"array"}},
      {"name":"write","p":{"path":"string","content":"string"}},
      {"name":"web_search","p":{"query":"string","maxResults":"number","timeoutMs":"number"}},
      {"name":"web_fetch","p":{"url":"string","maxChars":"number","timeoutMs":"number"}},
      {"name":"browser_open","p":{"url":"string","headed":"boolean","keep_open":"boolean","width":"number","height":"number"}},
      {"name":"browser_navigate","p":{"action":"string","url":"string","take_screenshot":"boolean"}},
      {"name":"browser_screenshot","p":{"highlight_interactive":"boolean","full_page":"boolean"}},
      {"name":"browser_click","p":{"target":"number","x":"number","y":"number","selector":"string","button":"string","double_click":"boolean","take_screenshot":"boolean"}},
      {"name":"browser_type","p":{"text":"string","target":"number","selector":"string","clear":"boolean","press_enter":"boolean","take_screenshot":"boolean"}},
      {"name":"browser_scroll","p":{"direction":"string","amount":"number","take_screenshot":"boolean"}},
      {"name":"browser_hover","p":{"target":"number","selector":"string","x":"number","y":"number","take_screenshot":"boolean"}},
      {"name":"browser_inspect_element","p":{"target":"number","selector":"string"}},
      {"name":"browser_console_logs","p":{"level":"string","clear":"boolean"}},
      {"name":"browser_close","p":{}},
      {"name":"codemode","p":{"code":"string"}}])");
    Json ptools = Json::array();
    for (const Json& t : pi.items()) {
      Json props = Json::object();
      for (const auto& [k, ty] : t["p"].entries()) {
        Json p = Json::object();
        p.set("type", ty);
        props.set(k, p);
      }
      Json params = Json::object();
      params.set("type", "object");
      params.set("properties", props);
      Json fn = Json::object();
      fn.set("name", t["name"]);
      fn.set("parameters", params);
      Json tool = Json::object();
      tool.set("type", "function");
      tool.set("function", fn);
      ptools.push(tool);
    }
    auto call = [](const std::string& fn, std::initializer_list<std::pair<const char*, std::string>> ps) {
      std::string s = "<tool_call>\n<function=" + fn + ">\n";
      for (const auto& [k, v] : ps) s += std::string("<parameter=") + k + ">\n" + v + "\n</parameter>\n";
      return s + "</function>\n</tool_call>";
    };
    const std::vector<std::string> calls = {
      call("read", {{"path", "src/main.py"}, {"offset", "10"}, {"limit", "40"}}),
      call("bash", {{"command", "grep -rn \"TODO\" src | head -n 20 && echo 'done \\ ok' > out.txt"}, {"timeout", "120"}}),
      call("edit", {{"path", "app/x.py"}, {"edits", "[{\"oldText\": \"x = 1\\n\", \"newText\": \"x = 2  # caf\xC3\xA9 \xE4\xB8\xAD\\n\"}, {\"oldText\": \"a\", \"newText\": \"b\"}]"}}),
      call("write", {{"path", "data/config.json"}, {"content", "{\"name\": \"test\", \"values\": [1, 2, 3]}\n<not a tag> \xF0\x9F\x98\x80"}}),
      call("web_search", {{"query", "intel arc opencl dp4a performance"}, {"maxResults", "5"}}),
      call("web_fetch", {{"url", "https://example.com/a?b=1&c=2"}, {"maxChars", "2000"}, {"timeoutMs", "15000"}}),
      call("browser_open", {{"url", "http://localhost:3000"}, {"headed", "false"}, {"keep_open", "True"}, {"width", "1280"}, {"height", "720"}}),
      call("browser_navigate", {{"action", "goto"}, {"url", "https://example.com/login"}, {"take_screenshot", "true"}}),
      call("browser_screenshot", {{"full_page", "true"}}),
      call("browser_click", {{"target", "12"}, {"button", "left"}, {"double_click", "False"}}),
      call("browser_type", {{"text", "hello \"world\"\nsecond line with </p> and <b>"}, {"selector", "#q"}, {"press_enter", "true"}}),
      call("browser_scroll", {{"direction", "down"}, {"amount", "600"}}),
      call("browser_hover", {{"selector", "a.nav[href='/docs']"}}),
      call("browser_inspect_element", {{"target", "7"}}),
      call("browser_console_logs", {{"level", "error"}, {"clear", "true"}}),
      "<tool_call>\n<function=browser_close>\n</function>\n</tool_call>",
      call("codemode", {{"code", "const r = await tools.mcp__db__lookup({ sku: \"A-17\" });\nif (r.n < 3 && r.ok) return `got ${r.n}`;"}}),
    };
    auto ppart = [&](const std::string& t, bool f) { return chat::partial_tool_calls(t, ptools, f); };
    auto pwhole = [&](const std::string& t) { return chat::parse_tool_calls(t, ptools); };
    for (size_t c = 0; c < calls.size(); ++c) {
      const std::string& t = calls[c];
      check(t, ppart, pwhole);
      const auto fin = ppart(t, true);
      Json args;
      if (fin.size() != 1 || !Json::try_parse(fin[0].args, args)) { fail("Pi tool: final arguments", t, t.size(), fin.empty() ? "none" : fin[0].args); continue; }
      const Json& types = pi[(int)c]["p"];
      for (const auto& [k, v] : dedupe(args).entries()) {
        const std::string ty = types[k].str();
        const bool good = ty == "string" ? v.is_string() : ty == "number" ? v.is_number() : ty == "boolean" ? v.is_bool() : ty == "array" ? v.is_array() : false;
        if (!good) fail(("Pi tool " + fin[0].name + ": type of " + k).c_str(), t, t.size(), v.dump() + " is not a " + ty);
      }
      // a long string / array value streams: half way through it, the arguments already hold its start
      for (const auto& [k, ty] : types.entries()) {
        if (ty.str() != "string" && ty.str() != "array") continue;
        const size_t ks = t.find("<parameter=" + k + ">\n");
        if (ks == std::string::npos) continue;
        const size_t vs = ks + k.size() + 13, ve = t.find("\n</parameter>", vs);
        if (ve - vs < 20) continue;
        const auto mid = ppart(t.substr(0, vs + (ve - vs) / 2), false);
        const std::string head = t.substr(vs, 6);
        const std::string want = ty.str() == "array" ? head : chat::json_str_body(head);
        if (mid.empty() || mid[0].args.find("\"" + k + "\":") == std::string::npos || mid[0].args.find(want) == std::string::npos)
          fail(("Pi tool " + fin[0].name + ": " + k + " not streamed").c_str(), t, vs + (ve - vs) / 2, mid.empty() ? "none" : mid[0].args);
      }
    }
    printf("Pi's tools: %zu calls checked\n", calls.size());
  }

  // random responses built from tag fragments and awkward characters
  std::mt19937 rng(12345);
  const std::vector<std::string> qa = {"<tool_call>", "</tool_call>", "<function=write>", "<function=edit>", "<function=read>",
    "<parameter=path>", "<parameter=content>", "<parameter=edits>", "<parameter=limit>", "</parameter>", "</function>", "\n", "\n",
    "a", "\xC3\xA9", "\"", "\\", "{", "}", "[", "]", ",", ":", "1", " ", "<", ">", "/", "=", "</", "<para", "</tool_c", "\xF0\x9F\x98\x80"};
  const std::vector<std::string> ga = {"<|tool_call>", "<tool_call|>", "call:", "write", "edit", "{", "}", "[", "]", ",", ":", Q, Q,
    "\"", "\\", "path", "content", " ", "\n", "1", "true", "None", "x", "\xC3\xA9", "<|", "|>", "\xF0\x9F\x98\x80"};
  for (int it = 0; it < iters && failures <= 10; ++it) {
    const bool g = it & 1;
    const auto& al = g ? ga : qa;
    std::string t = g ? "<|tool_call>call:" : "<tool_call>\n<function=write>\n";
    for (int n = rng() % 40; n > 0; --n) t += al[rng() % al.size()];
    if (g) check(t, gp, gw);
    else check(t, qp, qw);
  }

  // reasoning / content deltas, text fed byte by byte (characters split over tokens): every delta must end on a
  // whole UTF-8 character
  auto check_deltas = [](chat::OutputParserBase& p, const std::string& text) {
    std::string dr, dc, all;
    for (size_t n = 1; n <= text.size(); ++n) {
      p.update(text.substr(0, n), n == text.size(), dr, dc);
      for (const std::string* d : {&dr, &dc})
        if (chat::utf8_end(*d, 0, d->size()) != d->size()) fail("delta splits a UTF-8 character", text, n, *d);
      all += dr + dc;
    }
    if (all.find("\xE2\x98\x95") == std::string::npos) fail("delta text lost a character", text, text.size(), all);
  };
  chat::OutputParser qd(true, tools);
  check_deltas(qd, "caf\xC3\xA9 \xE2\x98\x95 \xF0\x9F\x98\x80</think>\n\nAnswer \xE2\x80\x94 na\xC3\xAFve \xE2\x98\x95.");
  gemma4::OutputParser gd;
  check_deltas(gd, "<|channel>thought\nthinking \xE2\x98\x95<channel|>Answer \xE2\x80\x94 \xF0\x9F\x98\x80 \xE2\x98\x95");

  // the parser wrappers: Qwen calls only after the reasoning, Gemma anywhere
  chat::OutputParser qparser(true, tools);
  const std::string qt = "reasoning <tool_call>\n<function=read>\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>";
  const auto qc = qparser.partial_tool_calls(qt, true);
  if (qc.size() != 1 || qc[0].args != "{\"path\":\"x\"}") fail("Qwen parser after </think>", qt, qt.size(), qc.empty() ? "none" : qc[0].args);

  if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
  printf("tool call streaming: ALL OK (%zu + %zu cases, %d random)\n", qwen.size(), gemma.size(), iters);
  return 0;
}

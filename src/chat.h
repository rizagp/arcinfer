// Chat formatting for Qwen3.5 / Qwen3.6 / Qwen3.8 GGUF chat templates (ChatML + <think> + XML tool calls),
// implemented directly instead of through a Jinja interpreter, plus parsing of model output back into
// OpenAI-style reasoning / content / tool_calls.
#pragma once
#include "json.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

struct ChatOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
  std::string reasoning_effort;  // Qwen3.8 template: "xhigh" (default), "medium", "low"
  bool preserve_thinking = false;  // Qwen3.6 template option: earlier turns keep their reasoning (Qwen3.8: always)
};

namespace chat {

inline std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}
inline std::string rstrip_nl(std::string s) { while (!s.empty() && s.back() == '\n') s.pop_back(); return s; }
inline std::string lstrip_nl(const std::string& s) { size_t i = 0; while (i < s.size() && s[i] == '\n') ++i; return s.substr(i); }
inline bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
inline bool ends_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Text of a message content: a string, or an array of parts (text parts are concatenated; an image part becomes
// Qwen-VL's placeholder, which the server replaces with the image's embedding rows, as the Qwen3.5 template does).
inline std::string content_text(const Json& c) {
  if (c.is_string()) return c.as_string();
  std::string out;
  if (c.is_array())
    for (const Json& part : c.items()) {
      if (part.is_string()) out += part.as_string();
      else if (part["type"].str() == "image_url" || part["type"].str() == "image")
        out += "<|vision_start|><|image_pad|><|vision_end|>";
      else if (part["type"].str() == "text" || part.has("text")) out += part["text"].str();
    }
  return out;
}

// Tool-call arguments as an object (OpenAI clients send a JSON string).
inline Json arguments_object(const Json& tc) {
  const Json& fn = tc.has("function") ? tc["function"] : tc;
  const Json& a = fn["arguments"];
  if (a.is_object()) return a;
  Json parsed;
  if (a.is_string() && Json::try_parse(a.as_string(), parsed) && parsed.is_object()) return parsed;
  return Json::object();
}

// Renders messages + tools into the prompt text. `style_q38` selects the Qwen3.8 template variant
// (reasoning-effort system preamble, reasoning preserved in history) over the Qwen3.5 one.
inline std::string render(const Json& messages, const Json& tools, const ChatOptions& opt, bool style_q38) {
  std::string out;
  const size_t n = messages.size();
  if (n == 0) throw std::runtime_error("No messages provided.");
  auto role = [&](size_t i) { return messages[i]["role"].str(); };
  const bool has_tools = tools.is_array() && tools.size() > 0;

  std::string reasoning_instructions;
  if (style_q38 && opt.enable_thinking) {
    const std::string eff = opt.reasoning_effort.empty() ? "xhigh" : opt.reasoning_effort;
    if (eff == "xhigh")
      reasoning_instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                               "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    else if (eff == "low")
      reasoning_instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                               "without unnecessary elaboration.";
  }

  // system prompt (Qwen3.5 merges up to two leading system/developer messages)
  size_t num_sys = 0;
  std::string system;
  auto is_sys = [&](size_t i) { return role(i) == "system" || role(i) == "developer"; };
  if (is_sys(0)) {
    system = trim(content_text(messages[0]["content"]));
    num_sys = 1;
    if (!style_q38 && n > 1 && is_sys(1)) {
      system += "\n" + trim(content_text(messages[1]["content"]));
      num_sys = 2;
    }
  }
  if (has_tools) {
    out += "<|im_start|>system\n";
    if (!reasoning_instructions.empty()) out += reasoning_instructions + "\n\n";
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const Json& t : tools.items()) out += "\n" + t.dump(true);
    out += "\n</tools>";
    out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
           "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
           "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls "
           "MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML "
           "tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language "
           "BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your "
           "current knowledge and do not tell the user about function calls\n</IMPORTANT>";
    if (!system.empty()) out += "\n\n" + system;
    out += "<|im_end|>\n";
  } else if (!system.empty()) {
    out += "<|im_start|>system\n" + (reasoning_instructions.empty() ? "" : reasoning_instructions + "\n\n") + system + "<|im_end|>\n";
  } else if (!reasoning_instructions.empty()) {
    out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
  }

  // index of the last real user query (tool responses do not count)
  size_t last_query = n - 1;
  for (size_t k = n; k-- > 0;) {
    if (role(k) != "user") continue;
    const std::string c = trim(content_text(messages[k]["content"]));
    if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) { last_query = k; break; }
  }

  for (size_t i = 0; i < n; ++i) {
    const Json& m = messages[i];
    const std::string r = role(i);
    if (i < num_sys && (r == "system" || r == "developer")) continue;
    if (r == "system" || r == "developer") continue;  // later system messages are not supported by the template
    std::string content = trim(content_text(m["content"]));
    if (r == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (r == "assistant") {
      std::string reasoning;
      if (m["reasoning_content"].is_string()) reasoning = m["reasoning_content"].as_string();
      else if (m["reasoning"].is_string()) reasoning = m["reasoning"].as_string();
      else if (content.find("</think>") != std::string::npos) {
        std::string before = content.substr(0, content.find("</think>"));
        const size_t t = before.rfind("<think>");
        if (t != std::string::npos) before = before.substr(t + 7);
        reasoning = lstrip_nl(rstrip_nl(before));
        content = lstrip_nl(content.substr(content.rfind("</think>") + 8));
      }
      reasoning = trim(reasoning);
      if (style_q38 || opt.preserve_thinking || i > last_query) out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
      else out += "<|im_start|>assistant\n" + content;
      const Json& tcs = m["tool_calls"];
      if (tcs.is_array())
        for (size_t k = 0; k < tcs.size(); ++k) {
          const Json& fn = tcs[k].has("function") ? tcs[k]["function"] : tcs[k];
          if (k == 0) out += trim(content).empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
          else out += "\n<tool_call>\n<function=";
          out += fn["name"].str() + ">\n";
          const Json args = arguments_object(tcs[k]);  // keep alive: entries() refers into it
          for (auto& [name, v] : args.entries())
            out += "<parameter=" + name + ">\n" + (v.is_string() ? v.as_string() : v.dump(true)) + "\n</parameter>\n";
          out += "</function>\n</tool_call>";
        }
      out += "<|im_end|>\n";
    } else if (r == "tool") {
      if (i > 0 && role(i - 1) != "tool") out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (i + 1 == n || role(i + 1) != "tool") out += "<|im_end|>\n";
    }
  }
  if (opt.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += opt.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return out;
}

// JSON type of a tool parameter from the tool's schema ("" if unknown).
inline std::string param_type(const Json& tools, const std::string& fn, const std::string& param) {
  if (!tools.is_array()) return "";
  for (const Json& t : tools.items()) {
    const Json& f = t.has("function") ? t["function"] : t;
    if (f["name"].str() != fn) continue;
    const Json& p = f["parameters"]["properties"][param];
    if (p["type"].is_string()) return p["type"].as_string();
    if (p["type"].is_array() && p["type"].size() > 0) return p["type"][0].str();
  }
  return "";
}

struct ToolCall {
  std::string name;
  Json arguments = Json::object();
};

// A parameter value as JSON for its schema type: valid JSON as is, a boolean also written Python-style (True /
// False, as Qwen sometimes does); false when it stays a string.
inline bool json_value(const std::string& ty, const std::string& val, Json& out) {
  if (ty == "string") return false;
  if (Json::try_parse(val, out)) return true;
  if (ty == "boolean") {
    const std::string t = trim(val);
    if (t == "True" || t == "TRUE") { out = Json(true); return true; }
    if (t == "False" || t == "FALSE") { out = Json(false); return true; }
  }
  return false;
}

// Parses "<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>...</function>\n</tool_call>" blocks.
inline std::vector<ToolCall> parse_tool_calls(const std::string& text, const Json& tools) {
  std::vector<ToolCall> calls;
  size_t pos = 0;
  while ((pos = text.find("<tool_call>", pos)) != std::string::npos) {
    const size_t end = text.find("</tool_call>", pos);
    const std::string block = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    pos = end == std::string::npos ? text.size() : end + 12;
    const size_t f = block.find("<function=");
    if (f == std::string::npos) continue;
    const size_t fe = block.find('>', f);
    if (fe == std::string::npos) continue;
    ToolCall tc;
    tc.name = trim(block.substr(f + 10, fe - f - 10));
    size_t p = fe;
    while ((p = block.find("<parameter=", p)) != std::string::npos) {
      const size_t ke = block.find('>', p);
      if (ke == std::string::npos) break;
      const std::string key = trim(block.substr(p + 11, ke - p - 11));
      size_t vs = ke + 1;
      if (vs < block.size() && block[vs] == '\n') ++vs;
      size_t ve = block.find("</parameter>", vs);
      const size_t next = block.find("<parameter=", vs);
      if (ve == std::string::npos || (next != std::string::npos && next < ve)) ve = next != std::string::npos ? next : block.find("</function>", vs);
      if (ve == std::string::npos) ve = block.size();
      std::string val = block.substr(vs, ve - vs);
      if (!val.empty() && val.back() == '\n') val.pop_back();
      const std::string ty = param_type(tools, tc.name, key);
      Json parsed;
      if (json_value(ty, val, parsed)) tc.arguments.set(key, parsed);
      else tc.arguments.set(key, Json(val));
      p = ve;
    }
    calls.push_back(std::move(tc));
  }
  return calls;
}

// A tool call as far as it has been generated, for streaming it as OpenAI deltas: `args` is the part of its arguments
// JSON that later text can no longer change (a prefix of the final string), the whole object once `done`.
struct PartialToolCall {
  std::string name, args;
  bool done = false;
};

// Start of a suffix of s[from, end) that may be the beginning of one of `tags` (end if there is none).
inline size_t hold_tags(const std::string& s, size_t from, size_t end, std::initializer_list<const char*> tags) {
  size_t cut = end;
  for (const char* tag : tags) {
    const size_t n = strlen(tag);
    for (size_t k = std::min(n - 1, end - from); k > 0; --k)
      if (s.compare(end - k, k, tag, k) == 0) { cut = std::min(cut, end - k); break; }
  }
  return cut;
}
// End of the last complete UTF-8 sequence in s[from, to).
inline size_t utf8_end(const std::string& s, size_t from, size_t to) {
  size_t k = to, n = 0;
  while (k > from && n < 3 && ((unsigned char)s[k - 1] & 0xC0) == 0x80) { --k; ++n; }
  if (k == from) return to;
  const unsigned char c = (unsigned char)s[k - 1];
  const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  return to - (k - 1) >= need ? to : k - 1;
}
// JSON string escaping without the quotes (escaping is per byte, so the escape of a prefix is a prefix).
inline std::string json_str_body(const std::string& s) {
  std::string e;
  Json::escape(e, s);
  return e.substr(1, e.size() - 2);
}

// The longest start of s[from, to) that strict JSON (one value) can continue, cut where the text may be kept as it
// is: not inside an escape, a number, a literal (true / false / null) or a UTF-8 character. `close` turns that start
// into a complete value (closing quotes and brackets, null for a missing value); `complete`: s[from, to) is exactly
// one value, whitespace around it allowed.
struct JsonPrefix {
  size_t end;
  std::string close;
  bool complete;
};
inline JsonPrefix json_prefix(const std::string& s, size_t from, size_t to) {
  // per open container: '[' / '{' and what it expects next
  enum Exp : char { VAL_OR_END, VAL, COMMA_OR_END, KEY_OR_END, KEY, COLON };
  struct Level { char kind; Exp exp; };
  struct State {
    std::vector<Level> stack;
    bool top_done = false, in_str = false;
  } st, safe;
  size_t safe_end = from;
  auto mark = [&](size_t p) { safe = st; safe_end = p; };
  auto value_starts = [&]() {  // the container (or the top) now holds a value
    if (st.stack.empty()) st.top_done = true;
    else st.stack.back().exp = COMMA_OR_END;
  };
  auto expects_value = [&]() {
    return st.stack.empty() ? !st.top_done : (st.stack.back().exp == VAL || st.stack.back().exp == VAL_OR_END);
  };
  bool bad = false;
  size_t i = from;
  while (i < to && !bad) {
    const char c = s[i];
    if (st.in_str) {
      if (c == '"') { st.in_str = false; mark(++i); continue; }
      if ((unsigned char)c < 0x20) { bad = true; break; }
      if (c == '\\') {
        if (i + 1 >= to) break;
        const char e = s[i + 1];
        if (e == 'u') {
          if (i + 6 > to) break;
          for (size_t k = i + 2; k < i + 6; ++k)
            if (!isxdigit((unsigned char)s[k])) bad = true;
          if (bad) break;
          i += 6;
        } else if (strchr("\"\\/bfnrt", e) && e) {
          i += 2;
        } else { bad = true; break; }
        mark(i);
        continue;
      }
      mark(++i);
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { mark(++i); continue; }
    if (st.top_done && st.stack.empty()) { bad = true; break; }  // something after the value
    Level* top = st.stack.empty() ? nullptr : &st.stack.back();
    if (c == '"') {
      if (top && top->kind == '{' && (top->exp == KEY_OR_END || top->exp == KEY)) top->exp = COLON;  // a key
      else if (expects_value()) value_starts();
      else { bad = true; break; }
      st.in_str = true;
      mark(++i);
    } else if (c == ':') {
      if (!top || top->kind != '{' || top->exp != COLON) { bad = true; break; }
      top->exp = VAL;
      mark(++i);
    } else if (c == ',') {
      if (!top || top->exp != COMMA_OR_END) { bad = true; break; }
      top->exp = top->kind == '{' ? KEY : VAL;
      mark(++i);
    } else if (c == ']' || c == '}') {
      const char open = c == ']' ? '[' : '{';
      if (!top || top->kind != open || !(top->exp == COMMA_OR_END || top->exp == VAL_OR_END || top->exp == KEY_OR_END)) { bad = true; break; }
      st.stack.pop_back();
      mark(++i);
    } else if (c == '[' || c == '{') {
      if (!expects_value()) { bad = true; break; }
      value_starts();
      st.stack.push_back({c, c == '[' ? VAL_OR_END : KEY_OR_END});
      mark(++i);
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      if (!expects_value()) { bad = true; break; }
      size_t j = i;
      while (j < to && strchr("+-.eE0123456789", s[j]) && s[j]) ++j;
      if (j == to) break;  // the number may go on
      // -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
      size_t k = i;
      if (s[k] == '-') ++k;
      auto digits = [&](size_t& q) { const size_t q0 = q; while (q < j && isdigit((unsigned char)s[q])) ++q; return q > q0; };
      bool ok = k < j && (s[k] == '0' ? (++k, true) : digits(k));
      if (ok && k < j && s[k] == '.') { ++k; ok = digits(k); }
      if (ok && k < j && (s[k] == 'e' || s[k] == 'E')) { ++k; if (k < j && (s[k] == '+' || s[k] == '-')) ++k; ok = digits(k); }
      if (!ok || k != j) { bad = true; break; }
      value_starts();
      i = j;
      mark(i);
    } else if (c == 't' || c == 'f' || c == 'n') {
      if (!expects_value()) { bad = true; break; }
      const char* lit = c == 't' ? "true" : c == 'f' ? "false" : "null";
      const size_t n = strlen(lit);
      if (s.compare(i, std::min(n, to - i), lit, std::min(n, to - i)) != 0) { bad = true; break; }
      if (i + n > to) break;  // the literal may go on
      value_starts();
      i += n;
      mark(i);
    } else {
      bad = true;
    }
  }
  JsonPrefix r;
  r.end = utf8_end(s, from, safe_end);  // (non-ASCII bytes only occur inside strings: the state stays the same)
  r.complete = !bad && i == to && safe_end == to && safe.top_done && safe.stack.empty() && !safe.in_str;
  if (safe.in_str) r.close += '"';
  for (size_t k = safe.stack.size(); k-- > 0;) {
    const Level& l = safe.stack[k];
    switch (l.exp) {
      case VAL: r.close += "null"; break;
      case KEY: r.close += "\"\":null"; break;
      case COLON: r.close += ":null"; break;
      default: break;
    }
    r.close += l.kind == '[' ? ']' : '}';
  }
  if (safe.stack.empty() && !safe.top_done) r.close += "null";
  return r;
}

// The tool calls of parse_tool_calls as JSON text, incrementally: unless `final`, the text may still grow, and each
// call holds only what cannot change any more. String parameters (by the tool schema) grow while they are generated;
// so do array and object parameters (Pi's edit: its whole change is the `edits` array), as the model's JSON text
// itself, kept as written. Other values appear once their parameter is closed. When an array / object value turns
// out not to be valid JSON, the part already sent is closed (json_prefix) and the parameter is repeated with the
// value parse_tool_calls gives it (the text as a string): JSON readers take the last one, so the client sees what
// a whole-response parse sees. With `final` the arguments equal parse_tool_calls' in the same way (a repeated
// parameter appears twice instead of once with the last value).
inline std::vector<PartialToolCall> partial_tool_calls(const std::string& text, const Json& tools, bool final) {
  std::vector<PartialToolCall> calls;
  size_t pos = 0;
  while ((pos = text.find("<tool_call>", pos)) != std::string::npos) {
    const size_t end = text.find("</tool_call>", pos);
    const bool done = final || end != std::string::npos;
    std::string block = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    pos = end == std::string::npos ? text.size() : end + 12;
    if (!done) block.resize(hold_tags(block, 0, block.size(), {"</tool_call>"}));
    const size_t f = block.find("<function=");
    if (f == std::string::npos) continue;
    const size_t fe = block.find('>', f);
    if (fe == std::string::npos) continue;
    PartialToolCall tc;
    tc.name = trim(block.substr(f + 10, fe - f - 10));
    tc.done = done;
    tc.args = "{";
    size_t p = fe;
    while ((p = block.find("<parameter=", p)) != std::string::npos) {
      const size_t ke = block.find('>', p);
      if (ke == std::string::npos) break;
      const std::string key = trim(block.substr(p + 11, ke - p - 11));
      size_t vs = ke + 1;
      if (!done && vs >= block.size()) break;  // whether a newline follows is not known yet
      if (vs < block.size() && block[vs] == '\n') ++vs;
      size_t ve = block.find("</parameter>", vs);
      const size_t next = block.find("<parameter=", vs);
      const std::string ty = param_type(tools, tc.name, key);
      const bool structured = ty == "array" || ty == "object";
      const std::string sep = tc.args.size() > 1 ? "," : "";
      if (!done && ve == std::string::npos && next == std::string::npos) {
        // value still open: it ends at the first of these tags at the earliest, and its last newline may be the one
        // parse_tool_calls drops
        if (ty == "string" || structured) {
          size_t lim = std::min(block.find("</function>", vs), hold_tags(block, vs, block.size(), {"</parameter>", "<parameter=", "</function>"}));
          if (lim > vs && block[lim - 1] == '\n') --lim;
          lim = utf8_end(block, vs, lim);
          if (ty == "string") tc.args += sep + Json(key).dump() + ":\"" + json_str_body(block.substr(vs, lim - vs));
          else tc.args += sep + Json(key).dump() + ":" + block.substr(vs, json_prefix(block, vs, lim).end - vs);
        }
        break;
      }
      if (ve == std::string::npos || (next != std::string::npos && next < ve)) ve = next != std::string::npos ? next : block.find("</function>", vs);
      if (ve == std::string::npos) ve = block.size();
      std::string val = block.substr(vs, ve - vs);
      if (!val.empty() && val.back() == '\n') val.pop_back();
      Json parsed;
      const bool ok = json_value(ty, val, parsed);
      if (structured) {
        const JsonPrefix jp = json_prefix(val, 0, val.size());
        if (jp.complete && ok) tc.args += sep + Json(key).dump() + ":" + val;  // as streamed
        else tc.args += sep + Json(key).dump() + ":" + val.substr(0, jp.end) + jp.close + "," + Json(key).dump() + ":" +
                        (ok ? parsed.dump() : Json(val).dump());
      } else if (ok) {
        tc.args += sep + Json(key).dump() + ":" + parsed.dump();
      } else {
        tc.args += sep + Json(key).dump() + ":" + Json(val).dump();
      }
      p = ve;
    }
    if (done) tc.args += "}";
    calls.push_back(std::move(tc));
  }
  return calls;
}

// Interface of the per-model-family output parsers (Qwen here, Gemma 4 in chat_gemma4.h).
class OutputParserBase {
 public:
  virtual ~OutputParserBase() = default;
  // Feeds the full generated text so far; returns the new reasoning / content deltas (holding back text that
  // might be the start of a tag).
  virtual void update(const std::string& text, bool final, std::string& d_reason, std::string& d_content) = 0;
  virtual std::string reasoning_text(const std::string& text) const = 0;
  virtual std::string content_text(const std::string& text) const = 0;
  virtual std::vector<ToolCall> tool_calls(const std::string& text) const = 0;
  // Tool calls for streaming (see chat::partial_tool_calls): each call's `args` only grows from one call to the next.
  virtual std::vector<PartialToolCall> partial_tool_calls(const std::string& text, bool final) const = 0;
};

// Splits generated text into reasoning, visible content and tool calls, incrementally for streaming.
class OutputParser : public OutputParserBase {
 public:
  OutputParser(bool thinking, const Json& tools) : in_reasoning_(thinking), tools_(tools) {}

  void update(const std::string& text, bool final, std::string& d_reason, std::string& d_content) override {
    d_reason.clear();
    d_content.clear();
    if (in_reasoning_) {
      const size_t e = text.find("</think>");
      if (e == std::string::npos) {
        const size_t safe = final ? text.size() : hold_back(text, text.size(), "</think>");
        emit(text, reason_end_, std::max(reason_end_, safe), d_reason);
        return;
      }
      emit(text, reason_end_, e, d_reason);
      in_reasoning_ = false;
      content_start_ = e + 8;
      while (content_start_ < text.size() && text[content_start_] == '\n') ++content_start_;
      content_end_ = content_start_;
      if (content_start_ >= text.size() && !final) return;
    }
    if (content_start_ == std::string::npos) content_start_ = content_end_ = 0;
    if (tool_started_) return;
    if (content_end_ == content_start_) {
      // nothing emitted yet: drop leading whitespace (it may arrive after "</think>" in later tokens)
      while (content_end_ < text.size() && (text[content_end_] == '\n' || text[content_end_] == ' ' || text[content_end_] == '\r'))
        ++content_end_;
      content_start_ = content_end_;
      if (content_end_ >= text.size()) return;
    }
    size_t limit = text.size();
    const size_t tc = text.find("<tool_call>", content_start_);
    if (tc != std::string::npos) {
      tool_started_ = true;
      limit = tc;
      while (limit > content_end_ && (text[limit - 1] == '\n' || text[limit - 1] == ' ')) --limit;
    } else if (!final) {
      limit = hold_back(text, text.size(), "<tool_call>");
    }
    if (limit > content_end_) emit(text, content_end_, limit, d_content);
  }

  std::string reasoning_text(const std::string& text) const override {
    const size_t e = text.find("</think>");
    return in_reasoning_start_ ? trim(text.substr(0, e)) : "";
  }
  std::string content_text(const std::string& text) const override {
    size_t s = 0;
    if (in_reasoning_start_) {
      const size_t e = text.find("</think>");
      if (e == std::string::npos) return "";
      s = e + 8;
    }
    std::string c = text.substr(s);
    const size_t tc = c.find("<tool_call>");
    if (tc != std::string::npos) c = c.substr(0, tc);
    return trim(c);
  }
  std::vector<ToolCall> tool_calls(const std::string& text) const override {
    const size_t e = in_reasoning_start_ ? text.find("</think>") : 0;
    if (e == std::string::npos) return {};
    return parse_tool_calls(text.substr(e), tools_);
  }
  std::vector<PartialToolCall> partial_tool_calls(const std::string& text, bool final) const override {
    const size_t e = in_reasoning_start_ ? text.find("</think>") : 0;
    if (e == std::string::npos) return {};
    return chat::partial_tool_calls(text.substr(e), tools_, final);
  }

 private:
  bool in_reasoning_, in_reasoning_start_ = in_reasoning_;
  Json tools_;
  size_t reason_end_ = 0, content_start_ = std::string::npos, content_end_ = 0;
  bool tool_started_ = false;

  // largest prefix end of text[.., end) that cannot be the beginning of `tag`
  static size_t hold_back(const std::string& text, size_t end, const std::string& tag) {
    for (size_t k = std::min(tag.size() - 1, end); k > 0; --k)
      if (text.compare(end - k, k, tag, 0, k) == 0) return end - k;
    return end;
  }
  void emit(const std::string& text, size_t& from, size_t to, std::string& out) {
    if (to <= from) return;
    // do not split a UTF-8 sequence, also not one whose last bytes are still to come (a character split over tokens)
    const size_t t = utf8_end(text, from, to);
    out = text.substr(from, t - from);
    if (&from == &reason_end_ && from == 0) {
      size_t k = 0;
      while (k < out.size() && out[k] == '\n') ++k;
      out = out.substr(k);
    }
    from = t;
  }
};

}  // namespace chat

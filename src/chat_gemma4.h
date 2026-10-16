// Gemma 4 output parsing: the reasoning channel (<|channel>thought ... <channel|>), visible content, and tool calls
// written as <|tool_call>call:NAME{key:value,...}<tool_call|>, where strings are delimited by <|"|> and keys are
// bare words (the format the Gemma 4 chat template uses for tool declarations, calls and responses).
#pragma once
#include "chat.h"

namespace gemma4 {

constexpr const char* kQ = "<|\"|>";  // Gemma 4 string delimiter token

// Parser for the argument syntax: {key:value,...}, [v,...], <|"|>text<|"|>, numbers, true / false / null.
struct ArgParser {
  const std::string& s;
  size_t i = 0;
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
  bool at(const char* t) const { return s.compare(i, strlen(t), t) == 0; }
  std::string str() {  // after the opening <|"|>
    const size_t e = s.find(kQ, i);
    std::string v = s.substr(i, e == std::string::npos ? std::string::npos : e - i);
    i = e == std::string::npos ? s.size() : e + strlen(kQ);
    return v;
  }
  std::string key() {
    ws();
    if (at(kQ)) { i += strlen(kQ); return str(); }
    if (i < s.size() && s[i] == '"') {  // JSON-style quoted key
      const size_t e = s.find('"', i + 1);
      std::string k = s.substr(i + 1, e == std::string::npos ? std::string::npos : e - i - 1);
      i = e == std::string::npos ? s.size() : e + 1;
      return k;
    }
    const size_t st = i;
    while (i < s.size() && s[i] != ':' && s[i] != '}' && s[i] != ',') ++i;
    return chat::trim(s.substr(st, i - st));
  }
  Json value() {
    ws();
    if (i >= s.size()) return Json();
    if (at(kQ)) { i += strlen(kQ); return Json(str()); }
    if (s[i] == '{') return object();
    if (s[i] == '[') {
      ++i;
      Json a = Json::array();
      for (;;) {
        ws();
        if (i >= s.size()) break;
        if (s[i] == ']') { ++i; break; }
        if (s[i] == '}') break;  // a stray '}' ends the array (value() would not move past it: endless loop)
        a.push(value());
        ws();
        if (i < s.size() && s[i] == ',') ++i;
      }
      return a;
    }
    if (s[i] == '"') {  // JSON string
      Json j;
      const size_t st = i;
      ++i;
      while (i < s.size() && s[i] != '"') i += s[i] == '\\' ? 2 : 1;
      ++i;
      if (Json::try_parse(s.substr(st, i - st), j)) return j;
      return Json(s.substr(st + 1, i - st - 2));
    }
    const size_t st = i;
    int depth = 0;
    while (i < s.size() && !(depth == 0 && (s[i] == ',' || s[i] == '}' || s[i] == ']'))) {
      if (s[i] == '{' || s[i] == '[') ++depth;
      else if (s[i] == '}' || s[i] == ']') --depth;
      ++i;
    }
    const std::string raw = chat::trim(s.substr(st, i - st));
    Json j;
    if (raw == "None") return Json();
    if (Json::try_parse(raw, j)) return j;
    return Json(raw);
  }
  Json object() {  // at '{'
    ++i;
    Json o = Json::object();
    for (;;) {
      ws();
      if (i >= s.size()) break;
      if (s[i] == '}') { ++i; break; }
      if (s[i] == ']') break;  // (an object left open before its array's end: closed here, the array takes the ']')
      const std::string k = key();
      ws();
      if (i < s.size() && s[i] == ':') ++i;
      o.set(k, value());
      ws();
      if (i < s.size() && s[i] == ',') ++i;
    }
    return o;
  }
};

// A long call sometimes nests a top-level argument inside the last element of an array argument (Pi's edit:
// {"edits": [{..., "path": ...}]} instead of {"edits": [...], "path": ...}). With the tool's schema, a required
// top-level parameter that is missing is moved up from there when the array's items do not have such a property.
inline void hoist_required(Json& args, const Json& tools, const std::string& name) {
  const Json* params = nullptr;
  for (const Json& t : tools.items())
    if (t["function"]["name"].str() == name) params = &t["function"]["parameters"];
  if (!params || !args.is_object()) return;
  for (const Json& r : (*params)["required"].items()) {
    const std::string req = r.str();
    if (req.empty() || args.has(req)) continue;
    for (const auto& [k, v] : args.entries()) {
      if (!v.is_array() || !v.size()) continue;
      const Json& item_props = (*params)["properties"][k]["items"]["properties"];
      if (item_props.has(req)) continue;  // (the items have such a field of their own)
      const Json& last = v[v.size() - 1];
      if (!last.is_object() || !last.has(req)) continue;
      Json nl = Json::object(), arr = Json::array();
      for (const auto& [lk, lv] : last.entries())
        if (lk != req) nl.set(lk, lv);
      for (size_t i = 0; i + 1 < v.size(); ++i) arr.push(v[i]);
      arr.push(nl);
      const Json moved = last[req];
      args.set(k, arr);
      args.set(req, moved);
      break;
    }
  }
}

inline std::vector<chat::ToolCall> parse_tool_calls(const std::string& text, const Json& tools = Json()) {
  std::vector<chat::ToolCall> calls;
  size_t pos = 0;
  while ((pos = text.find("<|tool_call>", pos)) != std::string::npos) {
    pos += 12;
    size_t end = text.find("<tool_call|>", pos);
    const std::string block = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    pos = end == std::string::npos ? text.size() : end + 12;
    size_t b = 0;
    while (b < block.size() && (block[b] == ' ' || block[b] == '\n')) ++b;
    if (block.compare(b, 5, "call:") == 0) b += 5;
    const size_t br = block.find('{', b);
    chat::ToolCall tc;
    tc.name = chat::trim(block.substr(b, br == std::string::npos ? std::string::npos : br - b));
    if (br != std::string::npos) {
      ArgParser p{block, br};
      Json a = p.object();
      if (a.is_object()) {
        hoist_required(a, tools, chat::trim(block.substr(b, br - b)));
        tc.arguments = a;
      }
    }
    if (!tc.name.empty()) calls.push_back(std::move(tc));
  }
  return calls;
}

// ArgParser, incrementally: writes the JSON of what it has read to `out` and, unless `final`, stops (returns false)
// where later text could still change the result; <|"|> strings are written while they grow. With `final` the
// output is ArgParser's object (a repeated key appears twice instead of once with the last value).
struct ArgStreamer {
  const std::string& s;
  bool final;
  size_t i = 0;
  std::string out;
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
  bool at(const char* t) const { return s.compare(i, strlen(t), t) == 0; }
  bool undecided() const {  // the text ends where a <|"|> may begin
    const size_t r = s.size() - i;
    return !final && r < strlen(kQ) && s.compare(i, r, kQ, r) == 0;
  }
  bool key(std::string& k) {
    ws();
    if (undecided()) return false;
    if (at(kQ)) {
      i += strlen(kQ);
      const size_t e = s.find(kQ, i);
      if (e == std::string::npos && !final) return false;
      k = s.substr(i, e == std::string::npos ? std::string::npos : e - i);
      i = e == std::string::npos ? s.size() : e + strlen(kQ);
      return true;
    }
    if (i < s.size() && s[i] == '"') {
      const size_t e = s.find('"', i + 1);
      if (e == std::string::npos && !final) return false;
      k = s.substr(i + 1, e == std::string::npos ? std::string::npos : e - i - 1);
      i = e == std::string::npos ? s.size() : e + 1;
      return true;
    }
    const size_t st = i;
    while (i < s.size() && s[i] != ':' && s[i] != '}' && s[i] != ',') ++i;
    if (i >= s.size() && !final) return false;
    k = chat::trim(s.substr(st, i - st));
    return true;
  }
  bool value() {
    ws();
    if (i >= s.size()) {
      if (!final) return false;
      out += "null";
      return true;
    }
    if (undecided()) return false;
    if (at(kQ)) {  // a string: written as it grows
      i += strlen(kQ);
      out += '"';
      const size_t e = s.find(kQ, i);
      if (e == std::string::npos && !final) {
        const size_t lim = chat::utf8_end(s, i, chat::hold_tags(s, i, s.size(), {kQ}));
        out += chat::json_str_body(s.substr(i, lim - i));
        return false;
      }
      out += chat::json_str_body(s.substr(i, e == std::string::npos ? std::string::npos : e - i)) + '"';
      i = e == std::string::npos ? s.size() : e + strlen(kQ);
      return true;
    }
    if (s[i] == '{') return object();
    if (s[i] == '[') {
      ++i;
      out += '[';
      for (bool first = true;; first = false) {
        ws();
        if (i >= s.size()) {
          if (!final) return false;
          break;
        }
        if (s[i] == ']') { ++i; break; }
        if (s[i] == '}') break;  // as ArgParser
        if (!first) out += ',';
        if (!value()) return false;
        ws();
        if (i < s.size() && s[i] == ',') ++i;
      }
      out += ']';
      return true;
    }
    if (s[i] == '"') {  // JSON string
      const size_t st = i;
      ++i;
      while (i < s.size() && s[i] != '"') i += s[i] == '\\' ? 2 : 1;
      if (i >= s.size() && !final) return false;
      ++i;
      Json j;
      if (!Json::try_parse(s.substr(st, i - st), j)) j = Json(s.substr(st + 1, i - st - 2));
      out += j.dump();
      return true;
    }
    const size_t st = i;
    int depth = 0;
    while (i < s.size() && !(depth == 0 && (s[i] == ',' || s[i] == '}' || s[i] == ']'))) {
      if (s[i] == '{' || s[i] == '[') ++depth;
      else if (s[i] == '}' || s[i] == ']') --depth;
      ++i;
    }
    if (i >= s.size() && !final) return false;
    const std::string raw = chat::trim(s.substr(st, i - st));
    Json j;
    if (raw == "None") j = Json();
    else if (!Json::try_parse(raw, j)) j = Json(raw);
    out += j.dump();
    return true;
  }
  bool object() {  // at '{'
    ++i;
    out += '{';
    for (bool first = true;; first = false) {
      ws();
      if (i >= s.size()) {
        if (!final) return false;
        break;
      }
      if (s[i] == '}') { ++i; break; }
      std::string k;
      if (!key(k)) return false;
      ws();
      if (i >= s.size() && !final) return false;  // a ':' may follow
      if (i < s.size() && s[i] == ':') ++i;
      out += (first ? "" : ",") + Json(k).dump() + ":";
      if (!value()) return false;
      ws();
      if (i < s.size() && s[i] == ',') ++i;
    }
    out += '}';
    return true;
  }
};

// The tool calls of parse_tool_calls as JSON text, incrementally (see chat::partial_tool_calls).
inline std::vector<chat::PartialToolCall> partial_tool_calls(const std::string& text, bool final, const Json& tools = Json()) {
  std::vector<chat::PartialToolCall> calls;
  size_t pos = 0;
  while ((pos = text.find("<|tool_call>", pos)) != std::string::npos) {
    pos += 12;
    const size_t end = text.find("<tool_call|>", pos);
    const bool done = final || end != std::string::npos;
    std::string block = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    pos = end == std::string::npos ? text.size() : end + 12;
    if (!done) block.resize(chat::hold_tags(block, 0, block.size(), {"<tool_call|>"}));
    size_t b = 0;
    while (b < block.size() && (block[b] == ' ' || block[b] == '\n')) ++b;
    if (block.compare(b, 5, "call:") == 0) b += 5;
    const size_t br = block.find('{', b);
    if (br == std::string::npos && !done) continue;  // the name is not complete yet
    chat::PartialToolCall tc;
    tc.name = chat::trim(block.substr(b, br == std::string::npos ? std::string::npos : br - b));
    tc.done = done;
    if (tc.name.empty()) continue;
    if (br == std::string::npos) {
      tc.args = done ? "{}" : "";
    } else if (!done) {
      tc.args = "";  // (the arguments come whole when the call is complete)
    } else {
      ArgParser p{block, br};
      Json a = p.object();
      if (!a.is_object()) a = Json::object();
      hoist_required(a, tools, tc.name);
      tc.args = a.dump();
    }
    calls.push_back(std::move(tc));
  }
  return calls;
}

// Streaming split of Gemma 4 output into reasoning / content / tool calls (same interface as chat::OutputParser).
// Gemma 4 may think more than once in an answer: a "<|channel>thought ... <channel|>" block before each tool call,
// not only at the start, and it may call a tool from inside a thought block without closing it. Every thought block
// goes to the reasoning (blocks joined by a blank line); a tool call ends the text (its JSON comes through
// partial_tool_calls / tool_calls); a "<channel|>" without an opening is dropped.
// in_thought: the prompt already opened a thought block (Gemma 4's template ends the prompt with
// "<|channel>thought\n" after a tool response when thinking is on): the output starts inside the reasoning.
class OutputParser : public chat::OutputParserBase {
 public:
  explicit OutputParser(bool in_thought = false, Json tools = Json()) : in_thought_(in_thought), tools_(std::move(tools)) {}
  struct Split {
    std::string reason, content;
    bool tool = false;
  };
  // The split of the whole output so far. Unless `final`, a tag that may still be completing at the end is held
  // back; the parts only grow as the text grows (the stream sends their new ends).
  static Split split(const std::string& text, bool final, bool in_thought = false) {
    static const std::string open = "<|channel>thought", close = "<channel|>", tool = "<|tool_call>";
    Split r;
    const size_t n = text.size(), npos = std::string::npos;
    size_t pos = 0;
    bool reasoning = in_thought, seg_start = true;
    auto first_of = [&](std::initializer_list<const std::string*> tags, size_t& which_len, const std::string*& which) {
      size_t best = npos;
      for (const std::string* t : tags) {
        const size_t k = text.find(*t, pos);
        if (k < best) { best = k; which = t; which_len = t->size(); }
      }
      return best;
    };
    // the end of the text that can be sent: before a possible beginning of one of the tags
    auto held = [&](std::initializer_list<const std::string*> tags) {
      if (final) return n;
      size_t lim = n;
      for (const std::string* t : tags)
        for (size_t k = std::min(t->size() - 1, n - pos); k > 0; --k)
          if (text.compare(n - k, k, *t, 0, k) == 0) { lim = std::min(lim, n - k); break; }
      return lim;
    };
    while (pos <= n) {
      if (!reasoning) {
        if (seg_start) {
          while (pos < n && (text[pos] == '\n' || text[pos] == ' ')) ++pos;
          seg_start = false;
        }
        size_t len = 0;
        const std::string* which = nullptr;
        const size_t k = first_of({&open, &close, &tool}, len, which);
        if (k == npos) {
          const size_t lim = held({&open, &close, &tool});
          if (lim > pos) r.content.append(text, pos, lim - pos);
          break;
        }
        r.content.append(text, pos, k - pos);
        pos = k + len;
        if (which == &tool) { r.tool = true; break; }
        if (which == &close) { seg_start = true; continue; }  // (a stray close: dropped)
        reasoning = true;
        if (pos < n && text[pos] == '\n') ++pos;  // (the newline after "thought")
        if (!r.reason.empty()) r.reason += "\n\n";
      } else {
        size_t len = 0;
        const std::string* which = nullptr;
        const size_t k = first_of({&close, &tool}, len, which);
        if (k == npos) {
          const size_t lim = held({&close, &tool});
          if (lim > pos) r.reason.append(text, pos, lim - pos);
          break;
        }
        r.reason.append(text, pos, k - pos);
        pos = k + len;
        if (which == &tool) { r.tool = true; break; }  // (a call from inside the thought block)
        reasoning = false;
        seg_start = true;
      }
    }
    return r;
  }

  void update(const std::string& text, bool final, std::string& d_reason, std::string& d_content) override {
    d_reason.clear();
    d_content.clear();
    const Split sp = split(text, final, in_thought_);
    emit(sp.reason, reason_sent_, d_reason);
    emit(sp.content, content_sent_, d_content);
  }
  // Whole-output split (non-streaming responses): computed from the text alone, independent of update().
  std::string reasoning_text(const std::string& text) const override { return chat::trim(split(text, true, in_thought_).reason); }
  std::string content_text(const std::string& text) const override { return chat::trim(split(text, true, in_thought_).content); }
  std::vector<chat::ToolCall> tool_calls(const std::string& text) const override { return parse_tool_calls(text, tools_); }
  std::vector<chat::PartialToolCall> partial_tool_calls(const std::string& text, bool final) const override {
    return gemma4::partial_tool_calls(text, final, tools_);
  }

 private:
  bool in_thought_ = false;
  Json tools_;
  size_t reason_sent_ = 0, content_sent_ = 0;

  // the new end of a growing part, UTF-8 characters kept whole
  static void emit(const std::string& part, size_t& sent, std::string& out) {
    if (part.size() <= sent) return;
    const size_t t = chat::utf8_end(part, sent, part.size());
    out = part.substr(sent, t - sent);
    sent = t;
  }
};

}  // namespace gemma4
